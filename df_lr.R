library(tidyverse)
library(data.table)
library(magrittr)

# read files
icd9 <- read_csv('icd9.csv') 
demo <- read_csv('demographic_detail.csv')
codeitem <- read_csv("d_codeditems.csv")
labevent <- fread('labevents.csv') %>% as_tibble()
icustay_detail <- read_csv('icustay_detail.csv')
demo_item <- read_csv('d_demographicitems.csv')
demo_event <- read_csv('demographicevents.csv')
demo_patient <- read_csv('d_patients.csv')
vital <- read_rds('vital_sign.rds')
chronic <- read_csv('cci2015.csv',col_names = TRUE)
colnames(chronic) <- c('code','description','chronic_i','body_system')
# remove '' and whitespace
for (i in colnames(chronic)){
  chronic[[i]] = str_replace_all(string=chronic[[i]],pattern = '\'',replacement = '')
  chronic[[i]] =  str_trim(chronic[[i]],side = 'both')
}

########### collect codes of chronic disease ########### 
icd9_unique <- icd9$code %>% unique()

# extract disease code in chronic data frame
extract_code <- function(disease){
  d <- chronic %>% filter(chronic_i == 1 & 
                            str_detect(description,disease)) %>%
    pull(code)
  
  idx <- icd9_unique %>%
    str_replace(pattern = '\\.', replacement = '') %in% d
  
  result <- icd9_unique[idx]
  return(result)
}

# 1. diabetes 
diabetes <- extract_code('DIABETES')
# 2. lung
lung <- extract_code('LUNG')
# 3. kidney disease 
kidney <- extract_code('KIDNEY')
# 4. combine three chronic disease
chronic <- c(kidney,diabetes,lung)
# 5. sepsis, severe sepsis, septic shock
target <- c(995.91,995.92,785.52)


########### Feature: chronic disease ########### 
# filter hospitalization with chronic disease 
pt_chronic <- icd9 %>%
  group_by(hadm_id) %>%
  filter(any(code %in% chronic))

pt_chronic_one <- pt_chronic %>%
  filter(code %in% diabetes | code %in% lung | code %in% kidney) %>%
  mutate(disease = case_when(code %in% diabetes~'diabetes',
                             code %in% lung ~ 'lung',
                             TRUE ~ 'kidney')) %>%
  select(hadm_id,disease) %>% 
  unique() # account for one patient with two kinds of diabetes/kidney/lung

########### outcome: severe sepsis, septic shock ########### 
# Three targets
# for each hospitalization, one hot encoding for three targets
pt_target <- pt_chronic %>%
  filter(code %in% target) %>%
  select(hadm_id,code) %>%
  mutate(n = 1) %>% spread(key=code,value = n, fill = 0) %>%
  rename('SepticShock' = `785.52`,'SeverSepsis'=`995.92`,'Sepsis' = `995.91`)

#### FINAL : join
pt_chronic_one %>% is.na() %>% sum() # make sure no NA in the original data 
pt_chronic_one <- pt_chronic_one %>% left_join(pt_target,by='hadm_id')
pt_chronic_one <- pt_chronic_one  %>% replace_na(list('SepticShock' = 0,'SeverSepsis' =0,'Sepsis'=0))


########### Feature: demongraphic ########### 
white <- c('WHITE','WHITE - RUSSIAN')
black <- c('BLACK/AFRICAN AMERICAN','BLACK/CAPE VERDEAN','BLACK/HAITIAN')
asian <- c('ASIAN','ASIAN - CHINESE','ASIAN - VIETNAMESE')
hispanic <- c('HISPANIC OR LATINO','HISPANIC/LATINO - PUERTO RICAN')
others <- c('OTHER','AMERICAN INDIAN/ALASKA NATIVE','MULTI RACE ETHNICITY',
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLAND')
unknown <- c('UNKNOWN/NOT SPECIFIED','PATIENT DECLINED TO ANSWER','UNABLE TO OBTAIN')

demo_ethnic <- demo %>%
  mutate(ethnic = case_when(ethnicity_descr %in% white ~'white',
                            ethnicity_descr %in% black ~'black',
                            ethnicity_descr %in% asian ~'asian',
                            ethnicity_descr %in% hispanic ~'hispanic',
                            ethnicity_descr %in% others ~'others',
                            ethnicity_descr %in% unknown ~'unknown')) %>%
  select(hadm_id,ethnic) 

#### FINAL : join
pt_chronic_one <- pt_chronic_one %>% left_join(demo_ethnic,by='hadm_id')

# extract year of charttime
charttime <- labevent %>%
  filter(!is.na(hadm_id)) %>%
  mutate(charttime = lubridate::as_datetime(charttime)) %>%
  mutate(year_chart = year(charttime)) %>%
  select(subject_id,hadm_id,year_chart) %>% unique() 
# account for one hadm_id corresponds to multiple icu stays

# extract dod
dob.gender <- demo_patient %>% select(subject_id,sex,dob)

# age
age <- charttime %>% left_join(dob.gender,by='subject_id')
age.gender<- age %>% mutate(dob = year(dob),
                            age = year_chart - dob) %>%
  select(hadm_id,sex,age)

#### FINAL : join
pt_chronic_one <- pt_chronic_one %>%
  left_join(age.gender,by='hadm_id')


##############################################
# check missing value
apply(pt_chronic_one,2,function(x){sum(is.na(x))})

pt_chronic_one %>%
  group_by(sex) %>%
  summarise(n = n()) #male
median(pt_chronic_one$age,na.rm = TRUE) #72

# impute missing value 
pt_chronic_one <- pt_chronic_one %>% replace_na(list(sex = 'M',age = 72))
str(pt_chronic_one)

################################################
# transform data type
pt_chronic_one %>% colnames()
f <- c('disease',"SepticShock",'Sepsis','SeverSepsis','ethnic','sex')
pt_chronic_one[f] <- lapply(pt_chronic_one[f],factor)
# pt_chronic_one %<>%
#   mutate_at(f,funs(factor(.)))


write_rds(pt_chronic_one,'df_lr.rds')


