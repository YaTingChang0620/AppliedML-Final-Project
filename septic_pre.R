library(tidyverse)
library(data.table)
library(rowr)
setwd("/Users/michellehsu/Desktop/Sepsis") 
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
### 
# chart <- read_csv('d_chartitems.csv')
# labitem <- read_csv("d_labitems.csv")
# micro <- read_csv('microbiologyevents.csv')
# meditem <- read_csv('d_meditems.csv')
# drgevent <- read_csv('drgevents.csv')
# medevent <- read_csv('medevents.csv')
# procedure <- read_csv('procedureevents.csv')

# read chronic disease indicator 
# (source: https://www.hcup-us.ahrq.gov/toolssoftware/chronic/chronic.jsp)
# 0 - non-chronic; 1 - chronic
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
  unique() %>%  # account for one patient with two kinds of diabetes/kidney/lung
  mutate(n = 1) %>% spread(key = disease, value = n, fill = 0) 

hadm <- pt_chronic_one %>% select(hadm_id)





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
                            ethnicity_descr %in% black ~'blacd',
                            ethnicity_descr %in% asian ~'asian',
                            ethnicity_descr %in% hispanic ~'hispanic',
                            ethnicity_descr %in% others ~'others',
                            ethnicity_descr %in% unknown ~'unknown')) %>%
  select(subject_id,hadm_id,ethnic) %>%
  mutate(n = 1)

demo_one <- demo_ethnic %>% spread(key=ethnic,value = n,fill = 0) %>% select(-subject_id)

# check one person belongs to one race 
# demo_one %>%
#   select(-hadm_id) %>% 
#   mutate(t = rowSums(.)) %>% select(t) %>% filter(t == 1)

#### FINAL : join
pt_chronic_one <- pt_chronic_one %>% left_join(demo_one,by='hadm_id')

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
pt_chronic_one <- pt_chronic_one %>% left_join(age.gender,by='hadm_id')






########### Feature: vital sign  ########### 
# blood pressure(52), heart rate(211), respiratory rate(618)
# oxygen saturation (834), temperature(678)

icu <- icustay_detail %>% select(hadm_id,icustay_id) # a hamd_id may correspond to many icustay_id
vital_hadm <- vital %>% left_join(icu,by='icustay_id') # vital doesn't have hadm_id

extract_vital <- function(id,vitalname){
  temp <- vital_hadm %>% filter(!is.na(itemid) & itemid == id) %>%
    group_by(hadm_id) %>%
    summarise(n = mean(value1num,na.rm=TRUE))
  colnames(temp) <- c('hadm_id',vitalname)
  return(temp)
}

items <- list(456,211,618,646,678)
signs <- list('blood.pressure','heart.rate','respiratory.rate',
              ' oxygen.saturation','temperature')
temp <- as_data_frame()

for(i in 1:len(items)){
  if(i==1){
    # print(i,'',signs[[i]])
    temp = extract_vital(items[[i]],signs[[i]])
    vitalsign <- hadm %>% left_join(temp,by='hadm_id')
  }else{
    temp = extract_vital(items[[i]],signs[[i]])
    vitalsign <- vitalsign %>% left_join(temp,by='hadm_id')
  }
}
apply(vitalsign,2,function(x){sum(is.na(x))/1590})
pt_chronic_one <- pt_chronic_one %>% left_join(vitalsign,by='hadm_id')

write_rds(pt_chronic_one,'cleaning_final.rds')


#################### THINGS TO DO
# 1. all data frames' column name
# 2. argument to summarise's column name