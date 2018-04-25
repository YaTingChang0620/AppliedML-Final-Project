library(tidyverse)
library(mice) 
library(dplyr)

df <- read_rds('cleaning_final.rds')
df = df %>% as.data.frame()
labels = df %>% select(SepticShock, Sepsis, SeverSepsis) 
#Check which columns have missing values
apply(df, FUN = (function(x) any(is.na(x))), MARGIN = 2) %>% as.data.frame()
#Missing value imputation with 10 imputations for 20 iteration.
mdat = mice(df %>% 
              select(-hadm_id, -SepticShock, -Sepsis, -SeverSepsis)%>% 
              mutate_if(is.character, as.factor), m = 10, maxit = 20) 

imputed = complete(mdat) %>% as_tibble()
imputed = data.frame(hadm_id = df$hadm_id) %>% bind_cols(imputed) %>% as_tibble()
imputed = cbind.data.frame(imputed, labels)
write_rds(imputed, 'df_imputed.rds')

#https://stats.stackexchange.com/questions/219013/how-do-the-number-of-imputations-the-maximum-iterations-affect-accuracy-in-mul?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

