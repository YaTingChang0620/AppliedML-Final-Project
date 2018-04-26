library(tidyverse)
library(caret)
library(DMwR)
# library(pROC)
# library(ROCR)

df <- read_rds('df_imputed.rds')
col <- c('diabetes','kidney','lung', 'asian', 'black','hispanic', 'others', 'unknown', 'white', 'sex','SepticShock', 'Sepsis', 'SeverSepsis')
df[col] = lapply(df[col],factor)

######### SMOTE
# source: 
sepsis <- df %>%
  select(-SepticShock,-SeverSepsis)
new_sepsis = DMwR::SMOTE(Sepsis ~.,sepsis, perc.over = 1000, perc.under = 200)
levels(new_sepsis$Sepsis) <- c('N','Y') # N=0, Y=1 in the original data


######### Train-Test split
splitIndex <- createDataPartition(new_sepsis$Sepsis, p = 0.8, list = FALSE) 
train <- new_sepsis[splitIndex,]
test <- new_sepsis[-splitIndex,]

######### GBM
fitControl <- trainControl(method = 'cv', 
                           number = 5) # 5 folds

grid <- expand.grid(n.trees = seq(100,5000,100), # number of boosting iteration
                    interaction.depth = seq(2,10,1), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 20)  # Min.Terminated node size 

fit.gbm <- train(Sepsis ~ ., data=train, 
                 method = 'gbm', 
                 trControl=fitControl, 
                 tuneGrid=grid, 
                 metric = 'Accuracy',
                 verbose = FALSE)

######### Within model evaluation
# heatmap
fit.gbm$results %>%
  select(interaction.depth,n.trees,Accuracy) %>%
  ggplot(aes(x = interaction.depth, y = n.trees )) +
  geom_tile(aes(fill = Accuracy)) +
  xlab('Tree Depth') +
  ylab('# boosting iteration') +
  theme_minimal()

# plot(fit.gbm)

# density
trellis.par.set(caretTheme())
densityplot(fit.gbm, pch = "|")

######### Choosing the final model
# confusion matrix
gbm_pred = predict(fit.gbm,newdata = test) %>% as_data_frame()
confusionMatrix(data = gbm_pred$value, reference = test$Sepsis,mode = "prec_recall")

# roc curve

  

