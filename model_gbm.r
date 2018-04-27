library(tidyverse)
library(caret)
library(DMwR)
library(pROC)
library(ROCR)

df <- read_rds('df_imputed.rds')

col <- c('diabetes','kidney','lung', 'asian', 'black','hispanic', 'others', 'unknown', 'white', 'sex','SepticShock', 'Sepsis', 'SeverSepsis')
df[col] = lapply(df[col],factor)


######################## SEPSIS ######################## 
######### SMOTE
# source: 
sepsis <- df %>%
  select(-SepticShock,-SeverSepsis)
new_sepsis = DMwR::SMOTE(Sepsis ~.,sepsis, perc.over = 1000, perc.under = 300)
levels(new_sepsis$Sepsis) <- c('N','Y') # N=0, Y=1 in the original data


######### Train-Test split
splitIndex <- createDataPartition(new_sepsis$Sepsis, p = 0.8, list = FALSE) 
train_sepsis <- new_sepsis[splitIndex,]
test_sepsis <- new_sepsis[-splitIndex,]

######### GBM
fitControl <- trainControl(method = 'cv', 
                           number = 5) # 5 folds

grid <- expand.grid(n.trees = seq(100,5000,100), # number of boosting iteration
                    interaction.depth = seq(2,10,1), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 20)  # Min.Terminated node size 
set.seed(12345)
fit.gbm <- train(Sepsis ~ ., data=train_sepsis, 
                 method = 'gbm', 
                 trControl=fitControl, 
                 tuneGrid=grid, 
                 metric = 'Accuracy',
                 verbose = FALSE)

######### Within model evaluation
# heatmap
gbm_sepsis_heatmap <- fit.gbm$results %>%
  select(interaction.depth,n.trees,Accuracy) %>%
  ggplot(aes(x = interaction.depth, y = n.trees )) +
  geom_tile(aes(fill = Accuracy)) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7") + 
  xlab('Tree Depth') +
  ylab('# boosting iteration') +
  theme_minimal() +
  ggtitle('gbm_sepsis') +
  theme(plot.title = element_text(hjust=0.5))
ggsave("gbm_sepsis_heatmap.png", plot = gbm_sepsis_heatmap, units = "in", width=8, height = 6)

# density
trellis.par.set(caretTheme())
densityplot(fit.gbm, pch = "|")

######### Choosing the final model
# confusion matrix
gbm_pred_roc = predict(fit.gbm,newdata = test_sepsis,type = 'prob')[,2] %>% as_data_frame()
gbm_pred = predict(fit.gbm,newdata = test_sepsis) %>% as_tibble()
confusionMatrix(data = gbm_pred$value, reference = test_sepsis$Sepsis,mode = "prec_recall")

# roc curve
auc_gbm_sepsis <- roc(test_sepsis$Sepsis, gbm_pred_roc$value)
png('gbm_sepsis_auc.png')
plot(auc_gbm_sepsis,ylim = c(0,1), print.thres = TRUE, main = paste('AUC:', round(auc_gbm_sepsis$auc[[1]],2)))
dev.off()


######################## SEPTIC SHOCK ######################## 

septicshock <- df %>%
  select(-Sepsis,-SeverSepsis)
septicshock %>% colnames()
new_septicshock = DMwR::SMOTE(SepticShock ~.,septicshock, perc.over = 350, perc.under = 350)
levels(new_septicshock$SepticShock) <- c('N','Y') # N=0, Y=1 in the original data


######### Train-Test split
splitIndex <- createDataPartition(new_septicshock$SepticShock, p = 0.8, list = FALSE) 
train_shock <- new_septicshock[splitIndex,]
test_shock <- new_septicshock[-splitIndex,]

######### GBM
fitControl <- trainControl(method = 'cv', 
                           number = 5) # 5 folds

grid <- expand.grid(n.trees = seq(100,5000,100), # number of boosting iteration
                    interaction.depth = seq(2,10,1), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 20)  # Min.Terminated node size 
set.seed(12345)
fit.gbm.shock <- train(SepticShock ~ ., data=train_shock, 
                 method = 'gbm', 
                 trControl=fitControl, 
                 tuneGrid=grid, 
                 metric = 'Accuracy',
                 verbose = FALSE)

######### Within model evaluation
# heatmap
gbm_shock_heatmap <- fit.gbm.shock$results %>%
  select(interaction.depth,n.trees,Accuracy) %>%
  ggplot(aes(x = interaction.depth, y = n.trees )) +
  geom_tile(aes(fill = Accuracy)) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7") + 
  xlab('Tree Depth') +
  ylab('# boosting iteration') +
  theme_minimal()+
  ggtitle('gbm_septic_shock') +
  theme(plot.title = element_text(hjust=0.5))
ggsave("gbm_shock_heatmap.png", plot = gbm_shock_heatmap, units = "in", width=8, height = 6)


# density
trellis.par.set(caretTheme())
densityplot(fit.gbm.shock, pch = "|")

######### Choosing the final model
# confusion matrix
gbm_pred_roc = predict(fit.gbm.shock,newdata = test_shock,type = 'prob')[,2] %>% as_data_frame()
gbm_pred = predict(fit.gbm.shock,newdata = test_shock) %>% as_tibble()
confusionMatrix(data = gbm_pred$value, reference = test_shock$SepticShock,mode = "prec_recall")

# roc curve
auc_gbm_shock <- roc(test_shock$SepticShock, gbm_pred_roc$value)
png('gbm_shock_auc.png')
plot(auc_gbm_shock,ylim = c(0,1), print.thres = TRUE, main = paste('AUC:', round(auc_gbm_shock$auc[[1]],2)))
dev.off()


######################## Severe Sepsis ######################## 
severe <- df %>%
  select(-Sepsis,-SepticShock)
severe %>% colnames()
new_severe = DMwR::SMOTE(SeverSepsis ~.,severe, perc.over = 200, perc.under = 200)
levels(new_severe$SeverSepsis) <- c('N','Y') # N=0, Y=1 in the original data


######### Train-Test split
splitIndex <- createDataPartition(new_severe$SeverSepsis, p = 0.8, list = FALSE) 
train_severe <- new_severe[splitIndex,]
test_severe <- new_severe[-splitIndex,]

######### GBM
fitControl <- trainControl(method = 'cv', 
                           number = 5) # 5 folds

grid <- expand.grid(n.trees = seq(100,5000,100), # number of boosting iteration
                    interaction.depth = seq(2,10,1), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 20)  # Min.Terminated node size 
set.seed(12345)
fit.gbm.severe <- train(SeverSepsis ~ ., data=train_severe, 
                       method = 'gbm', 
                       trControl=fitControl, 
                       tuneGrid=grid, 
                       metric = 'Accuracy',
                       verbose = FALSE)

######### Within model evaluation
# heatmap
gbm_severe_heatmap <- fit.gbm.severe$results %>%
  select(interaction.depth,n.trees,Accuracy) %>%
  ggplot(aes(x = interaction.depth, y = n.trees )) +
  geom_tile(aes(fill = Accuracy)) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7") + 
  xlab('Tree Depth') +
  ylab('# boosting iteration') +
  theme_minimal()
ggsave("gbm_severe_heatmap.png", plot = gbm_severe_heatmap, units = "in", width=8, height = 6)


# density
trellis.par.set(caretTheme())
densityplot(fit.gbm.shock, pch = "|")

######### Choosing the final model
# confusion matrix
gbm_pred_roc = predict(fit.gbm.severe,newdata = test_severe,type = 'prob')[,2] %>% as_data_frame()
gbm_pred = predict(fit.gbm.severe,newdata = test_severe) %>% as_tibble()
confusionMatrix(data = gbm_pred$value, reference = test_severe$SeverSepsis,mode = "prec_recall")

# roc curve
auc_gbm_severe <- roc(test_severe$SeverSepsis, gbm_pred_roc$value)
png('gbm_severe_auc.png')
plot(auc_gbm_severe,ylim = c(0,1), print.thres = TRUE, main = paste('AUC:', round(auc_gbm_severe$auc[[1]],2)))
dev.off()

##############################################################












