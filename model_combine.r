library(tidyverse)
library(caret)
library(DMwR)
library(pROC)
library(ROCR)
library(e1071)
library(randomForest)

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

grid <- expand.grid(n.trees = seq(100,4000,500), # number of boosting iteration
                    interaction.depth = seq(5,30,5), # max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 30)  # Min.Terminated node size 
set.seed(12345)
fit.gbm <- train(Sepsis ~ ., data=train_sepsis, 
                 method = 'gbm', 
                 trControl=fitControl, 
                 tuneGrid=grid, 
                 metric = 'Accuracy',
                 verbose = FALSE)
png('cv_gbm_sepsis.png')
plot(fit.gbm)
fit.gbm$bestTune
dev.off()


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
# gbm_sepsis_heatmap
ggsave("gbm_sepsis_heatmap.png", plot = gbm_sepsis_heatmap, units = "in", width=8, height = 6)

# density
trellis.par.set(caretTheme())
png('sepsis_gbm.png')
densityplot(fit.gbm, pch = "|", col = 'black')
dev.off()

######### Choosing the final model
# confusion matrix
gbm_pred_roc = predict(fit.gbm,newdata = test_sepsis,type = 'prob')[,2] %>% as_data_frame()
gbm_pred = predict(fit.gbm,newdata = test_sepsis) %>% as_tibble()
confusionMatrix(data = gbm_pred$value, reference = test_sepsis$Sepsis,mode = "prec_recall")

# roc curve
auc_gbm_sepsis <- roc(test_sepsis$Sepsis, gbm_pred_roc$value)
png('gbm_sepsis_auc.png')
plot(auc_gbm_sepsis,ylim = c(0,1), print.thres = TRUE)
auc_ = round(auc_gbm_sepsis$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz=TRUE)
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

grid <- expand.grid(n.trees = seq(100,4000,500), # number of boosting iteration
                    interaction.depth = seq(5,30,5), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 30)  # Min.Terminated node size 
set.seed(12345)
fit.gbm.shock <- train(SepticShock ~ ., data=train_shock, 
                 method = 'gbm', 
                 trControl=fitControl, 
                 tuneGrid=grid, 
                 metric = 'Accuracy',
                 verbose = FALSE)
fit.gbm.shock$bestTune
plot(fit.gbm.shock)

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
png('shock_gbm.png')
densityplot(fit.gbm.shock, pch = "|",col = 'black')
dev.off()

######### Choosing the final model
# confusion matrix
gbm_pred_roc = predict(fit.gbm.shock,newdata = test_shock,type = 'prob')[,2] %>% as_data_frame()
gbm_pred = predict(fit.gbm.shock,newdata = test_shock) %>% as_tibble()
confusionMatrix(data = gbm_pred$value, reference = test_shock$SepticShock,mode = "prec_recall")

# roc curve
auc_gbm_shock <- roc(test_shock$SepticShock, gbm_pred_roc$value)
png('gbm_shock_auc.png')
plot(auc_gbm_shock,ylim = c(0,1), print.thres = TRUE)
auc_ = round(auc_gbm_shock$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz=TRUE)
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

grid <- expand.grid(n.trees = seq(100,4000,500), # number of boosting iteration
                    interaction.depth = seq(5,30,5), # Max tree depth
                    shrinkage = .001, #learning rate
                    n.minobsinnode = 30)  # Min.Terminated node size 
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
  theme_minimal()+
  ggtitle('gbm_severe_sepsis') +
  theme(plot.title = element_text(hjust=0.5))
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
plot(auc_gbm_severe,ylim = c(0,1), print.thres = TRUE)
auc_ = round(auc_gbm_severe$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz=TRUE)
dev.off()

##############################################################
###Data preprocess
df <- read_rds('df_imputed.rds')
colnames(df)
col <- c('diabetes','kidney','lung', 'asian', 'black','hispanic', 'others', 'unknown', 'white', 'sex','SepticShock', 'Sepsis', 'SeverSepsis')
df[col] = lapply(df[col],factor)
summary(df)
df = df %>% select(-hadm_id)


###Random Forest Model Learning
##1.Sepsis
sepsis = df %>% select(-SepticShock, -SeverSepsis)
sepsis %>%
  group_by(Sepsis) %>% count()

#imbalanced data processing for sepsis: SMOTE
new_sepsis = SMOTE(Sepsis ~.,sepsis, perc.over = 1000, perc.under = 300)
new_sepsis %>%
  group_by(Sepsis) %>% count()
#mean: (440/1640) = 0.2682927

#train-test split
idx = createDataPartition(new_sepsis$Sepsis, p = 0.8, list = FALSE)
train_sepsis = new_sepsis[idx,]
test_sepsis = new_sepsis[-idx,]
train_x_sepsis = train_sepsis %>% select(-Sepsis)
train_y_sepsis = train_sepsis %>% select(Sepsis)
test_x_sepsis = test_sepsis %>% select(-Sepsis)
test_y_sepsis = test_sepsis %>% select(Sepsis)

#Baseline: Naive Bayes
nb_sepsis = naiveBayes(Sepsis ~., data=train_sepsis)
#NB prediction on the dataset
pred_nb_sepsis=predict(nb_sepsis, test_sepsis,type = 'raw')
#Confusion matrix to check accuracy
cm_nb_sepsis = table(pred_nb_sepsis,test_sepsis$Sepsis)
nb_accuracy_sepsis = sum(diag(cm_nb_sepsis))/sum(cm_nb_sepsis)
#roc for NB
auc_nb_sepsis = roc(test_sepsis$Sepsis, pred_nb_sepsis[,2])
auc_nb_sepsis$auc # 0.8439


#Random Forest for Sepsis
result_sepsis = data.frame(tree = double(),
                           node = double(), 
                           accuracy_rate = double())
set.seed(12345)
for (treenum in seq(from = 10, to = 300, by = 10)){#number of tree
  for (node in seq(from = 2, to = 16)){#tune max nodes(indirectly control for the depth)
    rf = randomForest(train_x_sepsis,train_y_sepsis$Sepsis, maxnodes = node, ntree=treenum)
    result_sepsis= rbind(result_sepsis, data.frame(tree = treenum, node = node, accuracy_rate =(sum(diag(rf$confusion)))/sum(rf$confusion)))
  }
}

#Plot heatmap
p1 = ggplot(result_sepsis, aes(result_sepsis[2], result_sepsis[1])) +
  geom_tile(aes(fill = result_sepsis[3])) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7")
rf_sepsis_plot <- p1 + labs(title = "rf_sepsis", x = "number of nodes", y = "number of trees", fill = "Accuracy") +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5))
ggsave('rf_sepsis.png',plot = rf_sepsis_plot, units = "in", width=8, height = 6)

#identify the best parameters
result_sepsis %>%
  filter(accuracy_rate == max(accuracy_rate)) 
#tuned tree with 20 trees and maxnodes of 16
rf_sepsis = randomForest(train_x_sepsis,train_y_sepsis$Sepsis, maxnodes = 16, ntree=20)
png('rf_var_sepsis.png')
varImpPlot(rf_sepsis,main = ' ')
dev.off()

rf_sepsis$importance
CM_sepsis= rf_sepsis$confusion
sum(diag(CM_sepsis))/sum(CM_sepsis) 

#Prediction 
pred.rf.sepsis = predict(rf_sepsis, test_x_sepsis, type = "prob")[,2] %>% as_tibble() 
colnames(pred.rf.sepsis)[1] = "value"
pred.rf.sepsis = pred.rf.sepsis %>%  mutate(truth = test_y_sepsis$Sepsis, rp_guess = value > 0.5)
rf_accuracy_sepsis = pred.rf.sepsis %>% select(-value) %>% table() %>%
  (function(.) sum(diag(.))/sum(.))(.)
rf_accuracy_sepsis 
#ROC 
auc_rf_sepsis = roc(test_y_sepsis$Sepsis, pred.rf.sepsis$value)
print(auc_rf_sepsis)

png('rf_sepsis_auc.png')
plot(auc_rf_sepsis, ylim=c(0,1), print.thres=TRUE)
auc_ = round(auc_rf_sepsis$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz = TRUE)
dev.off()

##################################################################################################################
##2.Servere sepsis
severe = df %>% select(-SepticShock, -Sepsis)
severe %>%
  group_by(SeverSepsis) %>% count()

#imbalanced data processing for servere sepsis: SMOTE
new_severe = SMOTE(SeverSepsis ~.,severe, perc.over = 200, perc.under = 200)
new_severe %>%
  group_by(SeverSepsis) %>% count()
#mean: (651/1519) = 0.4285714

#train-test split
idx = createDataPartition(new_severe$SeverSepsis, p = 0.8, list = FALSE)
train_severe = new_severe[idx,]
test_severe = new_severe[-idx,]
train_x_severe = train_severe %>% select(-SeverSepsis)
train_y_severe = train_severe %>% select(SeverSepsis)
test_x_severe = test_severe %>% select(-SeverSepsis)
test_y_severe = test_severe %>% select(SeverSepsis)

#Baseline: Naive Bayes
nb_severe = naiveBayes(SeverSepsis ~., data=train_severe)
nb_severe
#NB prediction on the dataset
pred_nb_severe=predict(nb_severe, test_severe,type = 'raw')
#Confusion matrix to check accuracy
cm_nb_severe = table(pred_nb_severe,test_severe$SeverSepsis)
nb_accuracy_severe = sum(diag(cm_nb_severe))/sum(cm_nb_severe) 
severe_roc = roc(test_severe$SeverSepsis, pred_nb_severe[,2])
severe_roc$auc # 0.72

#Random Forest
result_severe = data.frame(tree = double(),
                           node = double(), 
                           accuracy_rate = double())
set.seed(12345)
for (treenum in seq(from = 10, to =300, by = 10)){#number of tree
  for (node in seq(from = 2, to = 16)){#tune max nodes(indirectly control for the depth)
    rf = randomForest(train_x_severe,train_y_severe$SeverSepsis, maxnodes = node, ntree=treenum)
    result_severe= rbind(result_severe, data.frame(tree = treenum, node = node, accuracy_rate =(sum(diag(rf$confusion)))/sum(rf$confusion)))
  }
}

#Plot heatmap
p2 = ggplot(result_severe, aes(result_severe[2], result_severe[1])) +
  geom_tile(aes(fill = result_severe[3])) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7")
rf_severe_plot <- p2 + labs(title = "rf_severe_sepsis", x = "number of nodes", y = "number of trees", fill = "Accuracy")+
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5))
ggsave('rf_severe.png',plot = rf_severe_plot, units = "in", width=8, height = 6)



#identify the best parameters
result_severe %>%
  filter(accuracy_rate == max(accuracy_rate))
#tuned tree with 290 trees and maxnodes of 16
rf_severe = randomForest(train_x_severe,train_y_severe$SeverSepsis, maxnodes = 16, ntree=290)
png('rf_var_severe.png')
varImpPlot(rf_severe,main = ' ')
dev.off()


rf_severe$importance
CM_severe= rf_severe$confusion
sum(diag(CM_severe))/sum(CM_severe)
#Prediction 
pred.rf.severe = predict(rf_severe, test_x_severe, type = "prob")[,2] %>% as_tibble() 
colnames(pred.rf.severe)[1] = "value"
pred.rf.severe = pred.rf.severe %>%  mutate(truth = test_y_severe$SeverSepsis, rp_guess = value > 0.5)
rf_accuracy_severe = pred.rf.severe %>% select(-value) %>% table() %>%
  (function(.) sum(diag(.))/sum(.))(.)
rf_accuracy_severe 
#ROC 
auc_rf_severe = roc(test_y_severe$SeverSepsis, pred.rf.severe$value)
print(auc_rf_severe)
png('rf_severe_auc.png')
plot(auc_rf_severe, ylim=c(0,1), print.thres=TRUE)
auc_ = round(auc_rf_severe$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz = TRUE)
dev.off()


###################################################################################################################
##3.Septic shock
shock = df %>% select(-SeverSepsis, -Sepsis)
shock %>%
  group_by(SepticShock) %>% count()

#imbalanced data processing for septic shock: SMOTE
new_shock = SMOTE(SepticShock ~.,shock, perc.over = 350, perc.under = 350)
new_shock %>%
  group_by(SepticShock) %>% count()
#mean: (436/1580) = 0.2759494

#train-test split
idx = createDataPartition(new_shock$SepticShock, p = 0.8, list = FALSE)
train_shock = new_shock[idx,]
test_shock = new_shock[-idx,]
train_x_shock = train_shock %>% select(-SepticShock)
train_y_shock = train_shock %>% select(SepticShock)
test_x_shock = test_shock %>% select(-SepticShock)
test_y_shock = test_shock %>% select(SepticShock)

#Baseline: Naive Bayes
nb_shock = naiveBayes(SepticShock ~., data=train_shock)
nb_shock
#NB prediction on the dataset
pred_nb_shock=predict(nb_shock, test_shock,type = 'raw')
#Confusion matrix to check accuracy
cm_nb_shock = table(pred_nb_shock,test_shock$SepticShock)
nb_accuracy_shock = sum(diag(cm_nb_shock))/sum(cm_nb_shock) 
septic_roc <- roc(test_shock$SepticShock,pred_nb_shock[,2])
septic_roc$auc # 0.80

#Random Forest
result_shock = data.frame(tree = double(),
                          node = double(), 
                          accuracy_rate = double())
set.seed(12345)
for (treenum in seq(from = 10, to =300, by = 10)){#number of tree
  for (node in seq(from = 2, to = 16)){#tune max nodes(indirectly control for the depth)
    rf = randomForest(train_x_shock,train_y_shock$SepticShock, maxnodes = node, ntree=treenum)
    result_shock= rbind(result_shock, data.frame(tree = treenum, node = node, accuracy_rate =(sum(diag(rf$confusion)))/sum(rf$confusion)))
  }
}
#Plot heatmap
p3 = ggplot(result_shock, aes(result_shock[2], result_shock[1])) +
  geom_tile(aes(fill = result_shock[3])) +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7")
rf_shock_plot <- p3 + labs(title = "rf_septic_shock", x = "number of nodes", y = "number of trees", fill = "Accuracy")+
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))
ggsave('rf_shock.png',plot = rf_shock_plot,unit = 'in',width = 8, height = 6)


#identify the best parameters
result_shock %>%
  filter(accuracy_rate == max(accuracy_rate))
#tuned tree with 40 trees and maxnodes of 16
rf_shock = randomForest(train_x_shock,train_y_shock$SepticShock, maxnodes = 16, ntree=40)


png('rf_var_shock.png')
varImpPlot(rf_shock,main = ' ')
dev.off()

rf_shock$importance
CM_shock= rf_shock$confusion
sum(diag(CM_shock))/sum(CM_shock)

#Prediction 
pred.rf.shock = predict(rf_shock, test_x_shock, type = "prob")[,2] %>% as_tibble() 
colnames(pred.rf.shock)[1] = "value"
pred.rf.shock = pred.rf.shock %>%  mutate(truth = test_y_shock$SepticShock, rp_guess = value > 0.5)
rf_accuracy_shock = pred.rf.shock %>% select(-value) %>% table() %>%
  (function(.) sum(diag(.))/sum(.))(.)
rf_accuracy_shock 
#ROC 
auc_rf_shock = roc(test_y_shock$SepticShock, pred.rf.shock$value)
print(auc_rf_shock)
png('rf_shock_auc.png')
plot(auc_rf_shock, ylim=c(0,1), print.thres=TRUE)
auc_ = round(auc_rf_shock$auc,2)
legend('bottomright',legend = c('AUC:',auc_),horiz = TRUE)
dev.off()

#############################################################################################################################
###Summary for random forest
rf_summary = data.frame(target = character(),
                        NB_accuracy = double(), 
                        RF_accuracy = double(),
                        AUC = double())
rf_summary = rbind(rf_summary, data.frame(target = c("Sepsis"), NB_accuracy = nb_accuracy_sepsis, RF_accuracy = rf_accuracy_sepsis, AUC = auc_rf_sepsis$auc))
rf_summary = rbind(rf_summary, data.frame(target = c("Severe Sepsis"), NB_accuracy = nb_accuracy_severe, RF_accuracy = rf_accuracy_severe, AUC = auc_rf_severe$auc))
rf_summary = rbind(rf_summary, data.frame(target = c("Septic Shock"), NB_accuracy = nb_accuracy_shock, RF_accuracy = rf_accuracy_shock, AUC = auc_rf_shock$auc))
rf_summary = rf_summary %>% as_tibble()
rf_summary
#https://www.r-bloggers.com/understanding-naive-bayes-classifier-using-r/
#https://stackoverflow.com/questions/34997134/random-forest-tuning-tree-depth-and-number-of-trees?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa









