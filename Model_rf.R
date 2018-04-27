setwd("/Users/michellehsu/Desktop/AAMLP/AppliedML-Final-Project")
loadlibs = function(libs) {
  for(lib in libs) {
    class(lib)
    if(!do.call(require,as.list(lib))) {install.packages(lib)}
    do.call(require,as.list(lib))
  }
}
libs = c("tidyverse","randomForest","caret","pROC","ROCR","DMwR","e1071")
loadlibs(libs)

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
  group_by(Sepsis) %>% 
  count() %>%
  mutate(percentage = n/nrow(sepsis))

#imbalanced data processing for sepsis: SMOTE
new_sepsis = SMOTE(Sepsis ~.,sepsis, perc.over = 1000, perc.under = 300)
new_sepsis %>%
  group_by(Sepsis) %>% 
  count() %>%
  mutate(percentage = n/nrow(new_sepsis))

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
nb_sepsis
#NB prediction on the dataset
pred_nb_sepsis=predict(nb_sepsis, test_sepsis)
#Confusion matrix to check accuracy
cm_nb_sepsis = table(pred_nb_sepsis,test_sepsis$Sepsis)
nb_accuracy_sepsis = sum(diag(cm_nb_sepsis))/sum(cm_nb_sepsis)

#Random Forest for Sepsis
result_sepsis = data.frame(tree = double(),
               node = double(), 
               accuracy_rate = double())
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
p1 + labs(title = "Random Forest (Sepsis)", x = "number of nodes", y = "number of trees", fill = "Accuracy")

#identify the best parameters
result_sepsis %>%
  filter(accuracy_rate == max(accuracy_rate)) 
#tuned tree with 50 trees and maxnodes of 15
rf_sepsis = randomForest(train_x_sepsis,train_y_sepsis$Sepsis, maxnodes = 15, ntree=50)
varImpPlot(rf_sepsis, main = "Feature Importance")
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
plot(auc_rf_sepsis, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_rf_sepsis$auc[[1]],2)))
##################################################################################################################
##2.Servere sepsis
severe = df %>% select(-SepticShock, -Sepsis)
severe %>%
  group_by(SeverSepsis) %>% 
  count() %>%
  mutate(percentage = n/nrow(severe))

#imbalanced data processing for servere sepsis: SMOTE
new_severe = SMOTE(SeverSepsis ~.,severe, perc.over = 200, perc.under = 200)
new_severe %>%
  group_by(SeverSepsis) %>% 
  count() %>%
  mutate(percentage = n/nrow(new_severe))

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
pred_nb_severe=predict(nb_severe, test_severe)
#Confusion matrix to check accuracy
cm_nb_severe = table(pred_nb_severe,test_severe$SeverSepsis)
nb_accuracy_severe = sum(diag(cm_nb_severe))/sum(cm_nb_severe) 

#Random Forest
result_severe = data.frame(tree = double(),
                           node = double(), 
                           accuracy_rate = double())
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
p2 + labs(title = "Random Forest (Severe Sepsis)", x = "number of nodes", y = "number of trees", fill = "Accuracy")

#identify the best parameters
result_severe %>%
  filter(accuracy_rate == max(accuracy_rate))
#tuned tree with 290 trees and maxnodes of 16
rf_severe = randomForest(train_x_severe,train_y_severe$SeverSepsis, maxnodes = 16, ntree=290)
varImpPlot(rf_severe)
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
plot(auc_rf_severe, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_rf_severe$auc[[1]],2)))
###################################################################################################################
##3.Septic shock
shock = df %>% select(-SeverSepsis, -Sepsis)
shock %>%
  group_by(SepticShock) %>%
  count() %>%
  mutate(percentage = n/nrow(shock))

#imbalanced data processing for septic shock: SMOTE
new_shock = SMOTE(SepticShock ~.,shock, perc.over = 350, perc.under = 350)
new_shock %>%
  group_by(SepticShock) %>% 
  count() %>%
  mutate(percentage = n/nrow(new_shock))
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
pred_nb_shock=predict(nb_shock, test_shock)
#Confusion matrix to check accuracy
cm_nb_shock = table(pred_nb_shock,test_shock$SepticShock)
nb_accuracy_shock = sum(diag(cm_nb_shock))/sum(cm_nb_shock) 

#Random Forest
result_shock = data.frame(tree = double(),
                           node = double(), 
                           accuracy_rate = double())
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
p3 + labs(title = "Random Forest (Septic Shock)", x = "number of nodes", y = "number of trees", fill = "Accuracy")

#identify the best parameters
result_shock %>%
  filter(accuracy_rate == max(accuracy_rate))
#tuned tree with 40 trees and maxnodes of 16
rf_shock = randomForest(train_x_shock,train_y_shock$SepticShock, maxnodes = 16, ntree=40)
varImpPlot(rf_shock)
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
plot(auc_rf_shock, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_rf_shock$auc[[1]],2)))
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