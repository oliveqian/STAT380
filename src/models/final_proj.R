library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ggplot2)
library(ClusterR)
library(dplyr)
#library(plyr)

library(xgboost)

#feature engineering
raw_data <- fread("./project/volume/data/raw/training_data.csv")
raw_data_for_merge <- fread("./project/volume/data/raw/training_data.csv")

raw_data = raw_data %>%
  rename(.,'0' = subredditcars) %>%
  rename(.,'1' = subredditCooking) %>%
  rename(.,'3' = subredditmagicTCG) %>%
  rename(.,'2' = subredditMachineLearning) %>%
  rename(.,'4' = subredditpolitics) %>%
  rename(.,'5' = subredditReal_Estate) %>%
  rename(.,'6' = subredditscience) %>%
  rename(.,'7' = subredditStockMarket) %>%
  rename(.,'8' = subreddittravel) %>%
  rename(.,'9' = subredditvideogames) 
  
raw_data_processed <- melt(raw_data, id = c("id","text"))
raw_data_processed <- raw_data_processed[which(raw_data_processed$value==1)]
#drop the value variable
raw_data_processed <- select(raw_data_processed,1:3)

#make master file
train_embedding <- fread("./project/volume/data/raw/train_emb.csv")
test_embedding <- fread("./project/volume/data/raw/test_emb.csv")
test <- fread("./project/volume/data/raw/test_file.csv")
#library(plyr)
raw_data_for_merge$num <- 1: nrow(raw_data_for_merge)
train <- merge(raw_data_for_merge, raw_data_processed, by = c("id","text"))
train <- train[order(train$num)]
train <- train[,c(1:2,14)]

train_data_with_embedding <- cbind(train, train_embedding)
test_data_with_embedding <- cbind(test, test_embedding)
test_data_with_embedding$variable = 11

master <- rbind(train_data_with_embedding, test_data_with_embedding)

#PCA and tsne
label <- master$variable
text <- master$text
id <- master$id

master_process <- master
master_process$variable = NULL
master_process <- master_process[,-c(1,2)]

pca<-prcomp(master_process)
pca_dt<-data.table(unclass(pca)$x)



#tune perplexity later
tsne<-Rtsne(pca_dt,pca = F,perplexity=20,check_duplicates = FALSE)

#pca_dt$label <- label

tsne_dt<-data.table(tsne$Y)
tsne_dt = tsne_dt%>%
  rename(.,tsne1 = V1)%>%
  rename(.,tsne2 = V2)
tsne_dt$label <- label
tsne_dt_train = tsne_dt[1:200,]
tsne_dt_test = tsne_dt[201:nrow(tsne_dt),]
#xgboost
#train model
train_y = tsne_dt_train$label
train_x = tsne_dt_train[,c(1:2)]
test_x = tsne_dt_test[,c(1:2)]

train_y = as.matrix(train_y)
train_x = as.matrix(train_x)
test_x = as.matrix(test_x)


dtrain <- xgb.DMatrix(train_x,label=train_y,missing=NA)
dtest <- xgb.DMatrix(test_x)

param <- list(  objective           = "multi:softprob",
                gamma               =0.02,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.02,
                max_depth           = 40,
                min_child_weight    = 1,
                num_class           = 10,
                subsample           = 0.8,
                colsample_bytree    = 1.0,
                tree_method = 'hist'
)
watchlist <- list(train= dtrain)

XGBm<-xgb.cv( params=param,nfold=5,nrounds=3000,missing=NA,data=dtrain,print_every_n=1)
xgbm <- xgb.train( params=param,nrounds=3000,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)

pred <- predict(xgbm, newdata = dtest)
prediction <- matrix(pred, ncol = 10, byrow = TRUE)
prediction <- data.table(prediction)

fwrite(prediction,"./project/volume/data/processed/submit2.csv")




