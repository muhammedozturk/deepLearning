performance.calculation <- function(outcome, prob, threshold=0.5){

  m1 <- c(AUC=as.numeric(auc(outcome, prob)))

  classify <- ifelse(prob > threshold, TRUE, FALSE)
  m2 <- table(classify,outcome)

  tp <- fp <- fn <- tn <- NULL
  if(length(table(classify)) == 1){
    tp <- 0; fp <- 0; fn <- m2[1,2]; tn <- m2[1,1]
  }else{
    tp <- m2[2,2]; fp <- m2[2,1]; fn <- m2[1,2]; tn <- m2[1,1]
  }

  recall <- tp/(tp+fn)
  precision <- tp/(tp+fp)
  fpr <- fp/(fp+tn)
  tnr <- tn/(tn+fp)
  fmeasure <- 2*precision*recall/(precision+recall)
  accuracy <- (tn+tp)/(tn+fn+fp+tp)

  mcc.prod <- log2(tp + fp) + log2(tp + fn) + log2(tn + fp) + log2(tn + fn)
  mcc <- (tp * tn - fp * fn) / sqrt(2 ^ mcc.prod)
  gmean <- sqrt(precision*recall)
  gmeasure <- 2*recall*(1-fpr)/(recall + (1-fpr))
  pf <- fp/(tn+fp) # probability of false alarm
  balance <- 1- (sqrt( (0-pf)^2 + (1-recall)^2)/sqrt(2) )

  out <- c(m1,Precision=precision,Recall=recall,FPR=fpr,TNR=tnr,Fmeasure=fmeasure,Accuracy=accuracy,MCC=mcc,Gmean=gmean,Gmeasure=gmeasure,Balance=balance, TP=tp, FP=fp, FN=fn, TN=tn)
  out
}
#######################################################
library(pROC)
library(ROCR)
outcome<-c(1,1,0,0,1,1,1,1,1)
prob<-c(1,1,0,0,1,0,1,1,1)
performance.calculation(outcome, prob, threshold=0.5)

#########################################
library("NMOF")
library("gradDescent")
# NOT RUN {
testFun <- function(x){
liste <-c(x[1L],x[2L],x[3L])
   return(liste)
}
par1 <- c(0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9)
par2 <- c(1,2,3,4,5,6,7,8,9,10)
par3 <- c(10,20,30,40,100)
sol <- gridSearch(fun = testFun, levels = list(par1, par2,par3))
sol$minfun
sol$minlevels
#######################################
library("mxnet")
mydata=read.csv("electricity-normalized.csv")
length1 <- length(mydata)-1
classIndex <- length(mydata)
#################
learnRateList <- c(0.005,0.01,0.003,0.007,0.011,0.015,0.019,0.023,0.027,0.031,0.035,0.039,0.043,0.047,0.051,0.055,0.059,0.063,0.067,0.071,0.075,0.079,0.083,0.087,0.091,0.095,0.099,0.103,0.107,0.111,0.115,0.119,0.123,0.127,0.131,0.135,0.139,0.143,0.147,0.151,0.155,0.159,0.163,0.167,0.171,0.175,0.179,0.183,0.187,0.191,0.195,0.199,0.203,0.207,0.211,0.215,0.219,0.223,0.227,0.231,0.235,0.239,0.243,0.247,0.251,0.255,0.259,0.263,0.267,0.271,0.275,0.279,0.283,0.287,0.291,0.295,0.299,0.303,0.307,0.311,0.315,0.319,0.323,0.327,0.331,0.335,0.339,0.343,0.347,0.351,0.355,0.359,0.363,0.367,0.371,0.375,0.379,0.383,0.387,0.391,0.395,0.399,0.403,0.407,0.411,0.415,0.419,0.423,0.427,0.431,0.435,0.439,0.443,0.447,0.451,0.455,0.459,0.463,0.467,0.471,0.475,0.479,0.483,0.487,0.491,0.495,0.499,0.503,0.507,0.511,0.515,0.519,0.523,0.527,0.531,0.535,0.539,0.543,0.547,0.551,0.555,0.559,0.563,0.567,0.571,0.575,0.579,0.583,0.587,0.591,0.595,0.599,0.603,0.607,0.611,0.615,0.619,0.623,0.627,0.631,0.635,0.639,0.643,0.647,0.651,0.655,0.659,0.663,0.667,0.671,0.675,0.679,0.683,0.687,0.691,0.695,0.699,0.703,0.707,0.711,0.715,0.719,0.723,0.727,0.731,0.735,0.739,0.743,0.747,0.751,0.755,0.759,0.763,0.767,0.771,0.775,0.779,0.783,0.787,0.791,0.795,0.799,0.803,0.807,0.811,0.815,0.819,0.823
)
lengthLR <- length(learnRateList)
#####%70 %30 division########
all <- length(mydata[,1])
trainPart <- round(all*70  /100)
testPart <- trainPart+1
######%70 %30 end#########
train.ind = c(1:trainPart)
train.x = data.matrix(mydata[train.ind, 1:length1])
train.y = mydata[train.ind, classIndex]
test.x = data.matrix(mydata[testPart:all, 1:length1])
test.y = mydata[testPart:all, classIndex]
#########################################
sonucList <- list("accuracy")
sonucList2 <- list("learningRate")
sonucList3 <- list("hiddenDim")
sonucList4 <- list("batchSize")
########################################
i <- 1
j <- 2
k <- 3
while(i<20){
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=j, out_node=2, out_activation="softmax",
num.round=20, array.batch.size=k, learning.rate=i, momentum=0.9,
eval.metric=mx.metric.accuracy)
preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
sonuc <-performance.calculation(pred.label, test.y, threshold=0.5)
sonucList <- list.append(sonucList,sonuc[7])
sonucList2 <- list.append(sonucList2,sol$values[i])
sonucList3 <- list.append(sonucList3,sol$values[j])
sonucList4 <- list.append(sonucList4,sol$values[k])
i <- i+3
j <- j+3
k <- k+3
}
###############################################
sonucListGenel <- cbind(sonucList,sonucList2,sonucList3,sonucList4)
sonucListGenel
write.csv(sonucListGenel,file="D:/gridSonuc.csv")