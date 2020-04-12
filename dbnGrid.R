######################################
###################################
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

library("deepnet")
library("rlist")
mydata=read.csv("nomao.csv")
length1 <- length(mydata)-1
classIndex <- length(mydata)
#################
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
while(i<1498){
###############################
dnn <- dbn.dnn.train(train.x, train.y, hidden = sol$values[j],learningrate=sol$values[i],batchsize=sol$values[k])
## predict by dnn
yy <- nn.predict(dnn, test.x)
predicted <- 0
##Sifir vectoru olusturalim
length <- length(test.y)
indis <- c(1:length)
for(indis2 in indis){
		predicted[indis2] <- 0
}
##############predicted vectoru olussun ihtimale bagli#######################
for(indis2 in indis){
		if(yy[indis2]<0.5)
		{
			predicted[indis2]<- 0
		}
		else
		{
				predicted[indis2]<- 1
		}
}
i <- i+3
j <- j+3
k <- k+3
################call performance.calculation#############################
sonuc <- performance.calculation(test.y, predicted, threshold=0.2)
sonucList <- list.append(sonucList,sonuc[7])
sonucList2 <- list.append(sonucList2,sol$values[i])
sonucList3 <- list.append(sonucList3,sol$values[j])
sonucList4 <- list.append(sonucList4,sol$values[k])
#############################
}
sonucListGenel <- cbind(sonucList,sonucList2,sonucList3,sonucList4)
sonucListGenel
write.csv(sonucListGenel,file="D:/gridSonuc.csv")
#############################