######################################
###################################
#########################################
sonucList <- list("accuracy")
sonucList2 <- list("learningRate")
sonucList3 <- list("hiddenDim")
sonucList4 <- list("batchSize")
par1 <- c(0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9)
par2 <- c(1,2,3,4,5,6,7,8,9,10)
par3 <- c(10,20,30,40,100)
par1 <- sample(par1,1)
par2 <- sample(par2,1)
par3 <- sample(par3,1)
piece <- c(10,20,30,40,50,60)
piece <- sample(piece,1)
library("deepnet")
library("rlist")
dosya <- "dataset_37_diabetes.csv"
yol <- "C:/Users/User/Documents/"
yol <- paste(yol,"Random",dosya,sep="")
mydata=read.csv(dosya)
length1 <- length(mydata)-1
classIndex <- length(mydata)
#################
#####%70 %30 division########
all <- length(mydata[,1])
trainPart <- round(all*piece  /100)
testPart <- trainPart+1
######%70 %30 end#########
train.ind = c(1:trainPart)
train.x = data.matrix(mydata[train.ind, 1:length1])
train.y = mydata[train.ind, classIndex]
test.x = data.matrix(mydata[testPart:all, 1:length1])
test.y = mydata[testPart:all, classIndex]
#########################################
###############################
dnn <- dbn.dnn.train(train.x, train.y, hidden = par2,learningrate=par1,batchsize=par3)
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

################call performance.calculation#############################
sonuc <- performance.calculation(test.y, predicted, threshold=0.2)
par1
par2
par3
sonucList <- list.append(sonucList,sonuc[7])
sonucList2 <- list.append(sonucList2,par1)
sonucList3 <- list.append(sonucList3,par2)
sonucList4 <- list.append(sonucList4,par3)
#############################

sonucListGenel <- cbind(sonucList,sonucList2,sonucList3,sonucList4)
sonucListGenel
write.csv(sonucListGenel,file=yol)
#############################