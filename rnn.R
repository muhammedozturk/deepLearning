# Clear workspace
rm(list=ls())

# Load libraries
require(rnn)
require(rlist)
# Set seed for reproducibility purposes
set.seed(10)

mydata <- read.csv("electricity-normalized.csv")
length1 <- length(mydata)-1
classIndex <- length(mydata)
learnRateList <- c(0.005,0.01,0.003,0.007,0.011,0.015,0.019,0.023,0.027,0.031,0.035,0.039,0.043,0.047,0.051,0.055,0.059,0.063,0.067,0.071,0.075,0.079,0.083,0.087,0.091,0.095,0.099,0.103,0.107,0.111,0.115,0.119,0.123,0.127,0.131,0.135,0.139,0.143,0.147,0.151,0.155,0.159,0.163,0.167,0.171,0.175,0.179,0.183,0.187,0.191,0.195,0.199,0.203,0.207,0.211,0.215,0.219,0.223,0.227,0.231,0.235,0.239,0.243,0.247,0.251,0.255,0.259,0.263,0.267,0.271,0.275,0.279,0.283,0.287,0.291,0.295,0.299,0.303,0.307,0.311,0.315,0.319,0.323,0.327,0.331,0.335,0.339,0.343,0.347,0.351,0.355,0.359,0.363,0.367,0.371,0.375,0.379,0.383,0.387,0.391,0.395,0.399,0.403,0.407,0.411,0.415,0.419,0.423,0.427,0.431,0.435,0.439,0.443,0.447,0.451,0.455,0.459,0.463,0.467,0.471,0.475,0.479,0.483,0.487,0.491,0.495,0.499,0.503,0.507,0.511,0.515,0.519,0.523,0.527,0.531,0.535,0.539,0.543,0.547,0.551,0.555,0.559,0.563,0.567,0.571,0.575,0.579,0.583,0.587,0.591,0.595,0.599,0.603,0.607,0.611,0.615,0.619,0.623,0.627,0.631,0.635,0.639,0.643,0.647,0.651,0.655,0.659,0.663,0.667,0.671,0.675,0.679,0.683,0.687,0.691,0.695,0.699,0.703,0.707,0.711,0.715,0.719,0.723,0.727,0.731,0.735,0.739,0.743,0.747,0.751,0.755,0.759,0.763,0.767,0.771,0.775,0.779,0.783,0.787,0.791,0.795,0.799,0.803,0.807,0.811,0.815,0.819,0.823
)
lengthLR <- length(learnRateList)
#####%70 %30 division########
all <- length(mydata[,1])
trainPart <- round(all*70  /100)
testPart <- trainPart+1
######%70 %30 end#########
################PREDICTION#################################
X=as.matrix(mydata)
sonucList <- list("accuracy")
for(i in learnRateList){
# Train model. Keep out the last two sequences.
model <- trainr(Y = X[1:5000,],
                X = X[5001:10000,],
                learningrate = i,
                hidden_dim = 16,
                numepochs = 1500)

# Predicted values
Yp <- predictr(model, X[testPart:all,])
predicted <- 0
##Sifir vectoru olusturalim
length <- length(Yp[,classIndex])
indis <- c(1:length)
for(indis2 in indis){
		predicted[indis2] <- 0
}
for(indis2 in indis){
		if(Yp[indis2]<0.5)
		{
			predicted[indis2]<- 0
		}
		else
		{
				predicted[indis2]<- 1
		}
}
sonuc <- performance.calculation(X[testPart:all,classIndex], predicted, threshold=0.5)
sonucList <- list.append(sonucList,sonuc[7])
}
sonucListGenel <- cbind(learnRateList,sonucList)
write.csv(sonucListGenel,file="C:/mainFrame/makaleler/deepLearnTuning/dokumanlar/rnn/learnRateSonuc.csv")


