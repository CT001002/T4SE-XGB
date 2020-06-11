####ReliefF install.packages("CORElearn")
library(lattice)
library(ggplot2)
library(readxl)
library(CORElearn)


data1<-read.csv("1471_maxmin.csv",header=TRUE)
label<-read.csv("1471labels.csv",header=FALSE)


data<-as.data.frame(data1[,-1])
LB <- as.matrix(label)
data2<-data.frame(LB,data)
data2<-data.frame(LB,data3)
R_1 <- attrEval(V1~.,data2,estimator="ReliefFexpRank",ReliefIterations = 100);
R1 <- sort(-R_1)


write.table(R1,file = '1471_reliefF.txt',sep = '\n')


####gainratio install.packages("FSelector")
library("FSelector")
library(lattice)
library(readxl)


data1<-read.csv("1471_maxmin.csv",header=TRUE)
label<-read.csv("1471labels.csv",header=FALSE)


data<-as.data.frame(data1[,-1])
LB <- as.matrix(label)
data2<-data.frame(LB,data)
weights <- gain.ratio(V1~., data2)

write.table(weights,file = '1471_gainratio.txt',sep = '\n')