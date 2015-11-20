# Investigate of predict the manner by the Wearable Devices
yjcelly  
2015-11-19  


```r
library(knitr)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(kernlab)
library(nnet)
library(RSNNS)
```

```
## Loading required package: Rcpp
## 
## Attaching package: 'RSNNS'
## 
## The following objects are masked from 'package:caret':
## 
##     confusionMatrix, train
```


## Overview
The project will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and to predict the manner in which they did the exercise. 


## Data Description
The data lable is come from the 6 participants which were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The field of  "classe" is the lable. The training data  available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv .The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv .The test data without the lable.


## Data Prepare Process
At first, we must preprocess the data, including converting the abnormal value to zero, changing the empty value to zero, modify the string of yes and no to the numeric and so on.

```r
setwd("F:\\coursera\\8_Practical Machine Learning\\Writeup")

###read the csv data
traindata <- read.csv("pml-training.csv",stringsAsFactors=FALSE)
testdata <- read.csv("pml-testing.csv",stringsAsFactors=FALSE)

###convert the abnormal value to zero
traindata[traindata == "NA"] <-as.numeric(0)
testdata[testdata == "NA"] <- as.numeric(0)
traindata[traindata == "#DIV/0!"] <-as.numeric(0)
testdata[testdata == "#DIV/0!"] <- as.numeric(0)

###change the empty value and string zero to numeric zero
traindata[traindata == ""] <- as.numeric(0)
testdata[testdata == ""] <- as.numeric(0)
traindata[traindata == "0"] <- as.numeric(0)
testdata[testdata == "0"] <- as.numeric(0)

###modify the string of yes and no to numeric zero
traindata[traindata  == "yes"] <- as.numeric(1)
traindata[traindata == "no"] <- as.numeric(0)
testdata[testdata  == "yes"] <- as.numeric(1)
testdata[testdata  == "no"] <- as.numeric(0)

###other abnormal value to zero
traindata[is.na(traindata)] <-as.numeric(0)
testdata[is.na(testdata)] <- as.numeric(0)
```

Besides that, we also should extract the useful fields and  convert all data to numeric. An important step is to normalize the data for mean and square deviation.    

```r
predictColIndex=6:159
num <- sapply(traindata[, predictColIndex], as.numeric)
cleanTrainData <- data.frame(num)
num <- sapply(testdata[, predictColIndex], as.numeric)
cleanTestData <- data.frame(num)


cleanParamColIndex=1:154
preObj <- preProcess(cleanTrainData[,cleanParamColIndex],method=c("center","scale"))
```

```
## Warning in preProcess.default(cleanTrainData[, cleanParamColIndex], method
## = c("center", : These variables have zero variances: kurtosis_yaw_belt,
## skewness_yaw_belt, amplitude_yaw_belt, kurtosis_yaw_dumbbell,
## skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm,
## skewness_yaw_forearm, amplitude_yaw_forearm
```

```r
trainPreObj <- predict(preObj,cleanTrainData[,cleanParamColIndex])
testPreObj <-  predict(preObj,cleanTestData [,cleanParamColIndex])
```

At last, we combine the value and label for training data. For convenience, we give the test data an virtual label.

```r
clean_lab_trainData <- cbind(trainPreObj ,as.factor(traindata$classe) )
names(clean_lab_trainData) <- c(names(trainPreObj),"classe")
clean_lab_trainData  <- na.omit(clean_lab_trainData )

clean_lab_testData <- cbind(testPreObj ,as.factor( testdata$problem_id)  )
names(clean_lab_testData ) <- c(names(testPreObj ),"classe")
```



## Predict the Manner by Decision Tree
At first, we use a simple machine learning method of decision tree. In the decision tree, we should random split the data into train and develop set. We set the ratio of training data is 0.85. 

```r
inTrain <- createDataPartition(y=clean_lab_trainData$classe,p=0.85, list=FALSE)
training_85 <- clean_lab_trainData[inTrain,]
dev_15 <- clean_lab_trainData[-inTrain,]


modFit <- train(classe~.,method="rpart",data=training_85)
```

```
## Loading required package: rpart
```

```r
train_ret <- caret::confusionMatrix(training_85$classe,predict(modFit,training_85 ) )
dev_ret <- caret::confusionMatrix(dev_15$classe,predict(modFit,dev_15 ) )
```

After training, we give the result of in the training and develop set. The accuracy in the develop set is about 50% and it is not a very good predict.

```r
train_ret$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.5711631      0.4520327      0.5636106      0.5786907      0.4006595 
## AccuracyPValue  McnemarPValue 
##      0.0000000      0.0000000
```

```r
dev_ret$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   5.717199e-01   4.531597e-01   5.536054e-01   5.896916e-01   4.095853e-01 
## AccuracyPValue  McnemarPValue 
##   3.559286e-70  7.564251e-203
```


## Predict the Manner by Multi-layer Perceptron
We also try the multi-layer perceptron for predicting the manner. With the help of RSNNS package, we use the function of mlp. In the help of mlp function, we know the useage. Firstly, we should random sort for the data and split the data to train and develop set. Secondly, we should decode the class lable. It means to convert the class to sequence of 0 and 1. At last, call the function mlp and give the params. The cross-validation style is leave one out method.

```r
###random sort for all data
clean_lab_trainDataRandom <- clean_lab_trainData[sample(1:nrow(clean_lab_trainData ),length(1:nrow(clean_lab_trainData ))),1:ncol(clean_lab_trainData )]
inputValues= clean_lab_trainDataRandom[,1:154]

##decode the class lable, means convert the lable to sequence of 0 and 1.
outputTargets = decodeClassLabels(clean_lab_trainDataRandom[,155])

##split the data to train set and develop set
train_dev = splitForTrainingAndTest(inputValues, outputTargets , ratio=0.15)
model = mlp(train_dev$inputsTrain, train_dev$targetsTrain, size=30, learnFunc="Std_Backpropagation", learnFuncParams=c(0.1),maxit=100,hiddenActFunc = "Act_Logistic", inputsTest=train_dev$inputsTest, targetsTest=train_dev$targetsTest)

### predict and get the result
predictions = predict(model,train_dev$inputsTest)
dev_ret <- RSNNS::confusionMatrix(train_dev$targetsTest,predictions) 
test_ret <- predict(model,clean_lab_testData[,1:154])
```

The result of multi-layer Perceptron is very good. The accuracy is 95%.

```r
dev_ret
```

```
##        predictions
## targets   1   2   3   4   5
##       1 779   4   5   0   1
##       2   9 540  16   6   4
##       3   5   3 503  12   4
##       4   4   2  19 456   3
##       5   3   4  10   2 550
```

```r
sprintf("The accuracy of develop set is %.3f",(dev_ret[1,1]+dev_ret[2,2]+dev_ret[3,3]+dev_ret[4,4]+dev_ret[5,5])/sum(dev_ret))
```

```
## [1] "The accuracy of develop set is 0.961"
```

```r
max.col(test_ret)
```

```
##  [1] 2 1 2 1 1 3 4 2 1 1 4 3 2 1 5 5 1 2 2 2
```

## Expected out of sample error
As described in the end, we choose the method of multi-layer perceptron. At the develop set, the accuracy is close 95%. So we conclude the model will have a good result in the out of sample. We guess the  error of out of sample is about 5%-8% because the accuracy will decrease in the out of sample testset. It also a good predict for the manner.



