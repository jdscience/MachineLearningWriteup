# Machine Learning Assignment: Writeup
JDScience  
January 25, 2015  

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```
# Overview
This document will outline steps taken to understand, cleanup and prepare for a
machine learning algorithm. For this document, I have elected to use a "Random Forest"
model.

# Data Acquisition and Exploration
Per the instructions provided with the assignment, the data can be downloaded with
the following commands:

## Acquire and Load

```r
if(!file.exists("data")){
  dir.create("data")
}

fileUrl_Training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl_Training,destfile="./data/training.csv",method="curl")

fileUrl_Testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl_Testing,destfile="./data/testing.csv",method="curl")

list.files("./data") ; dateDownloaded <- date() ; dateDownloaded
```

```
## [1] "testing.csv"  "training.csv"
```

```
## [1] "Sun Jan 25 14:05:59 2015"
```

```r
training <- read.csv("./data/training.csv", as.is=TRUE)
testing <- read.csv("./data/testing.csv", as.is=TRUE)

dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

We can see that the dataset has 160 columns, and the training set has many rows
to use to build our model.

## Exploration
For a better picture of our model, a "summary(training)" was performed and reviewed.
This exposed a number of facts we will need to take into account for our analysis:
* There are many rows which contain NAs. We will need to decide whether to impute the data or not.
* There are some columns which seem to contain no data; we might need to remove them.
* We will be building a model to predict the "classe" column.
* The "user_name" column seems to be one we might convert via dummyVars().
* Some columns don't seem useful - the raw timestamp fields seem redundant with the timestamp field, for example

# Data Cleanup
A lot of small things need to be done in order to clean up our data. 

## Timestamp Fields
To start, the timestamp fields are removed and converted using this function:

```r
cleanup <- function(y) {
    #remove a couple specific rows
    y$raw_timestamp_part_1 <- NULL
    y$raw_timestamp_part_2 <- NULL

    # convert timestamp
    y$cvtd_timestamp <- as.POSIXct(strptime(y$cvtd_timestamp, "%d/%m/%Y %H:%M"))

    return(y)
}

training_clean <- cleanup(training)
testing_clean <- cleanup(testing)
```

## Flagging our Prediction Variable
A quick conversion to factor and renaming:

```r
training_clean$class <- as.factor(training$classe)
training_clean$classe <- NULL
```

## Identify and Remove Empty Columns
To remove the unnecessary columns, the nearZeroVar() function is employed, and
applied:

```r
nsv <- nearZeroVar(training_clean,saveMetrics=TRUE)
training_clean[,rownames(nsv)[nsv$nzv]] <- list(NULL)
testing_clean[,rownames(nsv)[nsv$nzv]] <- list(NULL)
```

## Dummy Variables
Since the activity was performed by different users, and unique user data may assist
in the prediction, we convert the user_name values to individual columns for our
model. After which, the user_name column is removed.

```r
training_dummies <- dummyVars(X ~ user_name, data = training_clean)
training_clean <- cbind(training_clean,predict(training_dummies,newdata=training_clean))
training_clean$user_name <- NULL

testing_clean <- cbind(testing_clean,predict(training_dummies,newdata=testing_clean))
testing_clean$user_name <- NULL
```

## Additional Cleanup
Several actions perform best when two requirements are met:
* All values in the data.frame must be numeric.
* Both the training and testing sets must have identical columns.

The work to accomplish these is identified here below:

```r
training_split_nonnum <- data.frame(training_clean$class)
colnames(training_split_nonnum)[colnames(training_split_nonnum)==
                                "training_clean.class"] <- "class"
training_split_nonnum$cvtd_timestamp <- training_clean$cvtd_timestamp
training_split_num <- training_clean[-2] # removes timestamp from nonnumeric
training_split_num$class <- NULL # removes class from nonnumeric
    
testing_split_nonnum <- data.frame(testing_clean$cvtd_timestamp)
colnames(testing_split_nonnum)[colnames(testing_split_nonnum)==
                                "testing_clean.cvtd_timestamp"] <- "cvtd_timestamp"
testing_split_nonnum$problem_id <- testing_clean$problem_id
testing_split_num <- testing_clean[-2] # removes timestamp from numeric
testing_split_num$problem_id <- NULL
```

## Remove the highly-correlated columns
The columns which very highly correlate to each other need to be removed:

```r
verysmallset <- complete.cases(training_split_num)
training_trimmed <- training_split_num[verysmallset,]
descrCorr <- abs(cor(training_trimmed))
diag(descrCorr) <- 0
highCorr <- findCorrelation(descrCorr, 0.95)
training_split_num <- training_split_num[, -highCorr]
testing_split_num  <- testing_split_num[, -highCorr]
```

# Impute and Principal Component Analysis
Now that the cleanup is primarily completed, we move onto the process of imputing
our dataset. Though not described above, it is worth noting that the "complete.cases()"
call above resulted in around 400 rows where all columns were populated. While that
allows easy correlation analysis as performed, it is not sufficient for generation
of the model.
Note that we will the the impute and PCA process, then applying the same output rules
to both the training and testing data sets.

```r
training_preprocessed_data <- preProcess(training_split_num, method=c("knnImpute", "pca"))
training_near_ready <- predict(training_preprocessed_data,training_split_num, drop=TRUE)
testing_near_ready <- predict(training_preprocessed_data,testing_split_num, drop=TRUE)

## recombine the sets
training_ready <- cbind(training_split_nonnum, training_near_ready)
testing_ready <- cbind(testing_split_nonnum, testing_near_ready)
```

# Model Training
With our data cleanup and preparations complete, we are ready to move onto the creation
of our model.

```r
date()
```

```
## [1] "Sun Jan 25 14:06:38 2015"
```

```r
modelFit <- train(class ~ .,data=training_ready, method="rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
date()
```

```
## [1] "Sun Jan 25 14:29:12 2015"
```

```r
modelFit
```

```
## Random Forest 
## 
## 19622 samples
##    38 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9905524  0.9880424  0.001511864  0.001913528
##   20    0.9886972  0.9856948  0.001525405  0.001930431
##   38    0.9779171  0.9720515  0.003652186  0.004619510
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
It is also worth noting that the model had a small amount of error, as can be seen
here:

```r
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5575    4    0    1    0 0.0008960573
## B   26 3763    8    0    0 0.0089544377
## C    1   19 3397    4    1 0.0073056692
## D    0    0   36 3167   13 0.0152363184
## E    0    0    1   13 3593 0.0038813418
```

# Prediction
Now that the model is built, we apply it to the testing set to create our prediction:

```r
testing_prediction <- predict(modelFit, newdata=testing_ready)
length(testing_prediction)
```

```
## [1] 20
```
Note that the length of the "testing_prediction" vector is 20 elements, matching each
row in our testing set.

