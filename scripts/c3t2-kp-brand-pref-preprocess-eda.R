#task: build model to predict customer brand preferences--starting with clean data, skipping eda


########### LOAD PKGS & DATA----
library(tidyverse)
library(here)
library(janitor)
library(skimr)
library(generics)
#ran in console--install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
#library(doSNOW) #enables doing training in parallel


 #read in training dataset
surveydata <- read.csv(here("data", "CompleteResponses.csv"))





########### DATA PREP----

#rename vars
cln_surveydata <- surveydata %>% 
  clean_names() %>% 
  rename(region = zipcode) %>% # new name = old name
  rename(edu = elevel)

#verify var name changes
glimpse(cln_surveydata)



#change predictor vars to factor dtype 
cln_surveydata$edu <- as.factor(cln_surveydata$edu)
cln_surveydata$car <- as.factor(cln_surveydata$car)
cln_surveydata$region <- as.factor(cln_surveydata$region)

#verify changes
glimpse(cln_surveydata)

#change target var to factor--so treated as classification
cln_surveydata$brand <- as.factor(cln_surveydata$brand)


#verify change
is.factor(cln_surveydata$brand)





########### PLOT VAR DISTRUBUTIONS----

#use hist() for continuous vars and barplot(table()) for categorical vars

#plot distributions of predictor vars
hist(cln_surveydata$salary)
hist(cln_surveydata$age)
hist(cln_surveydata$credit)
barplot(table(cln_surveydata$edu), 
        main = "Education Level",
        ylab = "Count")
barplot(table(cln_surveydata$car),
        main = "Car Type",
        ylab = "Count")
barplot(table(cln_surveydata$region),
        main = "Region",
        ylab = "Count")

#all predictor vars are aprox. uniformly distributed across bins/categories 


#plot distribution of obs in target var
barplot(table(cln_surveydata$brand))

#brand 1 is preferred by more  customers--since target var not uniformly distributed use stratified splitting/sampling


#get value counts for brand classes--class (im)balance relevant to interpreting model performance measures
cln_surveydata %>% count(brand)
# class 0 is 37.82582%




########### SETUP TRAIN-TEST SPLIT----

#set random number generator so same obs are used used in split every time--important for reproducibility
set.seed(123)


#specify partition of 75/25 (p = .75), list = FALSE to output vector, not list
in_train <- createDataPartition(cln_surveydata$brand,
                                p = .75, 
                                list = FALSE)

#use partition to create training dataset--[intraining,] tells R to use all columns when creating the training data
train_data <- cln_surveydata[in_train,]

#use partition to create testing dataset
test_data <- cln_surveydata[-in_train,]




########### SETUP PARALLEL PROCESSING----

#use makecluster (doSNOW) to tell R use parallel process
#pcluster <- makeCluster(4, type = "SOCK")

#register cluster so R will know what to use for pp
#registerDoSNOW(pcluster)





########### SETUP TRAINING & TUNING SCHEMES----

#setup 5-fold cross validation method train models
control <- trainControl(method = "repeatedcv",
                        number = 10,  
                        repeats = 1)

#setup parameter tuning grid FOR MANUAL TUNING ONLY
#grid <- expand.grid(size = c(2,7,12,18,24,34), k = c(1,2,3,4,5))




########### RF MODEL—TRAIN, PREDICT, & EVAL ########### 

# TRAIN & TUNE

#train & auto-tune random forest model, wrapper to time training
system.time(rf_fit1 <- train(brand~., 
                             data = train_data, 
                             method = "rf", 
                             trControl = control,
                             tunelength = 5))
#runtime w/ clstr elapsed: 321.69, w/o clstr elapsed: 804.95


#train & manually tune random forest model
#system.time(rf_fit2 <- train(brand~., 
#                             data = train_data, 
#                             method = "rf", 
#                             trControl = control,
#                             tunelength = grid))
#run time w/ cluster 338.33


# PREDICT

#use rf_fit1 to predict brand preferences for obs in the testing_data
(pred_rf_fit1 <- predict(rf_fit1, test_data))

# EVALUATE

#view confusion matrix to est model performance on test data
(perform_rf_fit1 <- confusionMatrix(pred_rf_fit1,
                                    test_data$brand))



########### GBM MODEL-TRAIN, PREDIT, & EVAL #########

# TRAIN & AUTO TUNE

#train a GBM (gradient boosting) model on the training_data, 3-fold crossval, w/ system.time wrapper
system.time(gbm_fit1 <- train(brand~., 
                              data = train_data, 
                              method = "gbm", 
                              trControl = control,
                              train.fraction = 0.5))
#runtime w/o clsr elapsed =98.54


# PREDICT

#use gbm_fit1 to predict brand preferences in testing_data
preds_gbm_fit1 <- predict(gbm_fit1, 
                          test_data)

# EVALUATE

#view confusion matrix of gbm model performance on test data
(perform_gbm_fit1 <- confusionMatrix(preds_gbm_fit1,
                                     test_data$brand))


# ALGO SELECTION

# models perform similarly, but gbm_fit1 is slightly better than rf_fit1, and gbm_fit trains in a fraction of the time. So I will use gbm_fit1 moving forward. 





########### VAR IMPORTANCE ######### ----

# RF MODEL 

#calculates importance of predictor vars
varImp(rf_fit1)

# top 3: salary 100, age 65, credit 9. After top 3, importance score for all other vars drops to below 1.



# GBM MODEL

# must load gbm pkg for varImp to run on gbm model  
library(gbm)

#calculate the importance of each feature/predictor var 
varImp(gbm_fit1)

# top 3 predictors (in order): age, salary credit. Interestingly, the most important var in the gbm model is different from the rf model. Not sure what to make of that.





########### FEATURE SELECTION ############

# Because 3 variables show significantly more importance than the other vars, I want to look at the relationship between those vars and brand preference more closely. 

#plot brand pref and salary
cln_surveydata %>% 
  ggplot(aes(x = brand, y = salary)) +
  geom_violin()

#plot brand pref and age
cln_surveydata %>% 
  ggplot(aes(x = brand, y = age)) +
  geom_violin()

#plot brand pref and credit
cln_surveydata %>% 
  ggplot(aes(x = brand, y = credit)) +
  geom_violin()



# I wonder if all or most of the predictive power of gbm_fit1 is coming from the top 3 vars. So I want to check that



# AGE ONLY
system.time(gbm_agefit <- train(brand~. -salary -edu -car -region -credit, 
                                   data = train_data, 
                                   method = "gbm", 
                                   trControl = control))

# predict brand pref for test_data
(pred_gbm_agefit <- predict(gbm_agefit, test_data))

#evaluate performance using confusion matrix
(perform_gbm_agefit <- confusionMatrix(pred_gbm_agefit,
                                          test_data$brand))

# finding: age only model had zero predictive power



# CREDIT ONLY
system.time(gbm_creditfit <- train(brand~. -salary -edu -car -region -age, 
                                data = train_data, 
                                method = "gbm", 
                                trControl = control))

# predict brand pref for test_data
(pred_gbm_creditfit <- predict(gbm_creditfit, test_data))

#evaluate performance using confusion matrix
(perform_gbm_creditfit <- confusionMatrix(pred_gbm_creditfit,
                                       test_data$brand))

# finding: credit only model had zero predictive power

# SALARY ONLY
system.time(gbm_salaryfit <- train(brand~. -age -edu -car -region -credit, 
                             data = train_data, 
                             method = "gbm", 
                             trControl = control))

#use gbm_salaryfit to predict brand pref for the test_data
(pred_gbm_salaryfit <- predict(gbm_salaryfit, test_data))

#evaluate performance of rf_salaryfit using confusion matrix
(perform_gbm_salaryfit <- confusionMatrix(pred_gbm_salaryfit,
                                       test_data$brand))

# finding: salary only model performs considerably worse than gbm_fit1, accuracy = 0.72



# AGE & CREDIT ONLY
system.time(gbm_agecredfit <- train(brand~. -edu -car -region -salary, 
                                   data = train_data, 
                                   method = "gbm", 
                                   trControl = control))

#use gbm_salaryfit to predict brand pref for the test_data
(pred_gbm_agecredfit <- predict(gbm_agecredfit, test_data))

#evaluate performance of rf_salaryfit using confusion matrix
(perform_gbm_agecredfit <- confusionMatrix(pred_gbm_agecredfit,
                                          test_data$brand))

# finding: age and credit only model had zero predictive power



#AGE, SALARY, AND CREDIT ONLY
system.time(gbm_agecredsalfit <- train(brand~. -edu -car -region, 
                                    data = train_data, 
                                    method = "gbm", 
                                    trControl = control))

#use gbm_salaryfit to predict brand pref for the test_data
(pred_gbm_agecredsalfit <- predict(gbm_agecredsalfit, test_data))

#evaluate performance of rf_salaryfit using confusion matrix
(perform_gbm_agecredsalfit <- confusionMatrix(pred_gbm_agecredsalfit,
                                           test_data$brand))

# finding: age only model had slightly more predictive power than the full model 0.9297 vs. 0.9232


########### MODEL SELECTION #############

# RF VS. GBM
# the rf and gbm models performed almost identicall

# FEATURE SELECTION
# so moving forward, i will use model with only age, salary, and credit vars as predictors.



########### PREDICT INCOMPLETE DATA ###########

#read in incomplete survey data
incompletesurveydata <- read.csv(here("data", "SurveyIncomplete.csv"))

glimpse(incompletesurveydata)



# DATA PREP

#clean var names and rename vars
incompletesurveydata <- incompletesurveydata %>% 
  clean_names() %>% 
  rename(edu = elevel) %>%    # new name = old name 
  rename(region = zipcode)

glimpse(incompletesurveydata)    #verify changes

#check for missing values 
sum(is.na(incompletesurveydata))

#change var dtypes to factor
incompletesurveydata$edu <- as.factor(incompletesurveydata$edu)
incompletesurveydata$car <- as.factor(incompletesurveydata$car)
incompletesurveydata$region <- as.factor(incompletesurveydata$region)
incompletesurveydata$brand <- as.factor(incompletesurveydata$brand)

glimpse(incompletesurveydata)   # verify changes


#use final model (gbm_agecredsalfit) to predict brand pref for entirely new dataset (incompletesurveydata)
pred_incompletedata <- predict(gbm_agecredsalfit,
                               newdata = incompletesurveydata)

#evaluate performance
(perform_incompletedata <- confusionMatrix(pred_incompletedata, incompletesurveydata$brand))


#stop parallel processing
#stopCluster(pcluster)


gbm_agecredsalfit$modelInfo
gbm_agecredsalfit$dots











