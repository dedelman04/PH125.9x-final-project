library(tidyverse)
library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(caret)
library(Matrix)
library(rpart)

#Load and join data
train_df <- read.csv(file="train_values.csv",
                     header=TRUE,
                     stringsAsFactors = TRUE)

train_label <- read.csv(file="train_labels.csv",
                        header=TRUE,
                        stringsAsFactors = TRUE)

train_data <- merge(x=train_df, y=train_label, by="row_id")

#Store the high, low, mean and median values for later use
low_mort <- as.integer(min(train_data$heart_disease_mortality_per_100k))
high_mort <- as.integer(max(train_data$heart_disease_mortality_per_100k))
hd_mean = mean(train_data$heart_disease_mortality_per_100k)
hd_med = median(train_data$heart_disease_mortality_per_100k)

#Rename a typo in the data set
train_data <- train_data %>% 
  mutate(health__pct_physical_inactivity = health__pct_physical_inacticity) %>%
  select(-health__pct_physical_inacticity)

#determine percent of records with NAs per feature
NAs <- data.frame(
  features = colnames(train_data %>% select(-heart_disease_mortality_per_100k, -yr, -row_id)),
  NA_pct = 
    as.numeric(format(sapply(train_data %>%
                               select(-heart_disease_mortality_per_100k,
                                      -yr, -row_id),
                             function(x){mean(ifelse(is.na(x), 1, 0))},
                             simplify=TRUE)*100, digits=3) )
) 

#Remove high NA features (higher than 10%)
high_NA <- NAs %>% 
  filter(NA_pct > 10) %>% 
  .$features %>% 
  as.character()

train_data <- train_data %>% select(-high_NA)

#Which features have minimum number of NAs (> 0)
min_missing <- min(NAs %>% filter(NA_pct > 0) %>% select(NA_pct))  
NA2 <- sapply(train_data %>% 
                select(as.vector(NAs$features[NAs$NA_pct == min_missing])),
              function(x){which(is.na(x))}, simplify=TRUE)

#What are the rowIDs for the rows with NA values on the features with the minimal percent of NAs
train_data <- train_data[-t(NA2)[1,c(1:2)], ]

#Rename original levels for area_rucc to better show on plots
levels(train_data$area__rucc) <- c("Metro - 1 Million +",
                                   "Metro - 250,000 to 1 Mil",
                                   "Metro - less than 250,000",
                                   "Nonmetro - Rural or < 2.5k, adjacent",
                                   "Nonmetro - Rural or < 2.5k, non-adjacent",
                                   "Nonmetro - 2,500 to 19,999, adjacent",
                                   "Nonmetro - 2,500 to 19,999, non-adjacent",
                                   "Nonmetro - 20,000 or more, adjacent",
                                   "Nonmetro - 20,000 or more, non-adjacent")

#Add "population" column using mapping
#Note - 2 different patterns map to 20-250k
train_data <- train_data %>% 
  mutate(population = ifelse(like(train_data$area__rucc, "19,999"), "2.5-20k",
                             ifelse(like(train_data$area__rucc, "2.5k"), "under 2.5k",
                                    ifelse(like(train_data$area__rucc, "20,000"), "20-250k",
                                           ifelse(like(train_data$area__rucc, "fewer than 250,000"), "20-250k",
                                                  ifelse(like(train_data$area__rucc, "250,000"), "250k-1M",
                                                         "1M+"))))))

#Force population into a factor
train_data$population <- factor(train_data$population, 
                                levels=c("under 2.5k",
                                         "2.5-20k",
                                         "20-250k",
                                         "250k-1M", "1M+"))


#Find median of air pollution, replace NA values with median of air pollution across all population
ap_med <- median(train_data$health__air_pollution_particulate_matter, na.rm=TRUE)
ap_NA <- which(is.na(train_data$health__air_pollution_particulate_matter))

train_data$health__air_pollution_particulate_matter[ap_NA] <- ap_med

#Create new groups for air pollution
cut_labels <- c("<= 10", "11", "12", "13", "14+")
cut_levels <- c(0, 10.9, 11.9, 12.9, 13.9, 100)

train_data$air_pollution <- cut(train_data$health__air_pollution_particulate_matter, cut_levels)
levels(train_data$air_pollution) <- cut_labels

#Which features have a NA % between 1 and 10
na_mid <- NAs[NAs$NA_pct > 1 & NAs$NA_pct < 10, ]

#Replace the NAs with the median for each feature
mid_cols <- as.vector(na_mid[, 1])

train_data <- train_data %>%
  mutate_at(mid_cols, ~ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))

#Create new columns for probability of having 2 or all 3 conditions
train_data <- train_data %>% 
  mutate(p_diab_obese = health__pct_adult_obesity * health__pct_diabetes) %>%
  mutate(p_obese_inact = health__pct_adult_obesity * health__pct_physical_inactivity) %>%
  mutate(p_diab_inact = health__pct_diabetes * health__pct_physical_inactivity) %>%
  mutate(p_all_three = health__pct_adult_obesity * health__pct_diabetes * health__pct_physical_inactivity )

#Remove unused columns from the data set
#These are the ones we're keeping
model_cols <- c("econ__economic_typology",
                "population",
                "air_pollution",
                "econ__pct_civilian_labor",
                "demo__pct_non_hispanic_african_american",
                "demo__death_rate_per_1k",
                "demo__pct_adults_bachelors_or_higher",
                "demo__pct_adults_less_than_a_high_school_diploma",
                "p_diab_obese",
                "p_obese_inact",
                "p_diab_inact",
                "p_all_three")

train_data <- train_data %>% select(model_cols, heart_disease_mortality_per_100k)

#Store the names of continuous and categorical columns
#Continuous
num_cols <- train_data %>% 
  select(-heart_disease_mortality_per_100k) %>%
  .[sapply(., is.numeric)] %>% 
  colnames()

#Categorical columns
cat_cols <- colnames(train_data)[sapply(train_data, is.factor)]

#Normalize the continuous variables using scale/z-score
#Create new DF with the scaled data
model_data <- cbind(train_data %>% select(heart_disease_mortality_per_100k, cat_cols),
                                   sapply(train_data[num_cols], scale, simplify=TRUE))



#####Modelling#####

###Create train and test sets from the larger dataset
set.seed(999)
test_index <- createDataPartition(model_data$heart_disease_mortality_per_100k, times = 1, p=0.5, list=FALSE)
train_set <- model_data[-test_index, ] 
test_set <- model_data[test_index, ]

###BEGIN Linear regression model###
fit <- lm(formula = heart_disease_mortality_per_100k ~ ., data = train_set)
y_hat <- predict(fit, test_set)
sqrt(mean((y_hat-test_set$heart_disease_mortality_per_100k)^2))

#Store results
results <- data.frame(method = "lm", 
                      RMSE = sqrt(mean((y_hat-test_set$heart_disease_mortality_per_100k)^2)),
                      TrainVal="N/A")

###END Linear regression model###


###BEGIN Regression Trees###
##Fit entire dataset##
#Start with complexity parameter = 0 then prune
rt_fit <- rpart(heart_disease_mortality_per_100k ~ ., data = model_data,
                control = rpart.control(cp=0, minsplit=2))
y_hat_rt <- predict(rt_fit)
#We know this is going to be 0 as each row is its own node
sqrt(mean((y_hat_rt-model_data$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="Regression Tree, Full data, CP=0",
                            RMSE = sqrt(mean((y_hat_rt-model_data$heart_disease_mortality_per_100k)^2)),
                            TrainVal = "N/A")
)

#Train the complexity parameter
#NOTE: This is not instantaneous
train_rt <- train(heart_disease_mortality_per_100k ~ .,
                  method = "rpart",
                  tuneGrid = data.frame(cp = seq(0, 0.05, len=100)),
                  data = model_data)

#ggplot(train_rt)

#Store the best tuned value
bt <- as.numeric(train_rt$bestTune[which.min(train_rt$bestTune)])

#Prune the regression tree using the best cp parameter
rt_pruned <- prune(rt_fit, cp=bt)
y_hat_pruned <- predict(rt_pruned)
sqrt(mean((y_hat_pruned - model_data$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="Reg Tree Full data - pruned",
                            RMSE=sqrt(mean((y_hat_pruned-model_data$heart_disease_mortality_per_100k)^2)),
                            TrainVal = paste("cp", format(bt, digits=3), sep="=")))

##Fit entire dataset##

##Run the same kind of model, but with Train/test set##
#Start with complexity parameter = 0 then prune
rt_fit <- rpart(heart_disease_mortality_per_100k ~ ., data = train_set,
                control = rpart.control(cp=0, minsplit=2))
y_hat_rt <- predict(rt_fit, train_set)
#We know this is going to be 0 as each row is its own node
sqrt(mean((y_hat_rt - train_set$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="Regression Tree, Train, CP=0",
                            RMSE = sqrt(mean((y_hat_rt - train_set$heart_disease_mortality_per_100k)^2)),
                            TrainVal="N/A")
)

#Train the complexity parameter                 
train_rt <- train(heart_disease_mortality_per_100k ~ .,
                  method = "rpart",
                  tuneGrid = data.frame(cp = seq(0, 0.05, len=100)),
                  data = train_set)

#ggplot(train_rt)

#Store the best tuned value
bt_train <- as.numeric(train_rt$bestTune[which.min(train_rt$bestTune)])

#Prune the regression tree using the best cp parameter; predict against test set
rt_pruned <- prune(rt_fit, cp=bt_train)
y_hat_pruned <- predict(rt_pruned, test_set)
sqrt(mean((y_hat_pruned - test_set$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="Reg Tree - pruned; train/test",
                            RMSE=sqrt(mean((y_hat_pruned - test_set$heart_disease_mortality_per_100k)^2)),
                            TrainVal = paste("cp", format(bt_train, digits=3), sep="=")))
###END Regression Trees###


###BEGIN Random Forests###

##randomForest
library(randomForest)

#Simple random forest with entire data set
train_rf <- randomForest(heart_disease_mortality_per_100k ~ ., data=model_data)

results <- rbind(results,
                 data.frame(method="randomForest",
                            RMSE=sqrt(mean((train_rf$predicted - model_data$heart_disease_mortality_per_100k)^2)),
                            TrainVal = "N/A"))

sqrt(mean((train_rf$predicted - model_data$heart_disease_mortality_per_100k)^2))
sqrt(mean((predict(train_rf) - model_data$heart_disease_mortality_per_100k)^2))

#Simple random forest with train/test set
train_rf_full <- randomForest(x=train_set[model_cols],
                              y=train_set$heart_disease_mortality_per_100k,
                              xtest = test_set[model_cols]
                              )

results <- rbind(results,
                 data.frame(method="randomForest, train/test",
                            RMSE=sqrt(mean((train_rf_full$test$predicted - test_set$heart_disease_mortality_per_100k)^2)),
                            TrainVal = "N/A"))

#Cross-validate over the mtry parameter, using train/test
####NOTE: THIS WILL TAKE SEVERAL MINUTES####
fit_rf_full <- train(method="rf",
                     x=train_set[model_cols],
                     y=train_set$heart_disease_mortality_per_100k,
                     tuneGrid = data.frame(mtry = seq(1,11)))

y_hat_rf_full <- predict(fit_rf_full$finalModel, test_set)

sqrt(mean((y_hat_rf_full - test_set$heart_disease_mortality_per_100k)^2, na.rm=TRUE))
sqrt(mean((predict(fit_rf_full, test_set) - test_set$heart_disease_mortality_per_100k)^2, na.rm=TRUE))

results <- rbind(results, 
                 data.frame(method="randomForest trained",
                            RMSE = sqrt(mean((y_hat_rf_full - test_set$heart_disease_mortality_per_100k)^2, na.rm=TRUE)),
                            TrainVal = paste("mtry", fit_rf_full$bestTune$mtry, sep="=")))
##randomForest

##Rborist
#Cross validate Rborist method over minNode and predFixed parameters
####NOTE: THIS COULD TAKE OVER AN HOUR####

library(Rborist)
tGrid = rbind(data.frame(predFixed=2, minNode = seq(2,50, by=2)),
              data.frame(predFixed=3, minNode = seq(2,50, by=2)),
              data.frame(predFixed=4, minNode = seq(2,50, by=2)),
              data.frame(predFixed=5, minNode = seq(2,50, by=2)))

fit_rf <- train(heart_disease_mortality_per_100k ~ .,
                method="Rborist",
                tuneGrid = tGrid,
                data=train_set)

y_hat_rf <- predict(fit_rf, test_set)
sqrt(mean((y_hat_rf-test_set$heart_disease_mortality_per_100k)^2))

bt1 <- as.numeric(fit_rf$bestTune[1])
bt2 <- as.numeric(fit_rf$bestTune[2])

results <- rbind(results,
                 data.frame(method="trained Rborist",
                            RMSE=sqrt(mean((y_hat_rf - test_set$heart_disease_mortality_per_100k)^2, na.rm=TRUE)),
                            TrainVal=paste(paste("predFixed", bt1, sep="="), paste("minNodes", bt2, sep="="))))

##Rborist
###END Random Forests###

##BEGIN Ensembles##
###Ensemble the 4 models together###
# linear regression -> fit; y_hat
# Regression Tree -> rt_pruned; y_hat_pruned
# randomForest -> fit_rf_full; y_hat_rf_full
# Rborist -> fit_rf; y_hat_rf

ens <- data.frame(lm = y_hat, rt = y_hat_pruned, rf = y_hat_rf_full, Rborist = y_hat_rf)

ens$ensemble <- rowMeans(ens)

ens_RMSE <- sqrt(mean((ens$ensemble - test_set$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="4 model ensemble",
                            RMSE=ens_RMSE,
                            TrainVal = "N/A")
)
##4 models

##Ensemble top 3 RMSE trained models (lm, rf, Rborist)
ens3 <- data.frame(lm = y_hat, rf = y_hat_rf_full, Rborist = y_hat_rf)
ens3$ensemble=rowMeans(ens3)
ens3_RMSE <- sqrt(mean((ens3$ensemble - test_set$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="3 model (lm, rf, Rbor) ensemble",
                            RMSE=ens3_RMSE,
                            TrainVal = "N/A")
)

##Ensemble only the trained random forest models
ens_rf <- data.frame(rf = y_hat_rf_full, Rborist = y_hat_rf)

ens_rf$ensemble=rowMeans(ens_rf)

ens_rf_RMSE <- sqrt(mean((ens_rf$ensemble - test_set$heart_disease_mortality_per_100k)^2))

results <- rbind(results,
                 data.frame(method="random forest ensemble",
                            RMSE=ens_rf_RMSE,
                            TrainVal = "N/A")
)

#write.csv(results, "model_results.csv", row.names = FALSE)
#results <- read.csv("model_results.csv", stringsAsFactors = FALSE)
