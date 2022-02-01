rm(list=ls())

library(mdsr)
library(rpart)
library(partykit)
library(rattle)
library(randomForest)
library(nnet)
library(class)
library(e1071)
library(ROCR)
library(gridExtra)

census <- read.csv(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  header = FALSE)
names(census) <- c("age", "workclass", "fnlwgt", "education",
                   "education.num", "marital.status", "occupation", "relationship",
                   "race", "sex", "capital.gain", "capital.loss", "hours.per.week",
                   "native.country", "income")
set.seed(364)
n <- nrow(census)
test_idx <- sample.int(n, size = round(0.2 * n))
train <- census[-test_idx,]
test <- census[test_idx,]

form <- as.formula("income ~ age + workclass + education + marital.status +
  occupation + relationship + race + sex + capital.gain + capital.loss +
  hours.per.week")

# Decision Tree -----------------------------------------------------------

mod_tree <- rpart(form, data = train)
income_tree_probs <- mod_tree %>% 
  predict(newdata = test, type = "prob") %>% 
  as.data.frame()
income_tree_probs %>% head()

mod_tree %>% fancyRpartPlot()

pred_tree <- ROCR::prediction(income_tree_probs$` >50K`, test$income)
perf_tree <- ROCR::performance(pred_tree, 'tpr', 'fpr')
perf_tree_df <- data.frame(perf_tree@x.values, perf_tree@y.values, perf_tree@alpha.values)
names(perf_tree_df) <- c("fpr", "tpr", "cut")
perf_tree_df %>% head()

roc_tree <- perf_tree_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_tree@y.name) + xlab(perf_tree@x.name) + ggtitle("Decision Tree")
roc_tree

# K-Nearest Neighbors -----------------------------------------------------

train_q <- train %>%
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
test_q <- test %>% 
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
income_knn <- knn(train_q, test = test_q, cl = train$income, k = 10, prob = TRUE)

income_knn_probs <- matrix(nrow = length(income_knn), ncol = 1)
for(i in 1:length(income_knn)) {
  p = attr(income_knn, 'prob')[i]
  income_knn_probs[i, 1] <- ifelse(income_knn[i] == ' >50K', p, 1 - p)
}
income_knn_probs <- income_knn_probs %>% as.data.frame()
names(income_knn_probs) <- c(' >50K')

income_knn_probs %>% head()

pred_knn <- ROCR::prediction(income_knn_probs$` >50K`, test$income)
perf_knn <- ROCR::performance(pred_knn, 'tpr', 'fpr')
perf_knn_df <- data.frame(perf_knn@x.values, perf_knn@y.values, perf_knn@alpha.values)
names(perf_knn_df) <- c("fpr", "tpr", "cut")
perf_knn_df %>% head()

roc_knn <- perf_knn_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_knn@y.name) + xlab(perf_knn@x.name) + ggtitle("K-Nearest Neighbors")
roc_knn

# Naive Bayes -------------------------------------------------------------

mod_nb <- naiveBayes(form, data = train)
income_nb_probs <- mod_nb %>%
  predict(newdata = test, type = "raw") %>%
  as.data.frame()
income_nb_probs %>% head()

pred_nb <- ROCR::prediction(income_nb_probs$` >50K`, test$income)
perf_nb <- ROCR::performance(pred_nb, 'tpr', 'fpr')
perf_nb_df <- data.frame(perf_nb@x.values, perf_nb@y.values, perf_nb@alpha.values)
names(perf_nb_df) <- c("fpr", "tpr", "cut")
perf_nb_df %>% head()

roc_nb <- perf_nb_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_nb@y.name) + xlab(perf_nb@x.name) + ggtitle("Naive Bayes")
roc_nb

# Neural Network ----------------------------------------------------------

mod_nn <- nnet(form, data = train, size = 5)
income_nn_probs <- mod_nn %>% 
  predict(newdata = test, type = "raw") %>% 
  as.data.frame()
income_nn_probs %>% head()

pred_nn <- ROCR::prediction(income_nn_probs$V1, test$income)
perf_nn <- ROCR::performance(pred_nn, 'tpr', 'fpr')
perf_nn_df <- data.frame(perf_nn@x.values, perf_nn@y.values, perf_nn@alpha.values)
names(perf_nn_df) <- c("fpr", "tpr", "cut")
perf_nn_df %>% head()

roc_nn <- perf_nn_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_nn@y.name) + xlab(perf_nn@x.name) + ggtitle("Neural Network")
roc_nn

# Null Model --------------------------------------------------------------

mod_null <- glm(income ~ 1, data = train, family = binomial)
income_null_probs <- mod_null %>% 
  predict(newdata = test, type = "response") %>% 
  as.data.frame()
income_null_probs %>% head()

pred_null <- ROCR::prediction(income_null_probs$., test$income)
perf_null <- ROCR::performance(pred_null, 'tpr', 'fpr')
perf_null_df <- data.frame(perf_null@x.values, perf_null@y.values, perf_null@alpha.values)
names(perf_null_df) <- c("fpr", "tpr", "cut")
perf_null_df %>% head()

roc_null <- perf_null_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_null@y.name) + xlab(perf_null@x.name) + ggtitle("Null Model")
roc_null

# Random Forest -----------------------------------------------------------

mod_forest <- randomForest(form, data = train, ntree = 201, mtry = 3)
income_forest_probs <- mod_forest %>% 
  predict(newdata = test, type = "prob") %>% 
  as.data.frame()
income_forest_probs %>% head()

pred_forest <- ROCR::prediction(income_forest_probs$` >50K`, test$income)
perf_forest <- ROCR::performance(pred_forest, 'tpr', 'fpr')
perf_forest_df <- data.frame(perf_forest@x.values, perf_forest@y.values, perf_forest@alpha.values)
names(perf_forest_df) <- c("fpr", "tpr", "cut")
perf_forest_df %>% head()

roc_forest <- perf_forest_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_forest@y.name) + xlab(perf_forest@x.name) + ggtitle("Random Forest")
roc_forest

# Graphing All Models -----------------------------------------------------

grid.arrange(roc_forest, roc_knn, roc_nb, roc_nn, roc_null, roc_tree)

ggplot() + 
  geom_line(data = perf_tree_df, aes(x = fpr, y = tpr), color = "red") + 
  geom_line(data = perf_knn_df, aes(x = fpr, y = tpr), color = "orange") + 
  geom_line(data = perf_nb_df, aes(x = fpr, y = tpr), color = "green") + 
  geom_line(data = perf_nn_df, aes(x = fpr, y = tpr), color = "blue") + 
  geom_line(data = perf_null_df, aes(x = fpr, y = tpr), color = "purple") + 
  geom_line(data = perf_forest_df, aes(x = fpr, y = tpr), color = "black")
