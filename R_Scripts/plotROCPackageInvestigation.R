rm(list=ls())

library(mdsr)
library(rpart)
library(partykit)
library(rattle)
library(randomForest)
library(e1071)
library(class)
library(nnet)
library(ROCR)
library(plotROC)
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

test_income_binary <- ifelse(test$income == ' <=50K', 0, 1)

# Decision Tree -----------------------------------------------------------

mod_tree <- rpart(form, data = train)
income_tree_probs <- mod_tree %>% 
  predict(newdata = test, type = "prob")
income_tree_probs <- income_tree_probs[, ' >50K']

mod_tree %>% fancyRpartPlot()

pred_tree <- ROCR::prediction(income_tree_probs, test$income)
perf_tree <- ROCR::performance(pred_tree, 'tpr', 'fpr')
perf_tree_df <- data.frame(perf_tree@x.values, perf_tree@y.values, perf_tree@alpha.values)
names(perf_tree_df) <- c("fpr", "tpr", "cut")

rocr_tree <- perf_tree_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_tree@y.name) + xlab(perf_tree@x.name) + ggtitle("Decision Tree ROCR")
rocr_tree

plotroc_tree_df <- data.frame(test_income_binary, income_tree_probs)
names(plotroc_tree_df) <- c('D', 'M')
plotroc_tree_df %>% head()

plotroc_tree_1 <- plotroc_tree_df %>% ggplot(aes(d = D, m = M)) +
  geom_roc(labelround = 2) + ggtitle("Decision Tree D & M Data Frame")
plotroc_tree_1

plotroc_tree_2 <- perf_tree_df %>% ggplot(aes(x = fpr, y = tpr, label = cut)) +
  geom_roc(stat = "identity", labelround = 2) + ggtitle("Decision Tree FPR, TPR, & Cut Data Frame")
plotroc_tree_2

plotroc_tree_1 %>% plot_interactive_roc()

plotroc_tree_2 %>% plot_interactive_roc()

calc_auc(plotroc_tree_1)["AUC"]

calc_auc(plotroc_tree_2)["AUC"]

# K-Nearest Neighbors -----------------------------------------------------

train_q <- train %>%
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
test_q <- test %>% 
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
income_knn <- knn(train_q, test = test_q, cl = train$income, k = 10, prob = TRUE)
income_knn_probs <- ifelse(income_knn == ' >50K', attr(income_knn, 'prob'), 1 - attr(income_knn, 'prob'))

pred_knn <- ROCR::prediction(income_knn_probs, test$income)
perf_knn <- ROCR::performance(pred_knn, 'tpr', 'fpr')
perf_knn_df <- data.frame(perf_knn@x.values, perf_knn@y.values, perf_knn@alpha.values)
names(perf_knn_df) <- c("fpr", "tpr", "cut")

rocr_knn <- perf_knn_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_knn@y.name) + xlab(perf_knn@x.name) + ggtitle("K-Nearest Neighbors")
rocr_knn

plotroc_knn_df <- data.frame(test_income_binary, income_knn_probs)
names(plotroc_knn_df) <- c('D', 'M')
plotroc_knn_df %>% head()

plotroc_knn_1 <- plotroc_knn_df %>% ggplot(aes(d = D, m = M)) +
  geom_roc(labelround = 2) + ggtitle("K-Nearest Neighbors D & M Data Frame")
plotroc_knn_1

plotroc_knn_2 <- perf_knn_df %>% ggplot(aes(x = fpr, y = tpr, label = cut)) +
  geom_roc(stat = "identity", labelround = 2) + ggtitle("K-Nearest Neighbors FPR, TPR, & Cut Data Frame")
plotroc_knn_2

plotroc_knn_1 %>% plot_interactive_roc()

plotroc_knn_2 %>% plot_interactive_roc()

calc_auc(plotroc_knn_1)["AUC"]

calc_auc(plotroc_knn_2)["AUC"]

# Naive Bayes -------------------------------------------------------------

mod_nb <- naiveBayes(form, data = train)
income_nb_probs <- mod_nb %>%
  predict(newdata = test, type = "raw")
income_nb_probs <- income_nb_probs[, ' >50K']

pred_nb <- ROCR::prediction(income_nb_probs, test$income)
perf_nb <- ROCR::performance(pred_nb, 'tpr', 'fpr')
perf_nb_df <- data.frame(perf_nb@x.values, perf_nb@y.values, perf_nb@alpha.values)
names(perf_nb_df) <- c("fpr", "tpr", "cut")
perf_nb_df %>% head()

rocr_nb <- perf_nb_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_nb@y.name) + xlab(perf_nb@x.name) + ggtitle("Naive Bayes ROCR")
rocr_nb

plotroc_nb_df <- data.frame(test_income_binary, income_nb_probs)
names(plotroc_nb_df) <- c('D', 'M')
plotroc_nb_df %>% head()

plotroc_nb_1 <- plotroc_nb_df %>% ggplot(aes(d = D, m = M)) +
  geom_roc(labelround = 5) + ggtitle("Naive Bayes D & M Data Frame")
plotroc_nb_1

plotroc_nb_2 <- perf_nb_df %>% ggplot(aes(x = fpr, y = tpr, label = cut)) +
  geom_roc(stat = "identity", labelround = 5) + ggtitle("Naive Bayes FPR, TPR, & Cut Data Frame")
plotroc_nb_2

plotroc_nb_1 %>% plot_interactive_roc()

plotroc_nb_2 %>% plot_interactive_roc()

calc_auc(plotroc_nb_1)["AUC"]

calc_auc(plotroc_nb_2)["AUC"]

# Neural Network ----------------------------------------------------------

mod_nn <- nnet(form, data = train, size = 5)
income_nn_probs <- mod_nn %>% 
  predict(newdata = test, type = "raw")
income_nn_probs <- income_nn_probs[, 1]

pred_nn <- ROCR::prediction(income_nn_probs, test$income)
perf_nn <- ROCR::performance(pred_nn, 'tpr', 'fpr')
perf_nn_df <- data.frame(perf_nn@x.values, perf_nn@y.values, perf_nn@alpha.values)
names(perf_nn_df) <- c("fpr", "tpr", "cut")
perf_nn_df %>% head()

rocr_nn <- perf_nn_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_nn@y.name) + xlab(perf_nn@x.name) + ggtitle("Neural Network ROCR")
rocr_nn

plotroc_nn_df <- data.frame(test_income_binary, income_nn_probs)
names(plotroc_nn_df) <- c('D', 'M')
plotroc_nn_df %>% head()

plotroc_nn_1 <- plotroc_nn_df %>% ggplot(aes(d = D, m = M)) +
  geom_roc(labelround = 3) + ggtitle("Neural Network D & M Data Frame")
plotroc_nn_1

plotroc_nn_2 <- perf_nn_df %>% ggplot(aes(x = fpr, y = tpr, label = cut)) +
  geom_roc(stat = "identity", labelround = 3) + ggtitle("Neural Network FPR, TPR, & Cut Data Frame")
plotroc_nn_2

plotroc_nn_1 %>% plot_interactive_roc()

plotroc_nn_2 %>% plot_interactive_roc()

calc_auc(plotroc_nn_1)["AUC"]

calc_auc(plotroc_nn_2)["AUC"]

# Random Forest -----------------------------------------------------------

mod_forest <- randomForest(form, data = train, ntree = 201, mtry = 3)
income_forest_probs <- mod_forest %>% 
  predict(newdata = test, type = "prob")
income_forest_probs <- income_forest_probs[, ' >50K']

pred_forest <- ROCR::prediction(income_forest_probs, test$income)
perf_forest <- ROCR::performance(pred_forest, 'tpr', 'fpr')
perf_forest_df <- data.frame(perf_forest@x.values, perf_forest@y.values, perf_forest@alpha.values)
names(perf_forest_df) <- c("fpr", "tpr", "cut")
perf_forest_df %>% head()

rocr_forest <- perf_forest_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_forest@y.name) + xlab(perf_forest@x.name) + ggtitle("Random Forest ROCR")
rocr_forest

plotroc_forest_df <- data.frame(test_income_binary, income_forest_probs)
names(plotroc_forest_df) <- c('D', 'M')
plotroc_forest_df %>% head()

plotroc_forest_1 <- plotroc_forest_df %>% ggplot(aes(d = D, m = M)) +
  geom_roc(labelround = 2) + ggtitle("Random Forest D & M Data Frame")
plotroc_forest_1

plotroc_forest_2 <- perf_forest_df %>% ggplot(aes(x = fpr, y = tpr, label = cut)) +
  geom_roc(stat = "identity", labelround = 2) + ggtitle("Random Forest FPR, TPR, & Cut Data Frame")
plotroc_forest_2

plotroc_forest_1 %>% plot_interactive_roc()

plotroc_forest_2 %>% plot_interactive_roc()

calc_auc(plotroc_forest_1)["AUC"]

calc_auc(plotroc_forest_2)["AUC"]

# Graphing All Models -----------------------------------------------------


rocr_all <- list(rocr_forest, rocr_knn, rocr_nb, rocr_nn, rocr_tree)
plotroc_all_list <- list(plotroc_forest_1, plotroc_knn_1, plotroc_nb_1, plotroc_nn_1, plotroc_tree_1)
lapply(plotroc_all_list, calc_auc)

plotroc_all_df <- data.frame(
  D = c(plotroc_forest_df$D, plotroc_knn_df$D, plotroc_nb_df$D, plotroc_nn_df$D, plotroc_tree_df$D),
  M = c(plotroc_forest_df$M, plotroc_knn_df$M, plotroc_nb_df$M, plotroc_nn_df$M, plotroc_tree_df$M),
  type = rep(c("RF", "KNN", "NB", "NN", "DT"), each = nrow(test))
)

plotroc_all_1 <- plotroc_all_df %>% ggplot(aes(d = D, m = M, color = type)) +
  geom_roc(n.cuts = 7, labelround = 2, show.legend = FALSE)
plotroc_all_1

plotroc_all_1 %>% plot_interactive_roc()

grid.arrange(rocr_forest, rocr_knn, rocr_nb, rocr_nn, rocr_tree)
grid.arrange(plotroc_forest_1, plotroc_knn_1, plotroc_nb_1, plotroc_nn_1, plotroc_tree_1)

rocr_all <- ggplot() + 
  geom_line(data = perf_tree_df, aes(x = fpr, y = tpr), color = "red") + 
  geom_line(data = perf_knn_df, aes(x = fpr, y = tpr), color = "orange") + 
  geom_line(data = perf_nb_df, aes(x = fpr, y = tpr), color = "green") + 
  geom_line(data = perf_nn_df, aes(x = fpr, y = tpr), color = "blue") + 
  geom_line(data = perf_forest_df, aes(x = fpr, y = tpr), color = "purple")

grid.arrange(plotroc_all_1, rocr_all)
