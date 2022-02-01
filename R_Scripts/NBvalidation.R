# Init --------------------------------------------------------------------

rm(list=ls())

library(mdsr)
library(e1071)
library(ROCR)
library(pROC)
library(gridExtra)

# Datasets ----------------------------------------------------------------

census <- read.csv(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  header = FALSE)
names(census) <- c("age", "workclass", "fnlwgt", "education",
                   "education.num", "marital.status", "occupation", "relationship",
                   "race", "sex", "capital.gain", "capital.loss", "hours.per.week",
                   "native.country", "income")
glimpse(census)

set.seed(364)
n <- nrow(census)
test_idx <- sample.int(n, size = round(0.2 * n))
train <- census[-test_idx,]
nrow(train)

test <- census[test_idx,]
nrow(test)

# Naive Bayes -------------------------------------------------------------

form <- as.formula("income ~ age + workclass + education + marital.status +
  occupation + relationship + race + sex + capital.gain + capital.loss +
  hours.per.week")

mod_nb <- naiveBayes(form, data = train)
income_nb <- predict(mod_nb, newdata = train)
confusion <- tally(income_nb ~ income, data = train, format = "count")
confusion

sum(diag(confusion)) / nrow(train)

income_probs <- mod_nb %>%
  predict(newdata = train, type = "raw") %>%
  as.data.frame()
head(income_probs, 3)

names(income_probs)

tally(~` >50K` > 0.5, data = income_probs, format = "percent")

tally(~` >50K` > 0.24, data = income_probs, format = "percent")

pred <- ROCR::prediction(income_probs[, 2], train$income)
perf <- ROCR::performance(pred, 'tpr', 'fpr')
class(perf) # can also plot(perf)

perf_df <- data.frame(perf@x.values, perf@y.values, perf@alpha.values)
names(perf_df) <- c("fpr", "tpr", "cut")
roc <- perf_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf@y.name) + xlab(perf@x.name)

confusion <- tally(income_nb ~ income, data = train, format = "count")
confusion

sum(diag(confusion)) / nrow(train)

tpr <- confusion[" >50K", " >50K"] / sum(confusion[, " >50K"])
fpr <- confusion[" >50K", " <=50K"] / sum(confusion[, " <=50K"])
roc + geom_point(x = fpr, y = tpr, size = 3)

# Comparing ROC Curves ----------------------------------------------------

iter <- 1000
cut_matrix <- matrix(nrow = iter, ncol = 4)
for(k in 1:iter) {
  income_nb_cut <- ifelse(income_probs$` >50K` > k/(iter + 1), " >50K", " <=50K") %>% as.factor()

  confusion <- tally(income_nb_cut ~ income, data = train, format = "count")
  
  tpr <- confusion[" >50K", " >50K"] / sum(confusion[, " >50K"])
  fpr <- confusion[" >50K", " <=50K"] / sum(confusion[, " <=50K"])

  cut_matrix[k, 1] <- k/(iter + 1)
  cut_matrix[k, 2] <- sum(diag(confusion)) / nrow(train)
  cut_matrix[k, 3] <- tpr
  cut_matrix[k, 4] <- fpr
}
cut_df <- cut_matrix %>% as.data.frame()
names(cut_df) <- c("cut", "accuracy", "tpr", "fpr")
pairs(cut_df, pch = 19)
(best_cut <- cut_df %>% top_n(1, accuracy) %>% select(cut) %>% as.numeric())
(best_acc <- cut_df %>% top_n(1, accuracy) %>% select(accuracy) %>% as.numeric())

income_nb_cut <- ifelse(income_probs$` >50K` > best_cut, " >50K", " <=50K") %>% as.factor()

confusion <- tally(income_nb_cut ~ income, data = train, format = "count")
confusion

tpr <- confusion[" >50K", " >50K"] / sum(confusion[, " >50K"])
fpr <- confusion[" >50K", " <=50K"] / sum(confusion[, " <=50K"])
roc1 <- perf_df %>% filter(cut >= 1/(iter + 1) & cut <= iter/(iter + 1)) %>% 
  ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf@y.name) + xlab(perf@x.name) +
  geom_point(x = fpr, y = tpr, size = 3)
roc2 <- cut_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf@y.name) + xlab(perf@x.name) +
  geom_point(x = fpr, y = tpr, size = 3, color = "black")
grid.arrange(roc1, roc2)

roc3 <- perf_df %>% filter(cut >= 1/(iter + 1) & cut <= iter/(iter + 1)) %>% 
  ggplot(aes(x = cut, y = tpr)) +
  geom_line() + ylab(perf@y.name) + xlab(perf@alpha.name)
roc4 <- cut_df %>% ggplot(aes(x = cut, y = tpr)) +
  geom_line() + ylab(perf@y.name) + xlab(perf@alpha.name)
grid.arrange(roc3, roc4)

roc5 <- perf_df %>% filter(cut >= 1/(iter + 1) & cut <= iter/(iter + 1)) %>% 
  ggplot(aes(x = cut, y = fpr)) +
  geom_line() + ylab(perf@x.name) + xlab(perf@alpha.name)
roc6 <- cut_df %>% ggplot(aes(x = cut, y = fpr)) +
  geom_line() + ylab(perf@x.name) + xlab(perf@alpha.name)
grid.arrange(roc5, roc6)

grid.arrange(roc1, roc2, roc3, roc4, roc5, roc6)

# It's apparent that the cutoff/alpha values in the performance object represents the threshold values

