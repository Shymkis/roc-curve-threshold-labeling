rm(list=ls())

library(mdsr)
library(class)
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

train_q <- train %>%
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
test_q <- test %>% 
  select(age, education.num, capital.gain, capital.loss, hours.per.week)

# Use.All = TRUE ----------------------------------------------------------

income_true <- knn(train_q, test = test_q, cl = train$income, k = 10, prob = TRUE, use.all = TRUE)

income_true_probs <- matrix(nrow = length(income_true), ncol = 1)
for(i in 1:length(income_true)) {
  p = attr(income_true, 'prob')[i]
  income_true_probs[i, 1] <- ifelse(income_true[i] == ' >50K', p, 1 - p)
}
income_true_probs <- income_true_probs %>% as.data.frame()
names(income_true_probs) <- c(' >50K')

income_true_probs %>% head()

pred_true <- ROCR::prediction(income_true_probs$` >50K`, test$income)
perf_true <- ROCR::performance(pred_true, 'tpr', 'fpr')
perf_true_df <- data.frame(perf_true@x.values, perf_true@y.values, perf_true@alpha.values)
names(perf_true_df) <- c("fpr", "tpr", "cut")
perf_true_df %>% head()

roc_true <- perf_true_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_true@y.name) + xlab(perf_true@x.name) + ggtitle("Use.All = TRUE")
roc_true

# Use.All = FALSE ---------------------------------------------------------

income_false <- knn(train_q, test = test_q, cl = train$income, k = 10, prob = TRUE, use.all = FALSE)

income_false_probs <- matrix(nrow = length(income_false), ncol = 1)
for(i in 1:length(income_false)) {
  p = attr(income_false, 'prob')[i]
  income_false_probs[i, 1] <- ifelse(income_false[i] == ' >50K', p, 1 - p)
}
income_false_probs <- income_false_probs %>% as.data.frame()
names(income_false_probs) <- c(' >50K')

income_false_probs %>% head()

pred_false <- ROCR::prediction(income_false_probs$` >50K`, test$income)
perf_false <- ROCR::performance(pred_false, 'tpr', 'fpr')
perf_false_df <- data.frame(perf_false@x.values, perf_false@y.values, perf_false@alpha.values)
names(perf_false_df) <- c("fpr", "tpr", "cut")
perf_false_df %>% head()

roc_false <- perf_false_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf_false@y.name) + xlab(perf_false@x.name) + ggtitle("Use.All = FALSE")
roc_false

# Graphing Both Models ----------------------------------------------------

grid.arrange(roc_true, roc_false)

ggplot() + 
  geom_line(data = perf_true_df, aes(x = fpr, y = tpr), color = "red") + 
  geom_line(data = perf_false_df, aes(x = fpr, y = tpr), color = "black")
