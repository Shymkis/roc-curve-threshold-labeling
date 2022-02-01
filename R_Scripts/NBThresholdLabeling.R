rm(list=ls())

library(e1071)
library(plotly)
library(ROCR)

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
mod_nb <- naiveBayes(form, data = train)
income_probs <- mod_nb %>%
  predict(newdata = train, type = "raw") %>%
  as.data.frame()

pred <- ROCR::prediction(income_probs[, 2], train$income)
perf <- ROCR::performance(pred, 'tpr', 'fpr')
perf_df <- data.frame(perf@x.values, perf@y.values, perf@alpha.values)
names(perf_df) <- c("fpr", "tpr", "cut")
roc <- perf_df %>% ggplot(aes(x = fpr, y = tpr, z = cut)) +
  geom_line() + geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf@y.name) + xlab(perf@x.name)
roc %>% ggplotly(tooltip = "z")
