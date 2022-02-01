rm(list=ls())

library(mdsr)
library(e1071)
library(plotly)
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
mod_nb <- naiveBayes(form, data = train)
income_probs <- mod_nb %>%
  predict(newdata = train, type = "raw") %>%
  as.data.frame()

pred <- ROCR::prediction(income_probs[, 2], train$income)
perf <- ROCR::performance(pred, 'tpr', 'fpr')
perf_df <- data.frame(perf@x.values, perf@y.values, perf@alpha.values)
names(perf_df) <- c("fpr", "tpr", "cut")

distinct_probs <- income_probs %>% distinct(` >50K`) %>% arrange(desc(` >50K`)) %>% select(` >50K`)
pred_cutoffs <- pred@cutoffs[[1]]

distinct_probs %>% nrow()
pred_cutoffs %>% length()

count <- 0
for (i in 2:length(pred_cutoffs)) {
  if (distinct_probs$` >50K`[i - 1] != pred_cutoffs[i]) {
    count <- count + 1
  }
}
count

identical(distinct_probs$` >50K`, pred_cutoffs[-1])

pred_cutoffs[1]

identical(pred_cutoffs, perf@alpha.values[[1]])

idx <- 4000
perf_cutoff <- perf@alpha.values[[1]][idx]
perf_fpr <- perf@x.values[[1]][idx]
perf_tpr <- perf@y.values[[1]][idx]

income_nb <- ifelse(income_probs$` >50K` >= perf_cutoff, ' >50K', ' <=50K') %>% as.factor()
confusion <- tally(income_nb ~ train$income)
confusion[2,1] / sum(confusion[,1]) == perf_fpr
confusion[2,2] / sum(confusion[,2]) == perf_tpr

