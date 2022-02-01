rm(list=ls())

library(mdsr)
library(rpart)
library(randomForest)
library(e1071)
library(ROCR)
library(plotROC)
library(ROCit)
library(gridExtra)

census <- read.csv(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  header = FALSE)
names(census) <- c("age", "workclass", "fnlwgt", "education",
                   "education.num", "marital.status", "occupation",
                   "relationship", "race", "sex", "capital.gain",
                   "capital.loss", "hours.per.week", "native.country",
                   "income")
set.seed(364)
n <- nrow(census)
test_idx <- sample.int(n, size = round(0.2 * n))
train <- census[-test_idx,]
test <- census[test_idx,]

form <- as.formula("income ~ age + workclass + education + marital.status +
  occupation + relationship + race + sex + capital.gain + capital.loss +
  hours.per.week")

mod <- rpart(form, data = train)
income_probs <- mod %>% predict(newdata = test, type = "prob")
df <- data.frame(predictions = income_probs[, ' >50K'],
                 labels = ifelse(test$income == ' >50K', 1, 0))

mod <- naiveBayes(form, data = train)
income_probs <- mod %>% predict(newdata = test, type = "raw")
df <- data.frame(predictions = income_probs[, ' >50K'],
                 labels = ifelse(test$income == ' >50K', 1, 0))

mod <- randomForest(form, data = train, ntree = 201, mtry = 3)
income_probs <- mod %>% predict(newdata = test, type = "prob")
df <- data.frame(predictions = income_probs[, ' >50K'],
                 labels = ifelse(test$income == ' >50K', 1, 0))

# ROCR --------------------------------------------------------------------

# Default Plot
?plot.performance

rocr_pred <- prediction(df$predictions, df$labels)
rocr_perf <- performance(rocr_pred, 'tpr', 'fpr')
rocr_perf %>% plot(colorize = TRUE,
                   print.cutoffs.at = seq(0, 1, by = 0.2), add = TRUE)

rocr_auc <- performance(rocr_pred, 'auc')@y.values[[1]]
title(paste("ROCR: AUC = ", rocr_auc))

# Plot Using ggplot
rocr_df <- data.frame(fpr = rocr_perf@x.values[[1]],
                      tpr = rocr_perf@y.values[[1]],
                      cut = rocr_perf@alpha.values[[1]])
rocr_ggplot <- rocr_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(slope = 1, intercept = 0, lty = 3) +
  ggtitle(paste("ROCR: AUC = ", rocr_auc))
rocr_ggplot

# plotROC -----------------------------------------------------------------

# Plot Using ggplot
plotroc_plot <- df %>% ggplot(aes(m = predictions, d = labels)) +
  geom_roc(cutoffs.at = seq(0, 1, by = 0.2)) +
  style_roc(theme = theme_gray()) +
  geom_abline(slope = 1, intercept = 0, lty = 3)

plotroc_auc <- calc_auc(plotroc_plot)[['AUC']]
plotroc_ggplot <- plotroc_plot +
  ggtitle(paste("plotROC: AUC = ", plotroc_auc))
plotroc_ggplot

# ROCit -------------------------------------------------------------------

# Deault Plot
?plot.rocit

rocit_obj <- rocit(df$predictions, df$labels)
rocit_plot <- rocit_obj %>% plot()

rocit_auc <- rocit_obj[['AUC']]
title(paste("ROCit: AUC = ", rocit_auc))

# Plot Using ggplot
rocit_yindex <- rocit_plot[["optimal Youden Index point"]]
rocit_df <- data.frame(fpr = rocit_obj[['FPR']],
                       tpr = rocit_obj[['TPR']],
                       cut = rocit_obj[['Cutoff']])
rocit_ggplot <- rocit_df %>% ggplot(aes(x = fpr, y = tpr, color = cut)) +
  geom_line() + geom_abline(slope = 1, intercept = 0, lty = 3) +
  ggtitle(paste("ROCit: AUC = ", rocit_auc)) +
  geom_point(aes(x = rocit_yindex[['FPR']],
                 y = rocit_yindex[['TPR']]), size = 3)
rocit_ggplot

# Combining ROCit and ROCR ------------------------------------------------

rocit_obj %>% plot()
rocr_perf %>% plot(print.cutoffs.at = seq(0, 1, by = 0.2), add = TRUE)
title(paste("ROCit + ROCR: AUC = ", (rocit_auc + rocr_auc) / 2))
