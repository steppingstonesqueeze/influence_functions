# What happens to classification accuracy and other classification metrics
# for a simple binary classification problem as training points removed at random?

# This experiment shows the metrics as function of a fraction of points removed
# by averaging over 100 realizations for each fraction

library(glmnet)
library(ggplot2)
library(tidyr)
library(randomForest)
library(ROCR)

EPS <- 0
margin1 <- 0.2

gen_all_data <- function(N, x_lim, y_lim) {
  # data is generated in each dim from random dist with lims -x_lim,x_lim and 
  #-y_lim, y_lim
  
  output <- list()
  
  output[[1]] <- runif(N, -x_lim, x_lim)
  output[[2]] <- runif(N, -y_lim, y_lim)
  
  return(output)
  
}

train_test_split <- function(df, N, N_train, N_test) {
  df <- df[sample(N),] # a shuffle just in case there is some initial adversarial ordering
  df_train <- df[1:N_train, ]
  df_test <- df[((N_train+1:N)), ]
  
  output <- list()
  
  output[[1]] <- df_train
  output[[2]] <- df_test
  
  return(output)
}

# two features x and y -- linear decision boundary at y = x

# data params  #

N <- 10000
train_split <- 0.8
N_train <- round(train_split * N)
N_test <- N - N_train
x_limit <- 10
y_limit <- 10

# generate train data - very simple for this first simple model #

features_data <- gen_all_data(N, x_limit, y_limit)

full_df <- data.frame(
  x = rep(0, N),
  y = rep(0, N),
  label = rep(0, N)
)

full_df$x <- features_data[[1]]
full_df$y <- features_data[[2]]

# now generate label as per the simple rule : if x > y label 1, x < y label 0
# in particular - x >= y + EPS label 1 x <= y-EPS label 0

#full_df$label <- with(
#  full_df, 
#  full_df$label <- 1 * ((full_df$y - full_df$x) < EPS)
#)

# label generation rule 2 : if y-x > 0 (or y > x), label 0 with a probability p
# 1 with 1-p ; if y-x < 0 or y < x label 1 with a probability p and 0 with 1-p

noise <- 0.2
p <- 1.0 - noise

# flip all
full_df$label <- ifelse(
  full_df$y - full_df$x > EPS,
  0,
  1
)

# add noise
full_df$label <- ifelse(
  runif(N, 0, 1) <= noise & abs(full_df$y - full_df$x) < margin1,
  abs(full_df$label - 1),
  full_df$label
)

#full_df$label <- ifelse(
#  full_df$y - full_df$x < EPS & runif(N, 0, 1) <= p,
#  1,
#  0
#)


# convert label into factor

full_df$label <- as.factor(full_df$label)

# get train and test data

train_test_data <- train_test_split(
  full_df,
  N,
  N_train,
  N_test
)

full_train_df <- train_test_data[[1]]
full_test_df <- train_test_data[[2]]

full_train_df <- full_train_df[complete.cases(full_train_df), ]
full_test_df <- full_test_df[complete.cases(full_test_df), ]

# completeness : plot the labels by color for both

g1 <- ggplot(data = full_train_df,
             aes(x = x, y = y)) + geom_point(aes(colour = label))
print(g1)

g2 <- ggplot(data = full_test_df,
             aes(x = x, y = y)) + geom_point(aes(colour = label))
print(g2)

# Fit a simple logistic regression model #

model1 <- glm(label ~ ., data = full_train_df, family = binomial)

# predict first on train data
model1.train.probs<- predict (model1, type="response")

# threshold the probs to 0 or 1 at 0.5
model1.train.pred <- rep(0, N_train)
model1.train.pred[model1.train.probs > 0.5] <- 1

#table(model1.train.pred, full_train_df$label)

# predict on test data

model1.test.probs <- predict(model1, full_test_df, type = "response")

# threshold the probs to 0 or 1 at 0.5

model1.test.pred <- rep(0, N_test)
model1.test.pred[model1.test.probs > 0.5] <- 1

#table(model1.test.pred, full_test_df$label)

# ROCR accuracy

model1.test.pred.obj <- prediction(
  model1.test.probs,
  full_test_df$label
)

test.acc <- performance(model1.test.pred.obj, measure = "acc")
acc <- max(!is.nan(test.acc@y.values[[1]]))

test.prec <- performance(model1.test.pred.obj, measure = "prec")
prec <- max(!is.nan(test.prec@y.values[[1]]))

test.rec <- performance(model1.test.pred.obj, measure = "rec")
rec <- max(!is.nan(test.rec@y.values[[1]]))

test.f <- performance(model1.test.pred.obj, measure = "f")
f <- max(!is.nan(test.f@y.values[[1]]))

test.auc <- performance(model1.test.pred.obj, measure = "auc")
auc <- max(!is.nan(test.auc@y.values[[1]]))

# now we eperiment w 100 runs per fraction level of points randomly removed

frac_present <- c(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

frac_removed <- 1.0 - frac_present

num_vals <- length(frac_present)

metrics_df <- data.frame(
  avg_acc = rep(0, num_vals),
  avg_prec = rep(0, num_vals),
  avg_rec = rep(0, num_vals),
  avg_f = rep(0, num_vals),
  avg_auc = rep(0, num_vals)
)

num_runs <- 100

ctr <- 1

for (fp in frac_present) {
  
  cat("frac present : ", fp, "\n")
  
  Np <- round(N_train * fp)
  
  avg_acc <- 0.0
  avg_prec <- 0.0
  avg_rec <- 0.0
  avg_f <- 0.0
  avg_auc <- 0.0
  
  for (runs in 1:num_runs) {
    
model1 <- glm(label ~ ., data = full_train_df[sample(N_train, Np, replace = F), ], 
              family = binomial)

# predict on test data

model1.test.probs <- predict(model1, full_test_df, type = "response")

model1.test.pred.obj <- prediction(
  model1.test.probs,
  full_test_df$label
)

test.acc <- performance(model1.test.pred.obj, measure = "acc")
acc <- max(!is.nan(test.acc@y.values[[1]]))

test.prec <- performance(model1.test.pred.obj, measure = "prec")
prec <- max(!is.nan(test.prec@y.values[[1]]))

test.rec <- performance(model1.test.pred.obj, measure = "rec")
rec <- max(!is.nan(test.rec@y.values[[1]]))

test.f <- performance(model1.test.pred.obj, measure = "f")
f <- max(!is.nan(test.f@y.values[[1]]))

test.auc <- performance(model1.test.pred.obj, measure = "auc")
auc <- max(!is.nan(test.auc@y.values[[1]]))

avg_acc <- avg_acc + acc
avg_prec <- avg_prec + prec
avg_rec <- avg_rec + rec
avg_f <- avg_f + f
avg_auc <- avg_auc + auc

} # loop over number of runs 
  
  avg_acc <- avg_acc / num_runs
  avg_prec <- avg_prec / num_runs
  avg_rec <- avg_rec / num_runs
  avg_f <- avg_f / num_runs
  avg_auc <- avg_auc / num_runs
  
  metrics_df[ctr,] <- c(avg_acc, avg_prec, avg_rec, avg_f, avg_auc)
  
  ctr <- ctr + 1
} # loop over fraction of data present
