# Robust trend estimation using the Theil-Sen estimator #

# Take all pairs of points -- compute the slope of the line that omly passes through them
# Now take the median of all slopes ; this is the slope of the estimated line, m

# The intercept is the median of y_i - m * x_i

# Original data

# The idea here is to take many small samples of the data without replacement, get medians from each such sample, and then find a median of all medians and use that as the
# median slope.

# the rest of the process is same 

require(tidyverse)
require(readr)
require(ggplot2)
require(dplyr)

gen_step_data <- function(N) {
  
N <- 100
x <- c(1:N)
N1 <- round(2*N / 3)
N2 <- N - N1

x1 <- c(1:N1)
x2 <- c((N1+1):N)

y1 <- rep(1000, N1)
y2 <- rep(5000, N2) # step change

x <- c(x1, x2)
y <- c(y1, y2)

orig_data_t <- tibble(
  x = x,
  y = y
)

# corrupt the data with noise and some outliers

corrupt <- which(abs(x) %% 15  == 0)

# to avoid cases of infnite slope where x's are precisely same, a small x jitter helps!

x <- x + runif(length(x), 0, 0.01)

# scenario 1 : noise is uniform
y_noisy <- y + runif(length(y), 0, 10) # all points are jittered slightly as in real data

# scenario 2 : noise is Gaussian
#y_noisy <- y + rnorm(length(y), 0, 30) # all points are jittered 

# the outliers are now chosen
y_noisy[corrupt] <- runif(length(corrupt), 50, 8000) # large deviation here

corrupt_data_t <- tibble(
  x = x,
  y = y_noisy
)


return(corrupt_data_t)

} # generate step data

gen_typical_data <- function(N) {
  
  N <- 100
  x <- c(1:N)
  m_orig <- -0.5
  intercept_orig <- 1
  y <- m_orig * x + intercept_orig
  
  orig_data_t <- tibble(
    x = x,
    y = y
  )
  
  # to avoid cases of infnite slope where x's are precisely same, a small x jitter helps!
  
  x <- x + runif(length(x), 0, 0.01)
  
  # corrupt the data with noise and some outliers
  
  # corrupt every fifth point
  corrupt <- which(abs(y) %% 4 == 0)
  
  # scenario 1 : noise is uniform
  y_noisy <- y + runif(length(y), 0, 30) # all points are jittered slightly as in real data
  
  # scenario 2 : noise is Gaussian
  #y_noisy <- y + rnorm(length(y), 0, 30) # all points are jittered 
  
  # the outliers are now chosen
  y_noisy[corrupt] <- runif(length(corrupt), -200, 200) # large deviation here
  
  corrupt_data_t <- tibble(
    x = x,
    y = y_noisy
  )
  
  return(corrupt_data_t)
}

gen_pathological_data <- function(N) {
  
  N1 <- N-1
  
  intercept_orig <- 1
  x1 <- seq(1, N1, by=2)
  x2 <- seq(2, N, by = 2)
  m_orig_x1 <- -0.5
  m_orig_x2 <- 0.5
  
  y1 <- m_orig_x1 * x1 + intercept_orig
  y2 <- m_orig_x2 * x2 + intercept_orig
  
  x <- c(x1, x2)
  y <- c(y1, y2)
  
  orig_data_t <- tibble(
    x = x,
    y = y
  )
  
  # to avoid cases of infnite slope where x's are precisely same, a small x jitter helps!
  
  x <- x + runif(length(x), 0, 0.01)
  
  # corrupt the data with noise and some outliers
  
  # corrupt every fifth point
  corrupt <- which(abs(y) %% 4 == 0)
  
  # scenario 1 : noise is uniform
  y_noisy <- y + runif(length(y), 0, 10) # all points are jittered slightly as in real data
  
  # scenario 2 : noise is Gaussian
  #y_noisy <- y + rnorm(length(y), 0, 30) # all points are jittered 
  
  # the outliers are now chosen
  y_noisy[corrupt] <- runif(length(corrupt), -200, 200) # large deviation here
  
  corrupt_data_t <- tibble(
    x = x,
    y = y_noisy
  )
  
  
}

corrupt_data_t <- gen_step_data(100)

#corrupt_data_t <- gen_typical_data(100)

#corrupt_data_t <- gen_pathological_data(100)

# plot 

corrupt_data_t %>%
  ggplot(aes(x = x, y = y)) + geom_point(colour = "red", size = 2)

# Theil-Sen estimator for slope 

nrow1 <- nrow(corrupt_data_t)

num_samples <- nrow1 * (nrow1 - 1) / 2

sample_nrow1 <- round(nrow1 / 30) # our samples 

num_samples_2 <- sample_nrow1 * (sample_nrow1 - 1) / 2

ratio <- round(num_samples / num_samples_2)

ctr_samples <- 50*ratio

median_slope <- numeric(length = ctr_samples)

for (ctrs in 1:ctr_samples) {

    cat("sample number ", ctrs, "\n")

# get the data sample 
    data_sample <- corrupt_data_t[sample(nrow1, sample_nrow1, replace = F), ] # pick your data sample for this round of sampling without replacement 

slopes <- numeric(length = num_samples_2)

ctr <- 1
for (pt1 in 1:(sample_nrow1 - 1)) {
  for (pt2 in (pt1+1):sample_nrow1) {
    x1 <- data_sample[pt1,1]
    y1 <- data_sample[pt1,2]
    x2 <- data_sample[pt2,1]
    y2 <- data_sample[pt2,2]
    
    slopes[ctr] <- (y2  - y1) / (x2 - x1)
    ctr <- ctr + 1
  }
}

# find the median

median_slope[ctrs] <- median(unlist(slopes))

}

# now get the median of medians

median_slope_2 <- median(unlist(median_slope))

intercepts <- numeric(length = nrow1)

for (pt1 in 1:nrow1) {
  intercepts[pt1] <- corrupt_data_t[pt1,2] - median_slope_2 * corrupt_data_t[pt1,1]
}

median_intercept <- median(unlist(intercepts))

# finally we can plot our robust estimator line

corrupt_data_t[,"fitline"] <- median_intercept + median_slope_2 * corrupt_data_t[,1]

# subtract the fit from the data to find residuals 

corrupt_data_t[,"residuals"] <- corrupt_data_t[,2] - corrupt_data_t[,3] # residuals

# rename the last two columns

# plot 

corrupt_data_t %>%
  ggplot() + geom_point(aes(x = x, y = y), colour = "red") + geom_line(aes(x = x, y = fitline), colour = "black")

# plot the residuals
corrupt_data_t %>%
  ggplot() + geom_point(aes(x = x, y = residuals), colour = "red")

dirName <- "/Users/girishnathan/work/code/R/robust_trend_estimation/results/"

fileName <- paste(dirName, "trend_fit_pathological_using_many_samples.pdf", sep="")

pdf(fileName)
corrupt_data_t %>%
  ggplot() + geom_point(aes(x = x, y = y), colour = "red") + geom_line(aes(x = x, y = fitline), colour = "black")
dev.off()

fileName2 <- paste(dirName, "trend_fit_pathological_plot_residuals_using_many_samples.pdf", sep="")

pdf(fileName2)
corrupt_data_t %>%
  ggplot() + geom_point(aes(x = x, y = residuals), colour = "red")
dev.off()


