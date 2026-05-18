# -------------------------------------------------------------
# Libraries
# -------------------------------------------------------------
library(BART)
library(GA)

# -------------------------------------------------------------
# 1. Generate binary training data
# -------------------------------------------------------------
set.seed(1)
p <- 8                      # number of binary variables
n_init <- 10                # number of initial points
X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)

# true objective (toy example)
true_f <- function(x) {
  stopifnot(length(x) == 8)
  linear_part <- sum(c(1.2, -0.8, 0.5, 1.0, -1.5, 0.3, 0.7, -0.4) * x)
  interaction_part <- 0.8 * x[1] * x[2] - 0.6 * x[3] * x[4] + 0.4 * x[5] * x[6]
  nonlinear_part <- 0.5 * sin(pi * sum(x) / 8)
  return(linear_part + interaction_part + nonlinear_part)
}

y <- apply(X, 1, true_f) 

# -------------------------------------------------------------
# 2. Fit BART surrogate
# -------------------------------------------------------------
bart_fit <- wbart(x.train = X, y.train = y,
                  ndpost = 1000, nskip = 200, ntree = 50)

# -------------------------------------------------------------
# 3. EI optimization directly in combinatorial space
# -------------------------------------------------------------
f_best <- max(y)

# Binary GA to maximize EI(x)
fitness <- function(x_vec) {
  # reshape as matrix (1 x p)
  x_mat <- matrix(as.numeric(x_vec), nrow = 1)
  
  # get posterior draws
  pred_draws <- predict(bart_fit, newdata = x_mat)
  
  # ensure correct orientation
  if (nrow(pred_draws) < ncol(pred_draws)) {
    pred_draws <- t(pred_draws)
  }
  
  mu <- mean(pred_draws)
  sigma <- sd(pred_draws)
  
  if (sigma == 0) return(0)
  
  z <- (mu - f_best) / sigma
  ei <- (mu - f_best) * pnorm(z) + sigma * dnorm(z)
  if (ei < 0) ei <- 0
  return(ei)
}
for(i in 1:50){
  
  GA_res <- ga(
    type = "binary",
    nBits = p,
    fitness = fitness,
    popSize = 60,
    maxiter = 200,
    run = 40,
    keepBest = TRUE
  )
  
  x_next <- as.numeric(GA_res@solution[1, ])
  cat("Next binary candidate:", x_next, "\n")
  
  # -------------------------------------------------------------
  # 4. Evaluate the new point and update the model
  # -------------------------------------------------------------
  y_next <- true_f(x_next) 
  X <- rbind(X, x_next)
  y <- c(y, y_next)
  
  bart_fit <- wbart(x.train = X, y.train = y,
                    ndpost = 1000, nskip = 200, ntree = 50)
  
  cat("Updated dataset size:", nrow(X), "\n")
}

GA_run <- ga(type = "binary", fitness = true_f, nBits = p,
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
ga_result <- list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)
