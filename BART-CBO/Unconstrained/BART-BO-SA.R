# -------------------------------------------------------------
# Libraries
# -------------------------------------------------------------
library(BART)

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
# 2. Fit initial BART surrogate
# -------------------------------------------------------------
bart_fit <- wbart(x.train = X, y.train = y,
                  ndpost = 1000, nskip = 200, ntree = 50)

# -------------------------------------------------------------
# 3. EI(x) function
# -------------------------------------------------------------
EI_fun <- function(x_vec) {
  
  x_mat <- matrix(as.numeric(x_vec), nrow = 1)
  
  pred_draws <- predict(bart_fit, newdata = x_mat)
  
  if (nrow(pred_draws) < ncol(pred_draws)) {
    pred_draws <- t(pred_draws)
  }
  
  mu <- mean(pred_draws)
  sigma <- sd(pred_draws)
  f_best <- max(y)
  
  if (sigma == 0) return(0)
  
  z <- (mu - f_best) / sigma
  ei <- (mu - f_best) * pnorm(z) + sigma * dnorm(z)
  if (ei < 0) ei <- 0
  return(ei)
}

# -------------------------------------------------------------
# 4. Simulated Annealing optimizer for EI
# -------------------------------------------------------------
SA_optimize_EI <- function(EI_fun, n_bits,
                           max_iter = 2000,
                           T_start = 1.0,
                           T_end = 1e-4,
                           alpha = 0.995) {
  
  # flip one bit
  flip_neighbor <- function(x) {
    idx <- sample(seq_len(length(x)), 1)
    x_new <- x
    x_new[idx] <- 1 - x_new[idx]
    return(x_new)
  }
  
  # initial point
  x_curr <- rbinom(n_bits, 1, 0.5)
  EI_curr <- EI_fun(x_curr)
  
  x_best <- x_curr
  EI_best <- EI_curr
  
  T <- T_start
  
  for (iter in 1:max_iter) {
    
    x_new <- flip_neighbor(x_curr)
    EI_new <- EI_fun(x_new)
    
    if (EI_new >= EI_curr) {
      accept <- TRUE
    } else {
      prob <- exp((EI_new - EI_curr) / T)
      accept <- runif(1) < prob
    }
    
    if (accept) {
      x_curr <- x_new
      EI_curr <- EI_new
    }
    
    if (EI_curr > EI_best) {
      x_best <- x_curr
      EI_best <- EI_curr
    }
    
    T <- max(T_end, T * alpha)
  }
  
  list(x_best = x_best, EI_best = EI_best)
}

# -------------------------------------------------------------
# 5. Bayesian Optimization Loop (50 iterations)
# -------------------------------------------------------------
for (i in 1:50) {
  
  SA_res <- SA_optimize_EI(EI_fun, n_bits = p,
                           max_iter = 2000,
                           T_start = 1.0,
                           T_end = 1e-4,
                           alpha = 0.995)
  
  x_next <- as.numeric(SA_res$x_best)
  cat("Next binary candidate:", x_next, "\n")
  
  # -------------------------------------------------------------
  # Evaluate the new point and update model
  # -------------------------------------------------------------
  y_next <- true_f(x_next)
  X <- rbind(X, x_next)
  y <- c(y, y_next)
  
  bart_fit <- wbart(x.train = X, y.train = y,
                    ndpost = 1000, nskip = 200, ntree = 50)
  
  cat("Updated dataset size:", nrow(X), "\n")
}

# -------------------------------------------------------------
# GA
# -------------------------------------------------------------

GA_run <- ga(type = "binary", fitness = true_f, nBits = p,
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
ga_result <- list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)

