# -------------------------------------------------------------
# Constrained BART-BO for 0/1 Knapsack
# -------------------------------------------------------------
library(BART)
library(GA)

set.seed(1)

# -------------------------
# Problem / knapsack setup
# -------------------------
p <- 8                       # number of items / binary variables

# item values (v) and weights (w) - change to your instance
values <- c(10, 5, 15, 7, 6, 18, 3, 12)
weights <- c(4, 2, 7, 3, 5, 1, 6, 4)
W <- 15                      # knapsack capacity

# True knapsack objective (linear)
true_f <- function(x) {
  stopifnot(length(x) == p)
  sum(values * x)
}

# Feasibility check (knapsack constraint)
constraint_fn <- function(x) {
  sum(weights * x) <= W
}

# -------------------------
# Initial design
# -------------------------
n_init <- 10
X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)

# Ensure at least one feasible initial point exists:
c_vals <- apply(X, 1, constraint_fn)
if (sum(c_vals) == 0) {
  # create a greedy feasible solution (fill items by value/weight ratio until capacity)
  ratio <- values / weights
  order_idx <- order(-ratio)
  x0 <- integer(p)
  remW <- W
  for (j in order_idx) {
    if (weights[j] <= remW) {
      x0[j] <- 1
      remW <- remW - weights[j]
    }
  }
  X <- rbind(X, x0)
  c_vals <- apply(X, 1, constraint_fn)
  message("No feasible initial points → appended greedy feasible point.")
}

# Evaluate objective at initial points
y <- apply(X, 1, true_f)

# -------------------------
# Fit initial BART models
# -------------------------
# Regression for objective
bart_fit <- wbart(
  x.train = X,
  y.train = y,
  ndpost = 1000, nskip = 200, ntree = 50
)

# Classification for feasibility (0/1 labels)
bart_class <- pbart(
  x.train = X,
  y.train = as.integer(c_vals),
  ndpost = 1000, nskip = 200, ntree = 50
)

# -------------------------
# Acquisition: CEI fitness
# -------------------------
# compute incumbent best among feasible points 
get_f_best <- function(y_vec, c_vec) {
  feas_idx <- which(as.logical(c_vec))
  if (length(feas_idx) == 0) return(NULL)
  max(y_vec[feas_idx])
}

f_best <- get_f_best(y, c_vals)  # may be NULL if no feasible observed

fitness <- function(x_vec) {
  # x_vec may be logical/integer; convert to numeric row matrix
  x_mat <- matrix(as.numeric(x_vec), nrow = 1)
  
  # --- Objective predictive draws (wbart) ---
  pred_draws <- predict(bart_fit, newdata = x_mat)
  # predict(wbart) often returns matrix: ndpost x nnew (or transpose); handle orientation:
  if (is.matrix(pred_draws)) {
    if (nrow(pred_draws) < ncol(pred_draws)) pred_draws <- t(pred_draws)
    pred_vec <- as.numeric(pred_draws)  # flatten draws for the single x
  } else {
    pred_vec <- as.numeric(pred_draws)
  }
  
  mu <- mean(pred_vec)
  sigma <- sd(pred_vec)
  
  # If we have no feasible observation yet, we prioritise feasibility:
  # So we set EI part but don't multiply by p_feas; instead return p_feas alone.
  # Otherwise compute Gaussian EI (safe fallback) — when sigma==0 EI= max(0, mu-f_best)
  if (is.null(f_best)) {
    EI <- NA  # indicate no feasible yet
  } else {
    if (sigma == 0) {
      EI <- max(0, mu - f_best)
    } else {
      z <- (mu - f_best) / sigma
      EI <- (mu - f_best) * pnorm(z) + sigma * dnorm(z)
      if (EI < 0) EI <- 0
    }
  }
  
  # --- Feasibility probability from pbart ---
  class_pred <- predict(bart_class, newdata = x_mat)
  # pbart predict sometimes returns a list with $prob.test.mean; handle common cases:
  if (is.list(class_pred) && !is.null(class_pred$prob.test.mean)) {
    p_feas <- as.numeric(class_pred$prob.test.mean)
  } else if (is.matrix(class_pred) || is.vector(class_pred)) {
    p_feas <- as.numeric(class_pred)
    # if pbart returned latent draws, map via pnorm (rare)
    if (any(p_feas < 0 | p_feas > 1)) p_feas <- pnorm(p_feas)
  } else {
    # fallback (shouldn't happen)
    p_feas <- 0.5
  }
  
  # --- Acquisition value ---
  if (is.null(f_best)) {
    # No feasible observed yet: prioritise feasibility, but still break ties by mu
    # We return p_feas + tiny * mu so solver tries feasible points with higher predicted value
    return(p_feas + 1e-6 * mu)
  } else {
    CEI <- EI * p_feas
    return(CEI)
  }
}

# -------------------------
# Sequential BO loop
# -------------------------
n_iter <- 50
for (iter in 1:n_iter) {
  
  GA_res <- ga(
    type = "binary",
    nBits = p,
    fitness = fitness,
    popSize = 60,
    maxiter = 200,
    run = 40,
    keepBest = TRUE,
    seed = sample.int(.Machine$integer.max, 1)
  )
  
  x_next <- as.numeric(GA_res@solution[1, ])
  cat(sprintf("Iteration %02d — GA proposed: %s\n", iter, paste(x_next, collapse = "")))
  # browser()
  # Evaluate true objective and feasibility
  y_next <- true_f(x_next)
  c_next <- constraint_fn(x_next)
  
  cat("  objective:", y_next, " feasible:", c_next, "\n")
  
  # Append data and update incumbent
  X <- rbind(X, x_next)
  y <- c(y, y_next)
  c_vals <- c(c_vals, as.integer(c_next))
  
  f_best <- get_f_best(y, c_vals)
  
  # Re-fit/update surrogates (naive refit each iteration; can be optimized)
  bart_fit <- wbart(
    x.train = X,
    y.train = y,
    ndpost = 1000, nskip = 200, ntree = 50
  )
  
  bart_class <- pbart(
    x.train = X,
    y.train = as.integer(c_vals),
    ndpost = 1000, nskip = 200, ntree = 50
  )
  
  cat("  Updated dataset size:", nrow(X), "Feasible seen:", sum(c_vals), "Current f_best:", ifelse(is.null(f_best), "NA", f_best), "\n\n")
}

# -------------------------
# Post-run: brute-force / GA benchmark (optional)
# -------------------------
# You can run a final GA to find the true best feasible solution (benchmark)
true_feas_eval <- function(x_vec) {
  x <- as.numeric(x_vec)
  if (constraint_fn(x)) return(true_f(x)) else return(-Inf)
}

GA_final <- ga(
  type = "binary",
  fitness = true_feas_eval,
  nBits = p,
  popSize = 200,
  maxiter = 1000,
  run = 200,
  monitor = FALSE
)

cat("Benchmark GA best feasible solution:", GA_final@solution[1, ], "value:", GA_final@fitnessValue, "\n")
