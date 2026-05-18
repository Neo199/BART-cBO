# Using GA
# p08 - solution
# 110111000110100100000111
# 13,549,094

# -------------------------------------------------------------
# Constrained BART-BO for 0/1 Knapsack - IMPROVED
# -------------------------------------------------------------
options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")
library(dartMachine)
library(GA)

set.seed(1)

# -------------------------
# Problem / knapsack setup
# -------------------------
# p <- 8                       # number of items / binary variables
# 
# # item values (v) and weights (w) - change to your instance
# values <- c(10, 5, 15, 7, 6, 18, 3, 12)
# weights <- c(4, 2, 7, 3, 5, 1, 6, 4)
# W <- 15                      # knapsack capacity

# -------------------------
# Problem / knapsack setup
# -------------------------
p <- 24                       # number of items / binary variables

# item values (v) and weights (w) - change to your instance
weights <- c(382745,
             799601,
             909247,
             729069,
             467902,
             44328,
             34610,
             698150,
             823460,
             903959,
             853665,
             551830,
             610856,
             670702,
             488960,
             951111,
             323046,
             446298,
             931161,
             31385,
             496951,
             264724,
             224916,
             169684)

values <- c( 825594,
             1677009,
             1676628,
             1523970,
             943972,
             97426,
             69666,
             1296457,
             1679693,
             1902996,
             1844992,
             1049289,
             1252836,
             1319836,
             953277,
             2067538,
             675367,
             853655,
             1826027,
             65731,
             901489,
             577243,
             466257,
             369261)
W <- 6404180                      # knapsack capacity

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
# Standardization helpers
# -------------------------
standardize <- function(x) {
  (x - mean(x)) / sd(x)
}

normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# Constraint violation measure (normalized)
constraint_violation <- function(x) {
  total_weight <- sum(weights * x)
  violation <- max(0, total_weight - W)  # 0 if feasible, positive if infeasible
  return(violation)
}

# -------------------------
# Initial design
# -------------------------
n_init <- 10
X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
X_df <- data.frame(X)        #As bartMachine takes a dataframe as input

# # Evaluate objective at initial points
# y <- apply(X, 1, true_f)
# 
# # Check feasibility
# c_vals <- apply(X, 1, constraint_fn)

y <- apply(X, 1, true_f)
c_vals <- apply(X, 1, constraint_fn)

# NEW: Also track constraint violations
c_violations <- apply(X, 1, constraint_violation)

# Store raw values and normalization bounds
y_raw <- y
y_min <- min(y)
y_max <- max(y)
y_normalized <- normalize_01(y)

# Normalize violations (important for scaling penalty)
c_viol_max <- max(c_violations)
if (c_viol_max > 0) {
  c_violations_normalized <- c_violations / c_viol_max
} else {
  c_violations_normalized <- c_violations
}

cat("Initial design: ", sum(c_vals), "feasible out of", length(c_vals), "\n")
cat("Max violation:", c_viol_max, "\n\n")

# cat("Initial design: ", sum(c_vals), "feasible out of", length(c_vals), "\n\n")

# -------------------------
# Create results dataframe
# -------------------------
results_df <- data.frame(X_df, y = y, feasible = c_vals, violation = c_violations)
colnames(results_df)[1:p] <- paste0("x", 1:p)

# -------------------------
# Fit BART models
# -------------------------
# Objective (normalized)
bart_fit <- bartMachine(
  X = X_df,
  y = y_normalized,
  num_trees = 100
)

# Feasibility (binary)
bart_class <- bartMachine(
  X = X_df,
  y = as.factor(c_vals),
  num_trees = 100
)

# NEW: Violation regression model (for better constraint handling)
bart_violation <- bartMachine(
  X = X_df,
  y = c_violations_normalized,
  num_trees = 100
)

# -------------------------
# Acquisition: CEI fitness
# -------------------------
get_f_best <- function(y_vec, c_vec) {
  feas_idx <- which(as.logical(c_vec))
  best_idx <- feas_idx[which.max(y_vec[feas_idx])]
  return(list(value = y_vec[best_idx], idx = best_idx))
}

f_best_info <- get_f_best(y, c_vals)
f_best <- f_best_info$value

# Print initial best
cat("Initial best feasible solution:\n")
cat("  x_best:", paste(X[f_best_info$idx, ], collapse = ""), "\n")
cat("  f_best:", f_best, "\n\n")


# -------------------------
# IMPROVED fitness function with normalized constraints
# -------------------------
penalty_weight <- 2.0  

fitness <- function(x_vec) {
  
  if (is.list(x_vec)) x_vec <- unlist(x_vec)
  x_vec <- as.numeric(x_vec)
  
  x_in <- as.data.frame(matrix(x_vec, nrow = 1))
  colnames(x_in) <- colnames(X_df)
  
  # ============================================================
  # Objective model (normalized scale)
  # ============================================================
  post_draws <- bart_machine_get_posterior(bart_fit, new_data = x_in)
  pred_vec <- as.numeric(post_draws$y_hat_posterior_samples)
  
  mu <- mean(pred_vec, na.rm = TRUE)
  sigma <- sd(pred_vec, na.rm = TRUE)
  
  # Handle NA/NaN cases
  if (is.na(mu) || is.nan(mu)) mu <- 0
  if (is.na(sigma) || is.nan(sigma) || sigma < 1e-10) sigma <- 1e-6
  
  # Convert current best to normalized scale
  if (!is.null(f_best)) {
    f_best_norm <- (f_best - y_min) / (y_max - y_min)
  } else {
    f_best_norm <- NULL
  }
  
  # Expected Improvement (normalized)
  if (is.null(f_best_norm)) {
    EI <- mu
  } else {
    if (sigma < 1e-6) {
      EI <- max(0, mu - f_best_norm)
    } else {
      z  <- (mu - f_best_norm) / sigma
      EI <- (mu - f_best_norm) * pnorm(z) + sigma * dnorm(z)
      EI <- max(EI, 0)
    }
  }
  
  # ============================================================
  # Constraint models
  # ============================================================
  # Probability of feasibility
  p_feas <- as.numeric(predict(bart_class, new_data = x_in, type = "prob"))
  p_feas <- max(min(p_feas, 1), 0)
  
  # Predicted violation (normalized)
  post_viol <- bart_machine_get_posterior(bart_violation, new_data = x_in)
  pred_viol_vec <- as.numeric(post_viol$y_hat_posterior_samples)
  pred_viol <- mean(pred_viol_vec)
  pred_viol <- max(0, pred_viol)  # Clamp to non-negative
  
  # ============================================================
  # Acquisition with normalized scales
  # ============================================================
  if (is.null(f_best_norm)) {
    # No feasible point yet → prioritize feasibility
    return(10 * p_feas - penalty_weight * pred_viol + 0.01 * mu)
  } else {
    # Constrained EI with violation penalty
    CEI <- EI * p_feas
    violation_penalty <- penalty_weight * pred_viol
    
    return(CEI - violation_penalty)
  }
}


# -------------------------
# BO loop with constraint normalization updates
# -------------------------
n_iter <- 150
infeasible_streak <- 0

for (iter in 1:n_iter) {
  
  # Adaptive penalty
  if (infeasible_streak > 3) {
    penalty_weight <- min(penalty_weight * 1.5, 20)
    cat("  [Adjusting penalty weight to", round(penalty_weight, 2), "]\n")
  }
  # browser()
  GA_res <- ga(
    type = "binary",
    nBits = p,
    fitness = fitness,
    popSize = 60,
    maxiter = 20,
    run = 40,
    keepBest = TRUE,
    seed = sample.int(.Machine$integer.max, 1)
  )
  
  x_next <- as.numeric(GA_res@solution[1, ])
  cat(sprintf("Iteration %02d — GA proposed: %s\n", iter, paste(x_next, collapse = "")))
  
  # Evaluate
  y_next <- true_f(x_next)
  c_next <- constraint_fn(x_next)
  c_viol_next <- constraint_violation(x_next)
  
  cat("  objective:", y_next, " feasible:", c_next, " violation:", c_viol_next, "\n")
  
  # Update streak
  if (!c_next) {
    infeasible_streak <- infeasible_streak + 1
  } else {
    infeasible_streak <- 0
    penalty_weight <- max(penalty_weight * 0.9, 1.0)
  }
  
  # Append data
  X <- rbind(X, x_next)
  X_df <- data.frame(X)
  y <- c(y, y_next)
  c_vals <- c(c_vals, as.integer(c_next))
  c_violations <- c(c_violations, c_viol_next)
  
  # Update normalization bounds
  y_min <- min(y)
  y_max <- max(y)
  y_normalized <- normalize_01(y)
  
  c_viol_max <- max(c_violations)
  if (c_viol_max > 0) {
    c_violations_normalized <- c_violations / c_viol_max
  } else {
    c_violations_normalized <- c_violations
  }
  
  # Update results
  new_row <- data.frame(t(x_next), y = y_next, feasible = as.integer(c_next), 
                        violation = c_viol_next)
  colnames(new_row)[1:p] <- paste0("x", 1:p)
  results_df <- rbind(results_df, new_row)
  
  # Update best
  f_best_info <- get_f_best(y, c_vals)
  f_best <- f_best_info$value
  
  # Refit all models with normalized data
  bart_fit <- bartMachine(
    X = X_df,
    y = y_normalized,
    num_trees = 100
  )
  
  bart_class <- bartMachine(
    X = X_df,
    y = as.factor(c_vals),
    num_trees = 100
  )
  
  bart_violation <- bartMachine(
    X = X_df,
    y = c_violations_normalized,
    num_trees = 100
  )
  
  # Print status
  if (!is.null(f_best)) {
    cat("  x_best:", paste(X[f_best_info$idx, ], collapse = ""), "\n")
  }
  
  feas_rate <- sum(c_vals) / length(c_vals)
  cat("  Dataset size:", nrow(X), "| Feasible:", sum(c_vals), 
      sprintf("(%.1f%%)", 100*feas_rate), "| f_best:", 
      ifelse(is.null(f_best), "NA", f_best), "\n\n")
}


# -------------------------
# Print final results
# -------------------------
cat("\n=== FINAL RESULTS ===\n")
cat("Total evaluations:", nrow(results_df), "\n")
cat("Feasible solutions found:", sum(results_df$feasible), 
    sprintf("(%.1f%%)\n", 100*sum(results_df$feasible)/nrow(results_df)))

if (!is.null(f_best)) {
  cat("\nBest feasible solution found by BART-BO:\n")
  cat("  x_best:", paste(X[f_best_info$idx, ], collapse = ""), "\n")
  cat("  f_best:", f_best, "\n")
  cat("  Weight:", sum(weights * X[f_best_info$idx, ]), "/", W, "\n\n")
}

# -------------------------
# Post-run: GA benchmark
# -------------------------
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

cat("Benchmark GA best feasible solution:\n")
cat("  x:", paste(GA_final@solution[1, ], collapse = ""), "\n")
cat("  value:", GA_final@fitnessValue, "\n")
cat("  Weight:", sum(weights * GA_final@solution[1, ]), "/", W, "\n")

# Performance comparison
if (!is.null(f_best)) {
  gap <- (GA_final@fitnessValue - f_best) / GA_final@fitnessValue * 100
  cat(sprintf("\nOptimality gap: %.2f%%\n", gap))
}