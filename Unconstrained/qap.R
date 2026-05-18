# -------------------------------------------------------------
# Unconstrained BART-BO for Binary Quadratic Programming
# Matching BOCS implementation with evaluation counting
# -------------------------------------------------------------
options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")
library(dartMachine)
library(GA)

set.seed(1)

# -------------------------
# Problem setup (matching BOCS)
# -------------------------
p <- 30                      # n_vars (start with BOCS default)
alpha <- 1                   # correlation length (BOCS uses alpha, not corr_len)
lambda <- 1e-4               # regularization parameter (try BOCS values: 0, 1e-4, 1e-2, 1)

# Generate Q matrix using BOCS approach
quad_mat <- function(n_vars, alpha) {
  # Decay function (note: BOCS uses -1 multiplier)
  K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
  
  # Compute the decay matrix
  decay <- matrix(0, n_vars, n_vars)
  for (i in 1:n_vars) {
    for (j in 1:n_vars) {
      decay[i, j] <- K(i, j)
    }
  }
  
  # Generate random quadratic model
  # and apply exponential decay to Q
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  Qa <- Q * decay
  
  return(Qa)
}

Q <- quad_mat(n_vars = p, alpha = alpha)

# -------------------------
# EVALUATION COUNTER - Global variable
# -------------------------
eval_counter <- 0

# CORRECTED: Quadratic objective matching BOCS with counter
true_f <- function(x) {
  # Increment counter
  eval_counter <<- eval_counter + 1
  
  x <- as.numeric(x)
  
  # Compute x^T * Q * x (scalar for single solution)
  quad_term <- as.numeric(t(x) %*% Q %*% x)
  
  # Regularization term: sum(x) matching BOCS reg_term
  # reg_term <- lambda * sum(x)
  
  # Total objective (MINIMIZATION)
  return(quad_term) # + reg_term)
}

# For batch evaluation (like BOCS model function)
true_f_batch <- function(X_mat) {
  # X_mat: n x p matrix
  # Returns: n-vector of objectives
  apply(X_mat, 1, true_f)
}

# -------------------------
# Standardization helpers
# -------------------------
normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# -------------------------
# Initial design (matching BOCS: n_init = 20)
# -------------------------
n_init <- 20

# Reset counter before starting
eval_counter <- 0

# BOCS uses: sample_models(n_init, n_vars)
# This likely generates uniform random binary samples
X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
X_df <- data.frame(X)

# Evaluate objective at initial points
y <- true_f_batch(X)
bo_evals_init <- eval_counter

cat("=== INITIALIZATION ===\n")
cat("Initial design evaluated at", n_init, "points\n")
cat("True objective evaluations:", bo_evals_init, "\n")
cat("Initial best (minimum):", min(y), "\n")
cat("Initial worst (maximum):", max(y), "\n\n")

# Store raw values and normalization bounds
y_raw <- y
y_min <- min(y)
y_max <- max(y)
y_normalized <- normalize_01(y)

# -------------------------
# Create results dataframe
# -------------------------
results_df <- data.frame(X_df, y = y)
colnames(results_df)[1:p] <- paste0("x", 1:p)

# -------------------------
# Fit BART model
# -------------------------
bart_fit <- bartMachine(
  X = X_df,
  y = y_normalized,
  num_trees = 100
)

# -------------------------
# Track best (minimum) value for MINIMIZATION
# -------------------------
f_best <- min(y)
best_idx <- which.min(y)

cat("Initial best solution:\n")
cat("  x_best:", paste(X[best_idx, ], collapse = ""), "\n")
cat("  Objective:", f_best, "\n\n")

# -------------------------
# Fitness function (Expected Improvement for MINIMIZATION)
# -------------------------
fitness <- function(x_vec) {
  
  if (is.list(x_vec)) x_vec <- unlist(x_vec)
  x_vec <- as.numeric(x_vec)
  
  x_in <- as.data.frame(matrix(x_vec, nrow = 1))
  colnames(x_in) <- colnames(X_df)
  
  # Get posterior predictions (normalized scale)
  post_draws <- bart_machine_get_posterior(bart_fit, new_data = x_in)
  pred_vec <- as.numeric(post_draws$y_hat_posterior_samples)
  
  mu <- mean(pred_vec, na.rm = TRUE)
  sigma <- sd(pred_vec, na.rm = TRUE)
  
  # Handle edge cases
  if (is.na(mu) || is.nan(mu)) mu <- 0
  if (is.na(sigma) || is.nan(sigma) || sigma < 1e-10) sigma <- 1e-6
  
  # Convert current best to normalized scale
  f_best_norm <- (f_best - y_min) / (y_max - y_min)
  
  # Expected Improvement for MINIMIZATION
  if (sigma < 1e-6) {
    EI <- max(0, f_best_norm - mu)
  } else {
    z  <- (f_best_norm - mu) / sigma
    EI <- (f_best_norm - mu) * pnorm(z) + sigma * dnorm(z)
    EI <- max(EI, 0)
  }
  
  return(EI)
}

# -------------------------
# BO loop (BOCS uses evalBudget = 120 total, so 100 iterations after init)
# -------------------------
n_iter <- 100  # evalBudget - n_init
bo_evals_before_loop <- eval_counter

cat("=== STARTING BO LOOP ===\n\n")

for (iter in 1:n_iter) {
  
  evals_before_iter <- eval_counter
  
  # Run GA to optimize acquisition function
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
  cat(sprintf("Iteration %03d — GA proposed: %s\n", iter, paste(x_next, collapse = "")))
  
  # Evaluate true objective (this increments counter)
  y_next <- true_f(x_next)
  
  evals_this_iter <- eval_counter - evals_before_iter
  
  cat("  Objective:", y_next, "\n")
  cat("  True evals this iter:", evals_this_iter, "\n")
  
  # Append data
  X <- rbind(X, x_next)
  X_df <- data.frame(X)
  y <- c(y, y_next)
  
  # Update normalization bounds
  y_min <- min(y)
  y_max <- max(y)
  y_normalized <- normalize_01(y)
  
  # Update results
  new_row <- data.frame(t(x_next), y = y_next)
  colnames(new_row)[1:p] <- paste0("x", 1:p)
  results_df <- rbind(results_df, new_row)
  
  # Update best (minimum)
  f_best <- min(y)
  best_idx <- which.min(y)
  
  # Refit BART model with normalized data
  bart_fit <- bartMachine(
    X = X_df,
    y = y_normalized,
    num_trees = 100
  )
  
  # Print status
  cat("  x_best:", paste(X[best_idx, ], collapse = ""), "\n")
  cat("  Dataset size:", nrow(X), "| Best objective (min):", f_best, "\n")
  cat("  Total true evals so far:", eval_counter, "\n\n")
}

bo_total_evals <- eval_counter

# -------------------------
# Print BART-BO final results
# -------------------------
cat("\n=== BART-BO FINAL RESULTS ===\n")
cat("Total evaluations:", nrow(results_df), "\n")
cat("Total TRUE objective evaluations:", bo_total_evals, "\n")
cat("  - Initialization:", bo_evals_init, "\n")
cat("  - BO iterations:", bo_total_evals - bo_evals_init, "\n")

cat("\nBest solution found by BART-BO:\n")
cat("  x_best:", paste(X[best_idx, ], collapse = ""), "\n")
cat("  **Minimized objective:**", f_best, "\n\n")

# -------------------------
# Post-run: GA benchmark with counter
# -------------------------
cat("=== RUNNING BENCHMARK GA ===\n")

# Reset counter for GA
eval_counter <- 0

ga_fitness <- function(x) {
  -true_f(x)  # Negate for GA (this will increment counter)
}

GA_final <- ga(
  type = "binary",
  fitness = ga_fitness,
  nBits = p,
  popSize = 200,
  maxiter = 100,
  run = 200,
  monitor = FALSE
)

ga_total_evals <- eval_counter
ga_best_obj <- -GA_final@fitnessValue

cat("Benchmark GA best solution:\n")
cat("  x:", paste(GA_final@solution[1, ], collapse = ""), "\n")
cat("  **Minimized objective:**", ga_best_obj, "\n")
cat("  **Total TRUE objective evaluations:**", ga_total_evals, "\n\n")

# -------------------------
# COMPARISON SUMMARY
# -------------------------
cat("=== EVALUATION EFFICIENCY COMPARISON ===\n")
cat(sprintf("BART-BO: %d true objective evaluations → objective = %.6f\n", 
            bo_total_evals, f_best))
cat(sprintf("Pure GA: %d true objective evaluations → objective = %.6f\n", 
            ga_total_evals, ga_best_obj))
cat(sprintf("\nEvaluation efficiency: BART-BO used %.1f%% of GA's evaluations\n",
            100 * bo_total_evals / ga_total_evals))

# Performance comparison
gap <- abs((f_best - ga_best_obj) / ga_best_obj * 100)
cat(sprintf("Optimality gap: %.2f%%\n", gap))

if (f_best < ga_best_obj) {
  improvement <- (ga_best_obj - f_best) / ga_best_obj * 100
  cat(sprintf("\n✓ BART-BO found a BETTER solution (%.2f%% improvement)\n", improvement))
  cat(sprintf("  using only %.1f%% of the evaluations!\n", 100 * bo_total_evals / ga_total_evals))
} else if (abs(gap) < 1) {
  cat("\n≈ Both methods found similar solutions\n")
  cat(sprintf("  but BART-BO used %.1f%% fewer evaluations!\n", 
              100 * (1 - bo_total_evals / ga_total_evals)))
} else {
  cat("\nGA found a slightly better solution,\n")
  cat(sprintf("but required %.1fx more evaluations\n", ga_total_evals / bo_total_evals))
}

# -------------------------
# Save detailed results
# -------------------------
cat("\n=== SUMMARY TABLE ===\n")
summary_table <- data.frame(
  Method = c("BART-BO", "Pure GA"),
  True_Evals = c(bo_total_evals, ga_total_evals),
  Best_Objective = c(f_best, ga_best_obj),
  Best_Solution = c(paste(X[best_idx, ], collapse = ""),
                    paste(GA_final@solution[1, ], collapse = ""))
)
print(summary_table)

