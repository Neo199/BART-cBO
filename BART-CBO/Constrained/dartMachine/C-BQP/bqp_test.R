# -------------------------------------------------------------
# Constrained BART-BO for Binary Quadratic Programming
# Constraint: sum(x) <= k_budget  (cardinality constraint)
#
# Extends the unconstrained BART-BO code by adding:
#   - violation surrogate  (predicts constraint excess)
#   - feasibility classifier  (P(feasible) estimate)
#   - constrained acquisition: EI * p_feas - penalty_w * pred_viol
#   - adaptive penalty weight (ramps up on infeasible streaks)
#   - feasibility-aware best tracking (only feasible incumbents)
#   - GLPK exact solver as an additional baseline
# -------------------------------------------------------------
options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

library(bartMachine)
library(GA)
library(ompr)
library(ompr.roi)
library(ROI)
library(ROI.plugin.glpk)
library(dplyr)

set.seed(1)

# -------------------------
# Problem setup
# -------------------------
p        <- 100      # number of binary variables
alpha    <- 1       # correlation length for Q matrix
k_budget <- 75      # CONSTRAINT: at most k_budget variables can be 1
# (cardinality constraint; set to p for unconstrained)

# Generate Q matrix (same as unconstrained version)
quad_mat <- function(n_vars, alpha) {
  K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
  decay <- matrix(0, n_vars, n_vars)
  for (i in 1:n_vars)
    for (j in 1:n_vars)
      decay[i, j] <- K(i, j)
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  Q * decay
}

Q <- quad_mat(n_vars = p, alpha = alpha)

# -------------------------
# Evaluation counter
# -------------------------
eval_counter <- 0

# -------------------------
# Objective and constraint helpers
# -------------------------

# True quadratic objective (MINIMIZATION)
true_f <- function(x) {
  eval_counter <<- eval_counter + 1
  x <- as.numeric(x)
  as.numeric(t(x) %*% Q %*% x)
}

true_f_batch <- function(X_mat) apply(X_mat, 1, true_f)

# Feasibility: 1 if sum(x) <= k_budget, else 0
bqp_feasible <- function(x) as.integer(sum(x) <= k_budget)

# Violation: how many variables exceed the budget (0 if feasible)
bqp_violation <- function(x) max(0L, as.integer(sum(x)) - k_budget)

# -------------------------
# Helper: best feasible incumbent
# Returns list(value, idx) or list(value=NULL, idx=NULL) if none feasible
# -------------------------
get_f_best_feasible <- function(y, feas) {
  idx <- which(as.logical(feas))
  if (!length(idx)) return(list(value = NULL, idx = NULL))
  bi <- idx[which.min(y[idx])]     # min because we MINIMISE
  list(value = y[bi], idx = bi)
}

# -------------------------
# Normalisation helpers
# -------------------------
normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# -------------------------
# Initial design
# -------------------------
n_init        <- 10
penalty_w     <- 5      # initial violation-penalty weight
eval_counter  <- 0

X      <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
X_df   <- as.data.frame(X)
colnames(X_df) <- paste0("x", 1:p)

y        <- true_f_batch(X)
feas_bin <- apply(X, 1, bqp_feasible)   # NEW
viols    <- apply(X, 1, bqp_violation)  # NEW

bo_evals_init <- eval_counter

cat("=== INITIALIZATION ===\n")
cat("Initial design evaluated at", n_init, "points\n")
cat("Feasible points in init:", sum(feas_bin), "/", n_init, "\n")
cat("True objective evaluations:", bo_evals_init, "\n\n")

# -------------------------
# Normalise for surrogates
# -------------------------
y_norm    <- normalize_01(y)
viol_norm <- normalize_01(viols)   # NEW

# -------------------------
# Fit surrogate models
# -------------------------
fit_surrogates <- function(X_df, y_norm, feas_bin, viol_norm,
                           num_trees = 100) {
  
  obj_model  <- bartMachine(X = X_df, y = y_norm,   num_trees = num_trees,
                            verbose = FALSE)
  
  # Violation surrogate (regression)
  viol_model <- bartMachine(X = X_df, y = viol_norm, num_trees = num_trees,
                            verbose = FALSE)
  
  # Feasibility classifier (only when both classes are observed)
  feas_model <- if (length(unique(feas_bin)) == 2) {
    bartMachine(X   = X_df,
                y   = factor(feas_bin, levels = c("0", "1")),
                num_trees = num_trees,
                verbose   = FALSE)
  } else {
    NULL   # will default to p_feas = 0.5
  }
  
  list(obj = obj_model, viol = viol_model, feas = feas_model)
}

cat("Fitting initial surrogates...\n")
models <- fit_surrogates(X_df, y_norm, feas_bin, viol_norm)
cat("Surrogates fitted OK\n\n")

# -------------------------
# Track best FEASIBLE incumbent
# -------------------------
fb_info        <- get_f_best_feasible(y, feas_bin)
f_best         <- fb_info$value    # NULL if no feasible point yet
best_idx       <- fb_info$idx

# Keep a copy of the best x vector (NEW — mirrors MCLP code)
best_x_so_far  <- if (!is.null(best_idx)) X[best_idx, ] else NULL

# Normalised incumbent (for EI computation)
f_best_norm <- if (!is.null(f_best)) {
  (f_best - min(y)) / max(max(y) - min(y), 1e-8)
} else NULL

# BO loop state
current_pw        <- penalty_w
infeasible_streak <- 0
true_evals        <- n_init

cat("=== INITIALIZATION SUMMARY ===\n")
if (!is.null(f_best)) {
  cat("Best feasible objective:", f_best, "\n")
  cat("Best feasible x:", paste(X[best_idx, ], collapse = ""), "\n\n")
} else {
  cat("No feasible point in initial design — acquisition uses exploration mode\n\n")
}

# -------------------------
# History data frame
# -------------------------
history <- data.frame(
  iteration       = rep(0, n_init),
  objective       = y,
  feasible        = feas_bin,
  violation       = viols,
  best_obj_so_far = ifelse(is.null(f_best), NA, f_best),
  true_evals      = seq_len(n_init)
)

# -------------------------
# Constrained acquisition function (NEW — from MCLP approach)
# EI * P(feasible) - penalty_w * predicted_violation
# Falls back to exploration mode if no feasible incumbent yet
# -------------------------
acq_fn <- function(x_vec) {
  
  if (is.list(x_vec)) x_vec <- unlist(x_vec)
  x_vec <- as.numeric(x_vec)
  
  x_in <- as.data.frame(matrix(x_vec, nrow = 1))
  colnames(x_in) <- colnames(X_df)
  
  tryCatch({
    # --- Objective surrogate ---
    post_obj  <- bart_machine_get_posterior(models$obj, new_data = x_in)
    obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
    mu        <- mean(obj_draws, na.rm = TRUE)
    sigma     <- sd(obj_draws,   na.rm = TRUE)
    if (is.na(mu))                     mu    <- 0
    if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
    
    # --- Feasibility probability ---
    # NOTE: for minimisation EI we want to MAXIMISE EI,
    # so we pass p_feas as a weight and negate violation
    p_feas <- if (is.null(models$feas)) {
      0.5
    } else {
      pv <- predict(models$feas, new_data = x_in, type = "prob")
      as.numeric(pv)
    }
    p_feas <- max(min(p_feas, 1), 0)
    if (is.na(p_feas)) p_feas <- 0.5
    
    # --- Violation surrogate ---
    post_viol  <- bart_machine_get_posterior(models$viol, new_data = x_in)
    viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
    pred_viol  <- max(0, mean(viol_draws, na.rm = TRUE))
    if (is.na(pred_viol)) pred_viol <- 0
    
    # --- Acquisition value ---
    # Because we MINIMISE, EI is improvement below incumbent:
    #   z = (f_best_norm - mu) / sigma  (positive when mu < f_best)
    #   EI = (f_best_norm - mu)*Phi(z) + sigma*phi(z)
    # When no feasible incumbent: use exploration (p_feas reward only)
    if (is.null(f_best_norm)) {
      # Exploration mode: steer toward feasibility + low predicted objective
      acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * (1 - mu)
    } else {
      z  <- (f_best_norm - mu) / sigma
      EI <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
      acq <- EI * p_feas - current_pw * pred_viol
    }
    
    if (is.na(acq) || is.nan(acq)) -1e6 else acq
    
  }, error = function(e) -1e6)
}

# -------------------------
# BO loop
# -------------------------
n_iter <- 250

bo_evals_before_loop <- eval_counter

cat("=== STARTING CONSTRAINED BO LOOP ===\n")
cat(sprintf("%-6s %-12s %-5s %-8s %-12s %-8s\n",
            "iter", "objective", "feas", "viol", "best_obj", "penalty"))
cat(strrep("-", 60), "\n")

for (iter in seq_len(n_iter)) {
  
  # Adaptive penalty: ramp up if we keep proposing infeasible points
  if (infeasible_streak > 3) {
    current_pw <- min(current_pw * 1.5, 50)
  }
  
  # Run GA to maximise constrained acquisition
  GA_res <- ga(
    type     = "binary",
    nBits    = p,
    fitness  = acq_fn,
    popSize  = 60,
    maxiter  = 20,
    run      = 40,
    keepBest = TRUE,
    monitor  = FALSE,
    seed     = iter
  )
  
  x_next <- as.numeric(GA_res@solution[1, ])
  
  # Evaluate true objective and constraint
  y_next    <- true_f(x_next)     # increments eval_counter
  feas_next <- bqp_feasible(x_next)
  viol_next <- bqp_violation(x_next)
  true_evals <- true_evals + 1
  
  # Update adaptive penalty state
  if (feas_next == 0) {
    infeasible_streak <- infeasible_streak + 1
  } else {
    infeasible_streak <- 0
    current_pw        <- max(current_pw * 0.9, penalty_w)
  }
  
  # Append data
  X        <- rbind(X, x_next)
  X_df     <- as.data.frame(X)
  colnames(X_df) <- paste0("x", 1:p)
  y        <- c(y, y_next)
  feas_bin <- c(feas_bin, feas_next)
  viols    <- c(viols, viol_next)
  
  # Update best feasible incumbent
  fb_info     <- get_f_best_feasible(y, feas_bin)
  f_best      <- fb_info$value
  best_idx    <- fb_info$idx
  if (!is.null(best_idx)) best_x_so_far <- X[best_idx, ]
  
  # Recompute normalised incumbent
  y_range     <- max(max(y) - min(y), 1e-8)
  f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
  
  # Update history
  history <- rbind(history, data.frame(
    iteration       = iter,
    objective       = y_next,
    feasible        = feas_next,
    violation       = viol_next,
    best_obj_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals      = true_evals
  ))
  
  cat(sprintf("%-6d %-12.4f %-5d %-8d %-12s %-8.2f\n",
              iter, y_next, feas_next, viol_next,
              ifelse(is.null(f_best), "NA", sprintf("%.4f", f_best)),
              current_pw))
  
  # Refit all three surrogates
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_surrogates(X_df, y_norm, feas_bin, viol_norm)
}

bo_total_evals <- eval_counter

cat("\n=== BART-BO FINAL RESULTS ===\n")
cat("Total true objective evaluations:", bo_total_evals, "\n")
cat("  - Initialization:", bo_evals_init, "\n")
cat("  - BO iterations:", bo_total_evals - bo_evals_init, "\n")
cat("Feasible evaluations:", sum(history$feasible, na.rm = TRUE),
    "/", nrow(history), "\n")
cat("Feasibility rate:", round(mean(history$feasible, na.rm = TRUE), 3), "\n")

if (!is.null(f_best)) {
  cat("\nBest feasible solution:\n")
  cat("  x_best:", paste(best_x_so_far, collapse = ""), "\n")
  cat("  sum(x_best):", sum(best_x_so_far), "  (budget:", k_budget, ")\n")
  cat("  **Minimized objective:**", f_best, "\n")
} else {
  cat("\nNo feasible solution found by BART-BO!\n")
}

# -------------------------
# GLPK exact solver baseline (NEW — from MCLP approach)
# Solves:  min  x^T Q x
#          s.t. sum(x) <= k_budget
#               x in {0,1}^p
# NOTE: x^T Q x is quadratic; GLPK only handles MIP.
# We linearise via the substitution y_ij = x_i * x_j.
# For small p this is tractable.
# -------------------------
cat("\n=== GLPK EXACT SOLVER ===\n")
cat("(Linearised BQP via y_ij = x_i * x_j substitution)\n\n")

glpk_t0 <- proc.time()["elapsed"]

# Collect non-zero (i,j) pairs from Q (upper triangle suffices since Q not symmetric)
ij_pairs <- which(Q != 0, arr.ind = TRUE)
n_pairs  <- nrow(ij_pairs)

lp_model <- MIPModel() %>%
  add_variable(x[i],    i = 1:p,       type = "binary") %>%
  add_variable(y[k],    k = 1:n_pairs, type = "binary") %>%
  
  # Linearisation constraints: y_k = x[i_k] * x[j_k]
  # y <= x_i, y <= x_j, y >= x_i + x_j - 1
  add_constraint(y[k] <= x[ij_pairs[k, 1]],                        k = 1:n_pairs) %>%
  add_constraint(y[k] <= x[ij_pairs[k, 2]],                        k = 1:n_pairs) %>%
  add_constraint(y[k] >= x[ij_pairs[k, 1]] + x[ij_pairs[k, 2]] - 1, k = 1:n_pairs) %>%
  
  # Cardinality constraint
  add_constraint(sum_expr(x[i], i = 1:p) <= k_budget) %>%
  
  # Objective: sum_{k} Q[i_k, j_k] * y_k
  set_objective(
    sum_expr(Q[ij_pairs[k, 1], ij_pairs[k, 2]] * y[k], k = 1:n_pairs),
    sense = "min"
  )

glpk_res  <- solve_model(lp_model, with_ROI(solver = "glpk", verbose = FALSE))
glpk_time <- proc.time()["elapsed"] - glpk_t0
glpk_x    <- as.integer(get_solution(glpk_res, x[i])$value)
glpk_obj  <- glpk_res$objective_value

cat(sprintf("GLPK optimal objective: %.6f  |  Time: %.2f s\n", glpk_obj, glpk_time))
cat(sprintf("GLPK sum(x): %d  (budget: %d)\n", sum(glpk_x), k_budget))
cat(sprintf("GLPK x: %s\n\n", paste(glpk_x, collapse = "")))

# -------------------------
# GA baseline (penalty-based, matching MCLP approach)
# -------------------------
cat("=== RUNNING BENCHMARK GA (PENALTY-BASED) ===\n")

eval_counter <- 0    # reset for fair comparison

ga_fitness_constrained <- function(x_vec) {
  x    <- as.numeric(x_vec)
  obj  <- true_f(x)                    # increments counter
  viol <- bqp_violation(x)
  -(obj + 1e3 * viol)                  # negate: GA maximises; penalise violation
}

GA_final   <- ga(
  type     = "binary",
  fitness  = ga_fitness_constrained,
  nBits    = p,
  monitor  = FALSE
)
ga_total_evals <- eval_counter
ga_x           <- as.integer(GA_final@solution[1, ])
ga_obj         <- true_f(ga_x) - 1   # one extra eval; undo negate
ga_obj         <- true_f_batch(matrix(ga_x, nrow = 1))
# Re-evaluate cleanly (doesn't count toward GA budget above)
ga_obj         <- as.numeric(t(ga_x) %*% Q %*% ga_x)
ga_feas        <- bqp_feasible(ga_x)
ga_viol        <- bqp_violation(ga_x)

cat(sprintf("GA objective: %.6f  |  Feasible: %d  |  Violation: %d\n",
            ga_obj, ga_feas, ga_viol))
cat(sprintf("GA sum(x): %d  (budget: %d)\n", sum(ga_x), k_budget))
cat(sprintf("GA x: %s\n", paste(ga_x, collapse = "")))
cat(sprintf("Total GA true-objective evaluations: %d\n\n", ga_total_evals))

# -------------------------
# Comparison summary
# -------------------------
cat("=== EVALUATION EFFICIENCY COMPARISON ===\n\n")

# BART-BO result
bart_obj    <- ifelse(is.null(f_best), NA, f_best)
bart_feas   <- ifelse(is.null(f_best), 0, 1)

# Gap from GLPK
gap_bart <- if (!is.na(bart_obj)) bart_obj - glpk_obj else NA
gap_ga   <- ga_obj - glpk_obj

cat(sprintf("%-12s  %10s  %8s  %8s  %8s  %8s\n",
            "Method", "Objective", "Feasible", "sum(x)", "Gap(GLPK)", "Evals"))
cat(strrep("-", 66), "\n")
cat(sprintf("%-12s  %10.4f  %8s  %8d  %8.4f  %8d\n",
            "GLPK",     glpk_obj, "YES", sum(glpk_x), 0, NA))
cat(sprintf("%-12s  %10s  %8s  %8d  %8s  %8d\n",
            "BART-BO",
            ifelse(is.na(bart_obj), "NA", sprintf("%.4f", bart_obj)),
            ifelse(bart_feas, "YES", "NO"),
            ifelse(!is.null(best_x_so_far), sum(best_x_so_far), NA),
            ifelse(is.na(gap_bart), "NA", sprintf("%.4f", gap_bart)),
            bo_total_evals))
cat(sprintf("%-12s  %10.4f  %8s  %8d  %8.4f  %8d\n",
            "GA",       ga_obj, ifelse(ga_feas, "YES", "NO"),
            sum(ga_x), gap_ga, ga_total_evals))

cat(sprintf("\nEvaluation efficiency: BART-BO used %.1f%% of GA's evaluations\n",
            100 * bo_total_evals / ga_total_evals))

if (!is.na(gap_bart) && bart_feas) {
  pct_gap <- 100 * abs(gap_bart) / abs(glpk_obj + 1e-10)
  cat(sprintf("BART-BO optimality gap from GLPK: %.2f%%\n", pct_gap))
}

# -------------------------
# Save outputs
# -------------------------
write.csv(history, "constrained_bqp_history.csv", row.names = FALSE)
cat("\nHistory saved to constrained_bqp_history.csv\n")

summary_table <- data.frame(
  Method      = c("GLPK", "BART-BO", "GA"),
  True_Evals  = c(NA, bo_total_evals, ga_total_evals),
  Objective   = c(glpk_obj, bart_obj, ga_obj),
  Feasible    = c(1, bart_feas, ga_feas),
  Budget_used = c(sum(glpk_x),
                  ifelse(!is.null(best_x_so_far), sum(best_x_so_far), NA),
                  sum(ga_x)),
  Gap_GLPK    = c(0, gap_bart, gap_ga)
)
print(summary_table)