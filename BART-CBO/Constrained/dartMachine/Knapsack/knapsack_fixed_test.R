# =============================================================================
# Single-instance test: BART-BO vs benchmark GA
# Penalty weight = 2, seed = 1001, n_iter = 250
# Run this to sanity-check before launching the full experiment.
# =============================================================================

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")
library(dartMachine)
library(GA)

# =============================================================================
# PROBLEM SETUP
# =============================================================================
p <- 24

weights <- c(382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
             823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
             323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684)

values <- c(825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
            1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538,
            675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261)

W <- 6404180

MAX_POSSIBLE_VALUE     <- sum(values)
MAX_POSSIBLE_VIOLATION <- sum(weights) - W

normalize_value     <- function(y) y / MAX_POSSIBLE_VALUE
normalize_violation <- function(v) v / MAX_POSSIBLE_VIOLATION

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

true_f <- function(x) {
  stopifnot(length(x) == p)
  sum(values * x)
}

constraint_fn <- function(x) {
  sum(weights * x) <= W
}

constraint_violation <- function(x) {
  max(0, sum(weights * x) - W)
}

get_f_best <- function(y_vec, c_vec) {
  feas_idx <- which(as.logical(c_vec))
  if (length(feas_idx) == 0) return(list(value = NULL, idx = NULL))
  best_idx <- feas_idx[which.max(y_vec[feas_idx])]
  list(value = y_vec[best_idx], idx = best_idx)
}

# =============================================================================
# BART-BO
# =============================================================================

run_bart_bo <- function(penalty_weight, seed_val, n_iter = 250, verbose = TRUE) {
  
  set.seed(seed_val)
  
  n_init <- 10
  X      <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
  X_df   <- data.frame(X)
  colnames(X_df) <- paste0("x", 1:p)
  
  y            <- apply(X, 1, true_f)
  c_vals       <- apply(X, 1, constraint_fn)
  c_violations <- apply(X, 1, constraint_violation)
  
  y_normalized            <- normalize_value(y)
  c_violations_normalized <- normalize_violation(c_violations)
  
  cat("Initial design:", sum(c_vals), "feasible out of", n_init, "\n\n")
  
  results_df <- data.frame(X_df, y = y, feasible = c_vals, violation = c_violations)
  colnames(results_df)[1:p] <- paste0("x", 1:p)
  
  bart_fit       <- bartMachine(X = X_df, y = y_normalized,            num_trees = 100, verbose = FALSE)
  bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),       num_trees = 100, verbose = FALSE)
  bart_violation <- bartMachine(X = X_df, y = c_violations_normalized, num_trees = 100, verbose = FALSE)
  
  f_best_info <- get_f_best(y, c_vals)
  f_best      <- f_best_info$value
  
  fitness <- function(x_vec) {
    if (is.list(x_vec)) x_vec <- unlist(x_vec)
    x_vec <- as.numeric(x_vec)
    x_in  <- as.data.frame(matrix(x_vec, nrow = 1))
    colnames(x_in) <- colnames(X_df)
    
    tryCatch({
      post_draws <- bart_machine_get_posterior(bart_fit, new_data = x_in)
      pred_vec   <- as.numeric(post_draws$y_hat_posterior_samples)
      mu         <- mean(pred_vec, na.rm = TRUE)
      sigma      <- sd(pred_vec,   na.rm = TRUE)
      if (is.na(mu) || is.nan(mu))                        mu    <- 0
      if (is.na(sigma) || is.nan(sigma) || sigma < 1e-10) sigma <- 1e-6
      
      f_best_norm <- if (!is.null(f_best)) normalize_value(f_best) else NULL
      
      if (is.null(f_best_norm)) {
        EI <- mu
      } else if (sigma < 1e-6) {
        EI <- max(0, mu - f_best_norm)
      } else {
        z  <- (mu - f_best_norm) / sigma
        EI <- max(0, (mu - f_best_norm) * pnorm(z) + sigma * dnorm(z))
      }
      
      p_feas <- as.numeric(predict(bart_class, new_data = x_in, type = "prob"))
      p_feas <- if (is.na(p_feas) || is.nan(p_feas)) 0.5 else max(0, min(1, p_feas))
      
      post_viol <- bart_machine_get_posterior(bart_violation, new_data = x_in)
      pred_viol <- mean(as.numeric(post_viol$y_hat_posterior_samples), na.rm = TRUE)
      pred_viol <- if (is.na(pred_viol) || is.nan(pred_viol)) 0 else max(0, pred_viol)
      
      acq_value <- if (is.null(f_best_norm)) {
        10 * p_feas - penalty_weight * pred_viol * (1 - p_feas) + 0.01 * mu
      } else {
        EI * p_feas - penalty_weight * pred_viol * (1 - p_feas)
      }
      
      if (is.na(acq_value) || is.nan(acq_value)) return(-1e6)
      return(acq_value)
      
    }, error = function(e) -1e6)
  }
  
  for (iter in 1:n_iter) {
    
    GA_res <- ga(
      type     = "binary",
      nBits    = p,
      fitness  = fitness,
      popSize  = 60,
      maxiter  = 20,
      run      = 15,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = sample.int(.Machine$integer.max, 1)
    )
    
    x_next      <- as.numeric(GA_res@solution[1, ])
    y_next      <- true_f(x_next)
    c_next      <- constraint_fn(x_next)
    c_viol_next <- constraint_violation(x_next)
    
    X            <- rbind(X, x_next)
    X_df         <- data.frame(X)
    colnames(X_df) <- paste0("x", 1:p)
    y            <- c(y, y_next)
    c_vals       <- c(c_vals, as.integer(c_next))
    c_violations <- c(c_violations, c_viol_next)
    
    y_normalized            <- normalize_value(y)
    c_violations_normalized <- normalize_violation(c_violations)
    
    new_row <- data.frame(t(x_next), y = y_next, feasible = as.integer(c_next),
                          violation = c_viol_next)
    colnames(new_row)[1:p] <- paste0("x", 1:p)
    results_df <- rbind(results_df, new_row)
    
    f_best_info <- get_f_best(y, c_vals)
    f_best      <- f_best_info$value
    
    bart_fit       <- bartMachine(X = X_df, y = y_normalized,            num_trees = 100, verbose = FALSE)
    bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),       num_trees = 100, verbose = FALSE)
    bart_violation <- bartMachine(X = X_df, y = c_violations_normalized, num_trees = 100, verbose = FALSE)
    
    if (verbose && iter %% 10 == 0) {
      feas_rate <- sum(c_vals) / length(c_vals)
      cat(sprintf("  Iter %3d | Feasible: %3d (%.1f%%) | f_best: %s\n",
                  iter, sum(c_vals), 100 * feas_rate,
                  ifelse(is.null(f_best), "NA", format(f_best, big.mark = ","))))
    }
  }
  
  list(
    results_df       = results_df,
    f_best           = f_best,
    x_best           = if (!is.null(f_best_info$idx)) X[f_best_info$idx, ] else NULL,
    n_feasible       = sum(c_vals),
    n_total          = length(c_vals),
    feasibility_rate = sum(c_vals) / length(c_vals),
    n_true_evals     = n_init + n_iter
  )
}

# =============================================================================
# BENCHMARK GA
# =============================================================================

run_benchmark_ga <- function(seed_val) {
  set.seed(seed_val)
  
  GA_final <- ga(
    type    = "binary",
    fitness = function(x_vec) {
      x <- as.numeric(x_vec)
      if (constraint_fn(x)) true_f(x) else -Inf
    },
    nBits   = p,
    popSize = 60,
    maxiter = 20,
    run     = 15,
    monitor = FALSE
  )
  
  list(
    x_best       = GA_final@solution[1, ],
    f_best       = GA_final@fitnessValue,
    n_true_evals = GA_final@iter * 60
  )
}

# =============================================================================
# RUN SINGLE TEST
# =============================================================================

PENALTY_WEIGHT <- 2
SEED           <- 1001
N_ITER         <- 250

cat("=============================================================================\n")
cat("SINGLE INSTANCE TEST — penalty weight =", PENALTY_WEIGHT, "| seed =", SEED, "\n")
cat("=============================================================================\n\n")

cat("--- Running BART-BO ---\n")
bart_result <- run_bart_bo(penalty_weight = PENALTY_WEIGHT, seed_val = SEED,
                           n_iter = N_ITER, verbose = TRUE)

cat("\n--- Running benchmark GA ---\n")
ga_result <- run_benchmark_ga(seed_val = SEED)

# =============================================================================
# RESULTS
# =============================================================================

bart_best <- bart_result$f_best
ga_best   <- ga_result$f_best

optimality_gap <- if (!is.null(bart_best) && !is.na(ga_best) && ga_best > 0)
  ((ga_best - bart_best) / ga_best) * 100 else NA

cat("\n")
cat("=============================================================================\n")
cat("RESULTS\n")
cat("=============================================================================\n")
cat(sprintf("BART-BO best:    %s\n",
            ifelse(is.null(bart_best), "NA (no feasible solution found)", format(bart_best, big.mark = ","))))
cat(sprintf("GA best:         %s\n", format(ga_best, big.mark = ",")))
cat(sprintf("Optimality gap:  %s\n",
            ifelse(is.na(optimality_gap), "NA", sprintf("%.2f%%", optimality_gap))))
cat(sprintf("Feasibility:     %.1f%% (%d / %d points)\n",
            bart_result$feasibility_rate * 100,
            bart_result$n_feasible, bart_result$n_total))
cat(sprintf("True evals — BART-BO: %d | GA: %d\n",
            bart_result$n_true_evals, ga_result$n_true_evals))

if (!is.null(bart_result$x_best)) {
  cat(sprintf("\nBest solution (BART-BO): %s\n",
              paste(bart_result$x_best, collapse = "")))
}