# =============================================================================
# HPC WORKER SCRIPT — Constrained BART-BO for 0/1 Knapsack
# Reads SLURM_ARRAY_TASK_ID to determine which (penalty_weight, instance) to run
# Usage:  Rscript knapsack_hpc_worker.R
# =============================================================================
options(java.parameters = "-Xmx4g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

suppressPackageStartupMessages({
  library(dartMachine)
  library(GA)
})

# =============================================================================
# EXPERIMENT GRID  — must match exactly what you use in the job script
# =============================================================================
penalty_weights <- c(0.1, 1, 2, 5, 10)
n_instances     <- 10
n_iter          <- 250

# Build the full grid (row order determines TASK_ID → job mapping)
grid <- expand.grid(
  penalty_weight = penalty_weights,
  instance       = seq_len(n_instances)
)
# grid row 1  →  TASK_ID 1
# grid row 2  →  TASK_ID 2  … etc.

# =============================================================================
# READ ARRAY TASK ID
# Passed as a command-line argument by the SLURM script:
#   srun Rscript knapsack_hpc_worker.R $SLURM_ARRAY_TASK_ID
# Falls back to env variable or 1 for interactive testing.
# =============================================================================
args    <- commandArgs(trailingOnly = TRUE)
task_id <- if (length(args) >= 1) as.integer(args[1]) else
           as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "1"))
if (is.na(task_id) || task_id < 1 || task_id > nrow(grid)) {
  stop(sprintf("Invalid task_id=%s  (grid has %d rows)", task_id, nrow(grid)))
}

pw       <- grid$penalty_weight[task_id]
inst     <- grid$instance[task_id]
seed_val <- 1000L + inst

cat("=============================================================================\n")
cat(sprintf("TASK %d  |  penalty_weight=%.1f  |  instance=%d  |  seed=%d\n",
            task_id, pw, inst, seed_val))
cat("=============================================================================\n")

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
base_output_dir <- "knapsack_results_GA"          # relative to working dir
pw_dir <- file.path(base_output_dir,
                    sprintf("penalty_%.1f", pw))
dir.create(pw_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# PROBLEM SETUP (24-item knapsack)
# =============================================================================
p       <- 24L
weights <- c(382745, 799601, 909247, 729069, 467902,  44328,  34610, 698150,
             823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
             323046, 446298, 931161,  31385, 496951, 264724, 224916, 169684)
values  <- c(825594,1677009,1676628,1523970, 943972,  97426,  69666,1296457,
             1679693,1902996,1844992,1049289,1252836,1319836, 953277,2067538,
             675367, 853655,1826027,  65731, 901489, 577243, 466257, 369261)
W       <- 6404180L

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
true_f <- function(x) sum(values * x)

constraint_fn <- function(x) sum(weights * x) <= W

normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

constraint_violation <- function(x) max(0, sum(weights * x) - W)

get_f_best <- function(y_vec, c_vec) {
  feas_idx <- which(as.logical(c_vec))
  if (length(feas_idx) == 0) return(list(value = NULL, idx = NULL))
  best_idx <- feas_idx[which.max(y_vec[feas_idx])]
  list(value = y_vec[best_idx], idx = best_idx)
}

# =============================================================================
# BART-BO
# =============================================================================
run_bart_bo <- function(penalty_weight, seed_val) {
  set.seed(seed_val)

  # Initial design
  n_init <- 10L
  X      <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
  X_df   <- as.data.frame(X)

  true_eval_count <- n_init
  y               <- apply(X, 1, true_f)
  c_vals          <- apply(X, 1, constraint_fn)
  c_violations    <- apply(X, 1, constraint_violation)

  y_normalized            <- normalize_01(y)
  c_viol_max              <- max(c_violations)
  c_violations_normalized <- if (c_viol_max > 0) c_violations / c_viol_max else c_violations

  cat(sprintf("Initial design: %d feasible / %d\n", sum(c_vals), n_init))

  results_df <- data.frame(X_df, y = y, feasible = c_vals, violation = c_violations)
  colnames(results_df)[1:p] <- paste0("x", 1:p)

  bart_fit       <- bartMachine(X = X_df, y = y_normalized,            num_trees = 100, verbose = FALSE)
  bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),       num_trees = 100, verbose = FALSE)
  bart_violation <- bartMachine(X = X_df, y = c_violations_normalized, num_trees = 100, verbose = FALSE)

  f_best_info           <- get_f_best(y, c_vals)
  f_best                <- f_best_info$value
  current_penalty_weight <- penalty_weight
  infeasible_streak      <- 0L

  # Acquisition function (closure over BART models)
  fitness <- function(x_vec) {
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

      y_min_loc <- min(y); y_max_loc <- max(y)
      f_best_norm <- if (!is.null(f_best)) (f_best - y_min_loc) / (y_max_loc - y_min_loc) else NULL

      EI <- if (is.null(f_best_norm)) {
        mu
      } else if (sigma < 1e-6) {
        max(0, mu - f_best_norm)
      } else {
        z  <- (mu - f_best_norm) / sigma
        max((mu - f_best_norm) * pnorm(z) + sigma * dnorm(z), 0)
      }

      p_feas <- as.numeric(predict(bart_class, new_data = x_in, type = "prob"))
      p_feas <- max(min(p_feas, 1), 0)
      if (is.na(p_feas) || is.nan(p_feas)) p_feas <- 0.5

      post_viol <- bart_machine_get_posterior(bart_violation, new_data = x_in)
      pred_viol <- mean(as.numeric(post_viol$y_hat_posterior_samples), na.rm = TRUE)
      if (is.na(pred_viol) || is.nan(pred_viol)) pred_viol <- 0
      pred_viol <- max(0, pred_viol)

      acq_value <- if (is.null(f_best_norm)) {
        10 * p_feas - current_penalty_weight * pred_viol + 0.01 * mu
      } else {
        EI * p_feas - current_penalty_weight * pred_viol
      }

      if (is.na(acq_value) || is.nan(acq_value)) return(-1e6)
      acq_value
    }, error = function(e) -1e6)
  }

  # BO loop
  for (iter in seq_len(n_iter)) {

    if (infeasible_streak > 3)
      current_penalty_weight <- min(current_penalty_weight * 1.5, 20)

    GA_res <- ga(
      type    = "binary",
      nBits   = p,
      fitness = fitness,
      popSize = 200,
      maxiter = 40,
      run     = 20,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = sample.int(.Machine$integer.max, 1)
    )

    x_next      <- as.numeric(GA_res@solution[1, ])
    y_next      <- true_f(x_next)
    c_next      <- constraint_fn(x_next)
    c_viol_next <- constraint_violation(x_next)
    true_eval_count <- true_eval_count + 1L

    if (!c_next) {
      infeasible_streak <- infeasible_streak + 1L
    } else {
      infeasible_streak      <- 0L
      current_penalty_weight <- max(current_penalty_weight * 0.9, penalty_weight)
    }

    X            <- rbind(X, x_next)
    X_df         <- as.data.frame(X)
    y            <- c(y, y_next)
    c_vals       <- c(c_vals, as.integer(c_next))
    c_violations <- c(c_violations, c_viol_next)

    y_normalized            <- normalize_01(y)
    c_viol_max              <- max(c_violations)
    c_violations_normalized <- if (c_viol_max > 0) c_violations / c_viol_max else c_violations

    new_row <- data.frame(t(x_next), y = y_next,
                          feasible  = as.integer(c_next),
                          violation = c_viol_next)
    colnames(new_row)[1:p] <- paste0("x", 1:p)
    results_df <- rbind(results_df, new_row)

    f_best_info <- get_f_best(y, c_vals)
    f_best      <- f_best_info$value

    bart_fit       <- bartMachine(X = X_df, y = y_normalized,            num_trees = 100, verbose = FALSE)
    bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),       num_trees = 100, verbose = FALSE)
    bart_violation <- bartMachine(X = X_df, y = c_violations_normalized, num_trees = 100, verbose = FALSE)

    if (iter %% 10 == 0) {
      feas_rate <- sum(c_vals) / length(c_vals)
      cat(sprintf("  Iter %3d | True evals: %4d | Feasible: %3d (%.1f%%) | f_best: %s\n",
                  iter, true_eval_count, sum(c_vals), 100 * feas_rate,
                  ifelse(is.null(f_best), "NA", format(f_best))))
    }
  }

  list(
    results_df       = results_df,
    f_best           = f_best,
    f_best_idx       = f_best_info$idx,
    x_best           = if (!is.null(f_best_info$idx)) X[f_best_info$idx, ] else NULL,
    n_feasible       = sum(c_vals),
    n_total          = length(c_vals),
    feasibility_rate = sum(c_vals) / length(c_vals),
    true_eval_count  = true_eval_count
  )
}

# =============================================================================
# BENCHMARK GA
# =============================================================================
run_benchmark_ga <- function(seed_val) {
  set.seed(seed_val)
  ga_eval_count <- 0L
  true_feas_eval <- function(x_vec) {
    x <- as.numeric(x_vec)
    ga_eval_count <<- ga_eval_count + 1L
    if (constraint_fn(x)) true_f(x) else -Inf
  }
  GA_final <- ga(type = "binary", fitness = true_feas_eval,
                 nBits = p, monitor = FALSE)
  list(x_best          = GA_final@solution[1, ],
       f_best          = GA_final@fitnessValue,
       true_eval_count = ga_eval_count)
}

# =============================================================================
# RUN THIS TASK
# =============================================================================
cat("Running BART-BO...\n")
t0_bart <- proc.time()
bart_result <- run_bart_bo(penalty_weight = pw, seed_val = seed_val)
t1_bart <- proc.time()

cat("Running benchmark GA...\n")
t0_ga <- proc.time()
ga_result <- run_benchmark_ga(seed_val = seed_val)
t1_ga <- proc.time()

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

# 1. Full iteration-level results (CSV)
instance_file <- file.path(pw_dir, sprintf("instance_%02d.csv", inst))
write.csv(bart_result$results_df, instance_file, row.names = FALSE)

# 2. Per-task summary (RDS — lossless, easy to rbind locally)
bart_best      <- bart_result$f_best
ga_best        <- ga_result$f_best
optimality_gap <- if (!is.null(bart_best) && !is.na(bart_best)) {
  ((ga_best - bart_best) / ga_best) * 100
} else NA_real_

summary_row <- data.frame(
  task_id               = task_id,
  penalty_weight        = pw,
  instance              = inst,
  seed                  = seed_val,
  bart_best             = ifelse(is.null(bart_best), NA_real_, bart_best),
  ga_best               = ga_best,
  optimality_gap        = optimality_gap,
  feasibility_pct       = bart_result$feasibility_rate * 100,
  n_feasible            = bart_result$n_feasible,
  n_total               = bart_result$n_total,
  bart_true_eval_count  = bart_result$true_eval_count,
  ga_true_eval_count    = ga_result$true_eval_count,
  bart_wall_sec         = as.numeric((t1_bart - t0_bart)["elapsed"]),
  ga_wall_sec           = as.numeric((t1_ga   - t0_ga  )["elapsed"]),
  x_best                = ifelse(is.null(bart_result$x_best), NA_character_,
                                 paste(bart_result$x_best, collapse = ""))
)

summary_rds <- file.path(pw_dir, sprintf("summary_instance_%02d.rds", inst))
saveRDS(summary_row, summary_rds)

summary_csv <- file.path(pw_dir, sprintf("summary_instance_%02d.csv", inst))
write.csv(summary_row, summary_csv, row.names = FALSE)

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
cat("\n--- RESULTS ---\n")
cat(sprintf("  BART-BO best:        %s\n",
            ifelse(is.na(summary_row$bart_best), "NA (no feasible)", summary_row$bart_best)))
cat(sprintf("  GA best:             %.0f\n",    ga_best))
cat(sprintf("  Optimality gap:      %s\n",
            ifelse(is.na(optimality_gap), "NA", sprintf("%.2f%%", optimality_gap))))
cat(sprintf("  Feasibility:         %.1f%% (%d/%d)\n",
            summary_row$feasibility_pct, bart_result$n_feasible, bart_result$n_total))
cat(sprintf("  BART-BO true evals:  %d\n",  bart_result$true_eval_count))
cat(sprintf("  GA true evals:       %d\n",  ga_result$true_eval_count))
cat(sprintf("  BART-BO wall time:   %.1f s\n", summary_row$bart_wall_sec))
cat(sprintf("  GA wall time:        %.1f s\n", summary_row$ga_wall_sec))
cat(sprintf("\nOutputs written to: %s\n", pw_dir))
cat("=============================================================================\n")
cat("TASK COMPLETE\n")
cat("=============================================================================\n")
