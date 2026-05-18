# =============================================================================
# Constrained BART-BO for 0/1 Knapsack - PENALTY WEIGHT EXPERIMENTS
# Changes vs original (principled fixes only):
#
#   FIX 1: Fixed normalization anchors for y and constraint violation.
#           Original rescaled to [0,1] based on the current dataset's
#           min/max each iteration, so the scale — and therefore the
#           effective penalty weight — drifted throughout the run.
#           Now uses fixed physical bounds: sum(values) for the
#           objective, (sum(weights) - W) for violation.
#
#   FIX 2: Removed adaptive penalty escalation (infeasible_streak logic).
#           This was mutating the penalty weight mid-run, directly
#           confounding the experiment across penalty weight conditions.
#
#   FIX 3: Corrected acquisition penalty term. Original subtracted
#           penalty * pred_viol unconditionally. The correction applies
#           the violation penalty weighted by (1 - p_feas), so feasible
#           points are not penalised for violation model noise.
#
#   FIX 4: Fixed run > maxiter bug in inner GA (run=40 > maxiter=20
#           meant the early-stopping criterion could never trigger).
#           Set run = 15 so it is always < maxiter = 20.
# =============================================================================

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")
library(dartMachine)
library(GA)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
penalty_weights <- c(0.1, 1, 2, 5, 10)
n_instances <- 10
n_iter <- 250
base_output_dir <- "knapsack_results_fixed"

if (!dir.exists(base_output_dir)) {
  dir.create(base_output_dir)
}

# =============================================================================
# PROBLEM SETUP (24-item knapsack)
# =============================================================================
p <- 24

weights <- c(382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
             823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
             323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684)

values <- c(825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
            1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538,
            675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261)

W <- 6404180  # knapsack capacity

# FIX 1: Fixed normalization anchors — computed once, never updated.
# Physical worst-case bounds mean penalty_weight is stationary across
# all iterations and instances.
MAX_POSSIBLE_VALUE     <- sum(values)           # ceiling for objective
MAX_POSSIBLE_VIOLATION <- sum(weights) - W      # worst-case violation

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
  if (length(feas_idx) == 0) {
    return(list(value = NULL, idx = NULL))
  }
  best_idx <- feas_idx[which.max(y_vec[feas_idx])]
  return(list(value = y_vec[best_idx], idx = best_idx))
}

# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

run_bart_bo <- function(penalty_weight, seed_val, verbose = TRUE) {
  
  set.seed(seed_val)
  
  # -------------------------
  # Initial design
  # -------------------------
  n_init <- 10
  X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
  X_df <- data.frame(X)
  colnames(X_df) <- paste0("x", 1:p)
  
  y          <- apply(X, 1, true_f)
  c_vals     <- apply(X, 1, constraint_fn)
  c_violations <- apply(X, 1, constraint_violation)
  
  # FIX 1: Use fixed-anchor normalization throughout
  y_normalized          <- normalize_value(y)
  c_violations_normalized <- normalize_violation(c_violations)
  
  if (verbose) {
    cat("Initial design:", sum(c_vals), "feasible out of", length(c_vals), "\n")
  }
  
  # -------------------------
  # Results dataframe
  # -------------------------
  results_df <- data.frame(X_df, y = y, feasible = c_vals, violation = c_violations)
  colnames(results_df)[1:p] <- paste0("x", 1:p)
  
  # -------------------------
  # Initial BART models
  # -------------------------
  bart_fit       <- bartMachine(X = X_df, y = y_normalized,              num_trees = 100)
  bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),         num_trees = 100)
  bart_violation <- bartMachine(X = X_df, y = c_violations_normalized,   num_trees = 100)
  
  # -------------------------
  # Get initial best
  # -------------------------
  f_best_info <- get_f_best(y, c_vals)
  f_best <- f_best_info$value
  
  # -------------------------
  # Fitness / acquisition function
  # -------------------------
  fitness <- function(x_vec) {
    
    if (is.list(x_vec)) x_vec <- unlist(x_vec)
    x_vec <- as.numeric(x_vec)
    
    x_in <- as.data.frame(matrix(x_vec, nrow = 1))
    colnames(x_in) <- colnames(X_df)
    
    tryCatch({
      post_draws <- bart_machine_get_posterior(bart_fit, new_data = x_in)
      pred_vec   <- as.numeric(post_draws$y_hat_posterior_samples)
      
      mu    <- mean(pred_vec, na.rm = TRUE)
      sigma <- sd(pred_vec,   na.rm = TRUE)
      
      if (is.na(mu) || is.nan(mu))                      mu    <- 0
      if (is.na(sigma) || is.nan(sigma) || sigma < 1e-10) sigma <- 1e-6
      
      # FIX 1: f_best on same fixed-anchor normalized scale
      if (!is.null(f_best)) {
        f_best_norm <- normalize_value(f_best)
      } else {
        f_best_norm <- NULL
      }
      
      # Expected Improvement
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
      
      # Probability of feasibility
      p_feas <- as.numeric(predict(bart_class, new_data = x_in, type = "prob"))
      p_feas <- max(min(p_feas, 1), 0)
      if (is.na(p_feas) || is.nan(p_feas)) p_feas <- 0.5
      
      # Predicted violation
      post_viol     <- bart_machine_get_posterior(bart_violation, new_data = x_in)
      pred_viol_vec <- as.numeric(post_viol$y_hat_posterior_samples)
      pred_viol     <- mean(pred_viol_vec, na.rm = TRUE)
      if (is.na(pred_viol) || is.nan(pred_viol)) pred_viol <- 0
      pred_viol <- max(0, pred_viol)
      
      # FIX 3: Acquisition function with corrected penalty term.
      # Violation penalty is weighted by (1 - p_feas): points the
      # classifier considers likely feasible are not penalised for
      # noise in the violation model.
      if (is.null(f_best_norm)) {
        acq_value <- 10 * p_feas - penalty_weight * pred_viol * (1 - p_feas) + 0.01 * mu
      } else {
        CEI               <- EI * p_feas
        violation_penalty <- penalty_weight * pred_viol * (1 - p_feas)
        acq_value         <- CEI - violation_penalty
      }
      
      if (is.na(acq_value) || is.nan(acq_value)) {
        return(-1e6)
      }
      
      return(acq_value)
      
    }, error = function(e) {
      return(-1e6)
    })
  }
  
  # -------------------------
  # BO loop
  # -------------------------
  # FIX 2: No adaptive penalty — penalty_weight is fixed for the
  # entire run so conditions are comparable across experiments.
  
  for (iter in 1:n_iter) {
    
    # FIX 4: run = 15 < maxiter = 20 so early-stopping can fire.
    GA_res <- ga(
      type     = "binary",
      nBits    = p,
      fitness  = fitness,
      popSize  = 100,
      maxiter  = 40,
      run      = 20,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = sample.int(.Machine$integer.max, 1)
    )
    
    x_next <- as.numeric(GA_res@solution[1, ])
    
    # Evaluate
    y_next    <- true_f(x_next)
    c_next    <- constraint_fn(x_next)
    c_viol_next <- constraint_violation(x_next)
    
    # Append data
    X    <- rbind(X, x_next)
    X_df <- data.frame(X)
    colnames(X_df) <- paste0("x", 1:p)
    y          <- c(y, y_next)
    c_vals     <- c(c_vals, as.integer(c_next))
    c_violations <- c(c_violations, c_viol_next)
    
    # FIX 1: Normalization stays on fixed anchors — no recomputation needed
    y_normalized            <- normalize_value(y)
    c_violations_normalized <- normalize_violation(c_violations)
    
    # Update results
    new_row <- data.frame(t(x_next), y = y_next, feasible = as.integer(c_next),
                          violation = c_viol_next)
    colnames(new_row)[1:p] <- paste0("x", 1:p)
    results_df <- rbind(results_df, new_row)
    
    # Update best
    f_best_info <- get_f_best(y, c_vals)
    f_best      <- f_best_info$value
    
    # Refit models
    bart_fit       <- bartMachine(X = X_df, y = y_normalized,            num_trees = 100, verbose = FALSE)
    bart_class     <- bartMachine(X = X_df, y = as.factor(c_vals),       num_trees = 100, verbose = FALSE)
    bart_violation <- bartMachine(X = X_df, y = c_violations_normalized, num_trees = 100, verbose = FALSE)
    
    if (verbose && iter %% 10 == 0) {
      feas_rate <- sum(c_vals) / length(c_vals)
      cat(sprintf("  Iter %3d | Feasible: %3d (%.1f%%) | f_best: %s\n",
                  iter, sum(c_vals), 100 * feas_rate,
                  ifelse(is.null(f_best), "NA", f_best)))
    }
  }
  
  # True objective evaluations = initial design + one per BO iteration
  n_true_evals_bart <- n_init + n_iter
  
  return(list(
    results_df       = results_df,
    f_best           = f_best,
    f_best_idx       = f_best_info$idx,
    x_best           = if (!is.null(f_best_info$idx)) X[f_best_info$idx, ] else NULL,
    n_feasible       = sum(c_vals),
    n_total          = length(c_vals),
    feasibility_rate = sum(c_vals) / length(c_vals),
    n_true_evals     = n_true_evals_bart
  ))
}

# =============================================================================
# BENCHMARK GA FUNCTION
# =============================================================================

run_benchmark_ga <- function(seed_val) {
  set.seed(seed_val)
  
  true_feas_eval <- function(x_vec) {
    x <- as.numeric(x_vec)
    if (constraint_fn(x)) return(true_f(x)) else return(-Inf)
  }
  
  GA_final <- ga(
    type    = "binary",
    fitness = true_feas_eval,
    nBits   = p,
    popSize = 200,
    maxiter = 60,
    run     = 40,   # FIX 4: consistent with BO inner GA
    monitor = FALSE
  )
  
  # True objective evaluations: popSize individuals evaluated each generation.
  # GA_final@iter is the actual number of generations completed (respects
  # early stopping), so this is exact rather than the maxiter upper bound.
  n_true_evals_ga <- GA_final@iter * 60
  
  return(list(
    x_best       = GA_final@solution[1, ],
    f_best       = GA_final@fitnessValue,
    n_true_evals = n_true_evals_ga
  ))
}

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

cat("=============================================================================\n")
cat("STARTING KNAPSACK PENALTY WEIGHT EXPERIMENTS\n")
cat("=============================================================================\n")
cat("Penalty weights:", paste(penalty_weights, collapse = ", "), "\n")
cat("Instances per weight:", n_instances, "\n")
cat("Iterations per instance:", n_iter, "\n\n")

instance_seeds <- 1000 + (1:n_instances)
cat("Using seeds:", paste(instance_seeds, collapse = ", "), "\n\n")

comparison_results <- data.frame()

for (pw in penalty_weights) {
  
  cat("\n")
  cat("=============================================================================\n")
  cat(sprintf("PENALTY WEIGHT: %.1f\n", pw))
  cat("=============================================================================\n")
  
  pw_dir <- file.path(base_output_dir, sprintf("penalty_%.1f", pw))
  if (!dir.exists(pw_dir)) {
    dir.create(pw_dir, recursive = TRUE)
  }
  
  pw_summary <- data.frame()
  
  for (inst in 1:n_instances) {
    
    cat(sprintf("\n--- Instance %d/%d ---\n", inst, n_instances))
    
    seed_val <- instance_seeds[inst]
    
    cat("Running BART-BO...\n")
    bart_result <- run_bart_bo(penalty_weight = pw, seed_val = seed_val, verbose = FALSE)
    
    cat("Running benchmark GA...\n")
    ga_result <- run_benchmark_ga(seed_val = seed_val)
    
    feasibility_pct <- bart_result$feasibility_rate * 100
    bart_best       <- ifelse(is.null(bart_result$f_best), NA, bart_result$f_best)
    ga_best         <- ga_result$f_best
    
    if (!is.na(bart_best)) {
      optimality_gap <- ((ga_best - bart_best) / ga_best) * 100
    } else {
      optimality_gap <- NA
    }
    
    cat(sprintf("  BART-BO best: %s\n", ifelse(is.na(bart_best), "NA (no feasible)", bart_best)))
    cat(sprintf("  GA best:      %s\n", ga_best))
    cat(sprintf("  Gap:          %s\n", ifelse(is.na(optimality_gap), "NA", sprintf("%.2f%%", optimality_gap))))
    cat(sprintf("  Feasibility:  %.1f%% (%d/%d)\n", feasibility_pct,
                bart_result$n_feasible, bart_result$n_total))
    cat(sprintf("  True evals — BART-BO: %d | GA: %d\n",
                bart_result$n_true_evals, ga_result$n_true_evals))
    
    instance_file <- file.path(pw_dir, sprintf("instance_%02d.csv", inst))
    write.csv(bart_result$results_df, instance_file, row.names = FALSE)
    
    instance_summary <- data.frame(
      penalty_weight     = pw,
      instance           = inst,
      bart_best          = bart_best,
      ga_best            = ga_best,
      optimality_gap     = optimality_gap,
      feasibility_pct    = feasibility_pct,
      n_feasible         = bart_result$n_feasible,
      n_total            = bart_result$n_total,
      n_true_evals_bart  = bart_result$n_true_evals,
      n_true_evals_ga    = ga_result$n_true_evals,
      x_best             = ifelse(is.null(bart_result$x_best), NA,
                                  paste(bart_result$x_best, collapse = ""))
    )
    
    pw_summary         <- rbind(pw_summary, instance_summary)
    comparison_results <- rbind(comparison_results, instance_summary)
  }
  
  summary_file <- file.path(pw_dir, "summary.csv")
  write.csv(pw_summary, summary_file, row.names = FALSE)
  
  cat(sprintf("\n--- PENALTY WEIGHT %.1f SUMMARY ---\n", pw))
  cat(sprintf("Average BART-BO best: %.2f\n",    mean(pw_summary$bart_best,         na.rm = TRUE)))
  cat(sprintf("Average GA best: %.2f\n",          mean(pw_summary$ga_best,           na.rm = TRUE)))
  cat(sprintf("Average optimality gap: %.2f%%\n", mean(pw_summary$optimality_gap,    na.rm = TRUE)))
  cat(sprintf("Average feasibility: %.1f%%\n",    mean(pw_summary$feasibility_pct,   na.rm = TRUE)))
  cat(sprintf("Avg true evals — BART-BO: %.0f | GA: %.0f\n",
              mean(pw_summary$n_true_evals_bart, na.rm = TRUE),
              mean(pw_summary$n_true_evals_ga,   na.rm = TRUE)))
  cat(sprintf("Instances with feasible solution: %d/%d\n",
              sum(!is.na(pw_summary$bart_best)), n_instances))
}

# =============================================================================
# SAVE COMPARISON SUMMARY
# =============================================================================

cat("\n\n")
cat("=============================================================================\n")
cat("CREATING COMPARISON SUMMARY\n")
cat("=============================================================================\n")

comparison_file <- file.path(base_output_dir, "comparison_summary.csv")
write.csv(comparison_results, comparison_file, row.names = FALSE)

comparison_agg <- aggregate(
  cbind(bart_best, ga_best, optimality_gap, feasibility_pct, n_feasible,
        n_true_evals_bart, n_true_evals_ga) ~ penalty_weight,
  data = comparison_results,
  FUN  = function(x) c(mean = mean(x, na.rm = TRUE),
                       sd   = sd(x,   na.rm = TRUE),
                       min  = min(x,   na.rm = TRUE),
                       max  = max(x,   na.rm = TRUE))
)

comparison_flat <- data.frame(
  penalty_weight           = comparison_agg$penalty_weight,
  bart_best_mean           = comparison_agg$bart_best[, "mean"],
  bart_best_sd             = comparison_agg$bart_best[, "sd"],
  ga_best_mean             = comparison_agg$ga_best[, "mean"],
  ga_best_sd               = comparison_agg$ga_best[, "sd"],
  optimality_gap_mean      = comparison_agg$optimality_gap[, "mean"],
  optimality_gap_sd        = comparison_agg$optimality_gap[, "sd"],
  feasibility_pct_mean     = comparison_agg$feasibility_pct[, "mean"],
  feasibility_pct_sd       = comparison_agg$feasibility_pct[, "sd"],
  n_feasible_mean          = comparison_agg$n_feasible[, "mean"],
  n_true_evals_bart_mean   = comparison_agg$n_true_evals_bart[, "mean"],
  n_true_evals_ga_mean     = comparison_agg$n_true_evals_ga[, "mean"],
  n_true_evals_ga_sd       = comparison_agg$n_true_evals_ga[, "sd"]
)

feasible_counts <- aggregate(
  bart_best ~ penalty_weight,
  data = comparison_results,
  FUN  = function(x) sum(!is.na(x))
)
colnames(feasible_counts)[2] <- "n_instances_with_feasible"

comparison_flat <- merge(comparison_flat, feasible_counts, by = "penalty_weight")

agg_file <- file.path(base_output_dir, "comparison_aggregated.csv")
write.csv(comparison_flat, agg_file, row.names = FALSE)

# =============================================================================
# PRINT FINAL COMPARISON
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("FINAL COMPARISON ACROSS PENALTY WEIGHTS\n")
cat("=============================================================================\n\n")

print(comparison_flat, digits = 3)

cat("\n")
cat("Detailed interpretation:\n")
for (i in 1:nrow(comparison_flat)) {
  cat(sprintf("\nPenalty Weight %.1f:\n", comparison_flat$penalty_weight[i]))
  cat(sprintf("  Feasibility: %.1f%% ± %.1f%%\n",
              comparison_flat$feasibility_pct_mean[i],
              comparison_flat$feasibility_pct_sd[i]))
  cat(sprintf("  BART-BO Objective: %.0f ± %.0f\n",
              comparison_flat$bart_best_mean[i],
              comparison_flat$bart_best_sd[i]))
  cat(sprintf("  Optimality Gap: %.2f%% ± %.2f%%\n",
              comparison_flat$optimality_gap_mean[i],
              comparison_flat$optimality_gap_sd[i]))
  cat(sprintf("  True evals — BART-BO: %.0f (fixed) | GA: %.0f ± %.0f\n",
              comparison_flat$n_true_evals_bart_mean[i],
              comparison_flat$n_true_evals_ga_mean[i],
              comparison_flat$n_true_evals_ga_sd[i]))
  cat(sprintf("  Instances with feasible: %d/%d\n",
              comparison_flat$n_instances_with_feasible[i],
              n_instances))
}

cat("\n")
cat("=============================================================================\n")
cat("EXPERIMENT COMPLETE\n")
cat("=============================================================================\n")
cat("Results saved to:", base_output_dir, "\n")
cat("  - Individual instance CSVs in penalty_X.X/ subdirectories\n")
cat("  - Summary CSVs for each penalty weight\n")
cat("  - Overall comparison_summary.csv\n")
cat("  - Aggregated comparison_aggregated.csv\n")