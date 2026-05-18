# =============================================================================
# MCLP — DART-CBO vs GA vs GLPK  (Multi-run comparison)
#
# Runs:
#   - GLPK    : 1 run  (exact MIP solver — baseline)
#   - GA      : 1 run  (pure genetic algorithm, no surrogate)
#   - DART-CBO: 10 runs (Bayesian optimisation with DART surrogate)
#
# Outputs (all in results_MCLP_comparison/):
#   - glpk_result.csv          (single row: metrics + x1..x16)
#   - ga_result.csv            (single row: metrics + x1..x16)
#   - dart_run_{i}.csv         (n_init+n_iter rows: history + x1..x16 per row)
#   - summary_all_runs.csv     (one row per run, all metrics)
#   - summary_method_stats.csv (aggregated mean/SD per method)
# =============================================================================

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

library(tidyr)
library(dplyr)
library(dartMachine)
library(GA)
library(ompr)
library(ompr.roi)
library(ROI)
library(ROI.plugin.glpk)

cat("=== Libraries loaded OK ===\n\n")

# =============================================================================
# SETTINGS
# =============================================================================
N_RUNS         <- 10
n_init         <-  10
n_iter         <- 250
num_trees      <- 100
penalty_w      <-   5
max_facilities <-   4
service_radius <- 5000

GA_POPSIZE <- 50
GA_MAXITER <- ceiling((n_init + n_iter) / GA_POPSIZE)

output_dir <- "results_MCLP_comparison"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# =============================================================================
# DATA + COVERAGE MATRIX
# =============================================================================
demand_data     <- read.csv("PhD-Data/SF_data/SF_demand_205_centroid_uniform_weight.csv")
facility_loc    <- read.csv("PhD-Data/SF_data/SF_store_site_16_longlat.csv")
distance_matrix <- read.csv("PhD-Data/SF_data/SF_network_distance_candidateStore_16_censusTract_205_new.csv")

distance_matrix$covered <- as.integer(distance_matrix$distance <= service_radius)

A_df <- distance_matrix %>%
  dplyr::select(DestinationName, name, covered) %>%
  tidyr::pivot_wider(names_from  = name,
                     values_from = covered,
                     values_fill = list(covered = 0))

A         <- as.matrix(A_df[, -1])
n_demand  <- nrow(A)
n_vars    <- ncol(A)
col_names <- paste0("x", seq_len(n_vars))

demand_lookup <- demand_data %>%
  dplyr::select(NAME, POP2000) %>%
  dplyr::rename(DestinationName = NAME, population = POP2000)

population_df <- A_df %>%
  dplyr::select(DestinationName) %>%
  dplyr::left_join(demand_lookup, by = "DestinationName")

population <- population_df$population
total_pop  <- sum(population)

cat(sprintf("Problem   : %d demand points, %d candidate facilities\n", n_demand, n_vars))
cat(sprintf("Population: %d  |  Max facilities: %d\n\n", total_pop, max_facilities))

# =============================================================================
# MCLP HELPERS
# =============================================================================
mclp_coverage <- function(x) {
  coverage_vec <- pmin(as.vector(A %*% x), 1)
  sum(coverage_vec * population)
}

mclp_feasible  <- function(x) as.integer(sum(x) <= max_facilities)
mclp_violation <- function(x) max(0L, sum(x) - max_facilities)

normalize_01 <- function(x) {
  r <- range(x)
  if (diff(r) == 0) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

get_f_best <- function(cov, feas) {
  idx <- which(as.logical(feas))
  if (!length(idx)) return(list(value = NULL, idx = NULL))
  bi <- idx[which.max(cov[idx])]
  list(value = cov[bi], idx = bi)
}

# =============================================================================
# 1. GLPK — exact reference solution (1 run)
# =============================================================================
cat("=================================================================\n")
cat("GLPK EXACT SOLVER\n")
cat("=================================================================\n")

glpk_t0 <- proc.time()["elapsed"]

lp_model <- MIPModel() %>%
  add_variable(x[j], j = 1:n_vars,   type = "binary") %>%
  add_variable(z[i], i = 1:n_demand, type = "binary") %>%
  add_constraint(z[i] <= sum_expr(A[i, j] * x[j], j = 1:n_vars), i = 1:n_demand) %>%
  add_constraint(sum_expr(x[j], j = 1:n_vars) <= max_facilities) %>%
  set_objective(sum_expr(population[i] * z[i], i = 1:n_demand), sense = "max")

lp_res    <- solve_model(lp_model, with_ROI(solver = "glpk"))
glpk_obj  <- lp_res$objective_value
glpk_time <- proc.time()["elapsed"] - glpk_t0
glpk_pct  <- 100 * glpk_obj / total_pop

glpk_x   <- as.integer(get_solution(lp_res, x[j])$value)
glpk_pct <- 100 * glpk_obj / total_pop

cat(sprintf("GLPK optimal coverage: %.0f people (%.1f%%)  |  Time: %.2f s\n",
            glpk_obj, glpk_pct, glpk_time))
cat(sprintf("GLPK facilities selected: %s\n\n", paste(which(glpk_x == 1), collapse = ", ")))

# Single row: metrics + full x vector
glpk_result <- data.frame(
  method         = "GLPK",
  run            = 1,
  best_coverage  = glpk_obj,
  coverage_pct   = glpk_pct,
  gap_from_glpk  = 0,
  gap_pct        = 0,
  true_evals     = NA,
  time_seconds   = glpk_time,
  feasible_evals = NA,
  feasible_rate  = NA,
  as.data.frame(matrix(glpk_x, nrow = 1, dimnames = list(NULL, col_names)))
)
write.csv(glpk_result, file.path(output_dir, "glpk_result.csv"), row.names = FALSE)
cat(sprintf("Saved: glpk_result.csv\n\n"))

# =============================================================================
# 2. GA BASELINE — log every fitness call to capture all evaluated x vectors
# =============================================================================
cat("=================================================================\n")
cat("GA BASELINE\n")
cat("=================================================================\n\n")

# Log every solution evaluated by GA
ga_eval_log <- list()

ga_fitness <- function(x_vec) {
  x    <- as.integer(round(x_vec))
  cov  <- mclp_coverage(x)
  viol <- mclp_violation(x)
  feas <- mclp_feasible(x)
  # Append to log in parent environment
  ga_eval_log[[length(ga_eval_log) + 1]] <<- c(
    coverage  = cov,
    feasible  = feas,
    violation = viol,
    x
  )
  cov - 1e7 * viol
}

ga_t0  <- proc.time()["elapsed"]

ga_res <- ga(
  type     = "binary",
  fitness  = ga_fitness,
  nBits    = n_vars,
  keepBest = TRUE,
  monitor  = FALSE,
  seed     = 42
)

ga_time  <- proc.time()["elapsed"] - ga_t0
ga_x     <- as.integer(ga_res@solution[1, ])
best_cov <- mclp_coverage(ga_x)
ga_evals <- length(ga_eval_log)
ga_pct   <- 100 * best_cov / total_pop
ga_gap   <- glpk_obj - best_cov
ga_gap_pct <- 100 * ga_gap / glpk_obj

cat(sprintf("Coverage: %.0f (%.2f%%)  |  Gap: %.0f (%.2f%%)  |  Evals: %d  |  Time: %.1f s\n",
            best_cov, ga_pct, ga_gap, ga_gap_pct, ga_evals, ga_time))
cat(sprintf("GA facilities selected: %s\n\n", paste(which(ga_x == 1), collapse = ", ")))

# Build full evaluation history from log
ga_log_mat <- do.call(rbind, ga_eval_log)
colnames(ga_log_mat) <- c("coverage", "feasible", "violation", col_names)

ga_history <- as.data.frame(ga_log_mat) %>%
  mutate(
    method        = "GA",
    run           = 1,
    iteration     = seq_len(n()),
    true_evals    = seq_len(n()),
    best_cov_so_far = cummax(ifelse(feasible == 1, coverage, NA_real_))
  ) %>%
  dplyr::select(method, run, iteration, coverage, feasible, violation,
                best_cov_so_far, true_evals, everything())

write.csv(ga_history, file.path(output_dir, "ga_result.csv"), row.names = FALSE)
cat(sprintf("Saved: ga_result.csv  (%d rows, each with x1..x%d)\n\n", nrow(ga_history), n_vars))

ga_result_summary <- data.frame(
  method         = "GA",
  run            = 1,
  best_coverage  = best_cov,
  coverage_pct   = ga_pct,
  gap_from_glpk  = ga_gap,
  gap_pct        = ga_gap_pct,
  true_evals     = ga_evals,
  time_seconds   = ga_time,
  feasible_evals = sum(ga_history$feasible, na.rm = TRUE),
  feasible_rate  = mean(ga_history$feasible, na.rm = TRUE)
)

# =============================================================================
# DART SURROGATE FITTING HELPER
# =============================================================================
fit_dart_models <- function(X_df, y_norm, feas_bin, viol_norm) {
  
  obj_model  <- bartMachine(X = X_df, y = y_norm,   num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  viol_model <- bartMachine(X = X_df, y = viol_norm, num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  
  feas_model <- if (length(unique(feas_bin)) == 2) {
    cat("  [feas model] fitting classifier on all data so far\n")
    bartMachine(X = X_df,
                y = factor(feas_bin, levels = c("0", "1")),
                num_trees = num_trees, alpha = 1, verbose = FALSE)
  } else {
    NULL
  }
  
  list(obj = obj_model, feas = feas_model, viol = viol_model)
}

# =============================================================================
# 3. DART-CBO — 10 runs
# =============================================================================
cat("=================================================================\n")
cat("DART-CBO  (10 runs)\n")
cat("=================================================================\n\n")

dart_summary_rows <- vector("list", N_RUNS)

for (run in seq_len(N_RUNS)) {
  
  cat(sprintf("--- DART-CBO run %d / %d ---\n", run, N_RUNS))
  dart_t0 <- proc.time()["elapsed"]
  
  # ---- Initial design -------------------------------------------------------
  set.seed(run * 7)
  X        <- matrix(rbinom(n_init * n_vars, 1, 0.5), nrow = n_init)
  X_df     <- as.data.frame(X); colnames(X_df) <- col_names
  
  y        <- apply(X, 1, mclp_coverage)
  feas_bin <- apply(X, 1, mclp_feasible)
  viols    <- apply(X, 1, mclp_violation)
  
  cat(sprintf("Initial design: %d feasible / %d\n", sum(feas_bin), n_init))
  
  # ---- Fit initial surrogates -----------------------------------------------
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  
  cat("Fitting initial DART surrogates...\n")
  models <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
  if (is.null(models$feas))
    cat("  [feas model] both classes not yet observed — p_feas = 0.5\n")
  cat("Surrogates fitted OK\n\n")
  
  # ---- BO state -------------------------------------------------------------
  fb_info     <- get_f_best(y, feas_bin)
  f_best      <- fb_info$value
  y_range     <- max(max(y) - min(y), 1e-8)
  f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
  
  current_pw        <- penalty_w
  infeasible_streak <- 0
  true_evals        <- n_init
  
  # ---- History initialised with initial design rows (includes x vectors) ----
  history <- data.frame(
    iteration       = rep(0, n_init),
    coverage        = y,
    feasible        = feas_bin,
    violation       = viols,
    best_cov_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals      = seq_len(n_init),
    as.data.frame(X)
  )
  colnames(history)[(ncol(history) - n_vars + 1):ncol(history)] <- col_names
  
  # ---- Acquisition function -------------------------------------------------
  acq_fn <- function(x_vec) {
    x_in <- as.data.frame(matrix(as.numeric(x_vec), nrow = 1))
    colnames(x_in) <- col_names
    
    tryCatch({
      post_obj  <- bart_machine_get_posterior(models$obj, new_data = x_in)
      obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
      mu        <- mean(obj_draws, na.rm = TRUE)
      sigma     <- sd(obj_draws,   na.rm = TRUE)
      if (is.na(mu))                     mu    <- 0
      if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
      
      p_feas <- if (is.null(models$feas)) {
        0.5
      } else {
        as.numeric(predict(models$feas, new_data = x_in, type = "prob"))
      }
      p_feas <- max(min(p_feas, 1), 0)
      if (is.na(p_feas)) p_feas <- 0.5
      
      post_viol  <- bart_machine_get_posterior(models$viol, new_data = x_in)
      viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
      pred_viol  <- max(0, mean(viol_draws, na.rm = TRUE))
      if (is.na(pred_viol)) pred_viol <- 0
      
      if (is.null(f_best_norm)) {
        acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * mu
      } else {
        z  <- (mu - f_best_norm) / sigma
        EI <- max((mu - f_best_norm) * pnorm(z) + sigma * dnorm(z), 0)
        acq <- EI * p_feas - current_pw * pred_viol
      }
      
      if (is.na(acq) || is.nan(acq)) -1e6 else acq
      
    }, error = function(e) -1e6)
  }
  
  # ---- BO loop --------------------------------------------------------------
  cat("Starting BO loop...\n\n")
  cat(sprintf("%-6s %-10s %-5s %-8s %-10s\n",
              "iter", "coverage", "feas", "viol", "best_cov"))
  cat(strrep("-", 48), "\n")
  
  for (iter in seq_len(n_iter)) {
    
    if (infeasible_streak > 3) current_pw <- min(current_pw * 1.5, 20)
    
    ga_res <- ga(
      type     = "binary",
      fitness  = acq_fn,
      nBits    = n_vars,
      popSize  = 100,
      maxiter  = 40,
      run      = 20,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = iter
    )
    x_new <- as.numeric(ga_res@solution[1, ])
    
    cov_new    <- mclp_coverage(x_new)
    feas_new   <- mclp_feasible(x_new)
    viol_new   <- mclp_violation(x_new)
    true_evals <- true_evals + 1
    
    if (feas_new == 0) {
      infeasible_streak <- infeasible_streak + 1
    } else {
      infeasible_streak <- 0
      current_pw        <- max(current_pw * 0.9, penalty_w)
    }
    
    X        <- rbind(X, x_new)
    X_df     <- as.data.frame(X); colnames(X_df) <- col_names
    y        <- c(y, cov_new)
    feas_bin <- c(feas_bin, feas_new)
    viols    <- c(viols, viol_new)
    
    fb_info     <- get_f_best(y, feas_bin)
    f_best      <- fb_info$value
    y_range     <- max(max(y) - min(y), 1e-8)
    f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
    
    # Build new row with metrics + x vector
    new_row <- data.frame(
      iteration       = iter,
      coverage        = cov_new,
      feasible        = feas_new,
      violation       = viol_new,
      best_cov_so_far = ifelse(is.null(f_best), NA, f_best),
      true_evals      = true_evals,
      as.data.frame(matrix(x_new, nrow = 1))
    )
    colnames(new_row)[(ncol(new_row) - n_vars + 1):ncol(new_row)] <- col_names
    history <- rbind(history, new_row)
    
    cat(sprintf("%-6d %-10.0f %-5d %-8.0f %-10s\n",
                iter, cov_new, feas_new, viol_new,
                ifelse(is.null(f_best), "NA", as.character(round(f_best)))))
    
    y_norm    <- normalize_01(y)
    viol_norm <- normalize_01(viols)
    models    <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
  }
  
  # ---- Save single CSV per run (history + all x vectors) -------------------
  write.csv(history,
            file.path(output_dir, sprintf("dart_run_%02d.csv", run)),
            row.names = FALSE)
  
  dart_time  <- proc.time()["elapsed"] - dart_t0
  best_cov   <- ifelse(is.null(f_best), NA_real_, f_best)
  cov_pct    <- 100 * best_cov / total_pop
  gap        <- glpk_obj - best_cov
  gap_pct    <- 100 * gap / glpk_obj
  feas_n     <- sum(history$feasible, na.rm = TRUE)
  feas_r     <- feas_n / nrow(history)
  
  # Best x recovered directly from X matrix via incumbent index
  best_x <- as.integer(X[fb_info$idx, ])
  
  cat(sprintf("\n>> Run %d done | Coverage: %.0f (%.2f%%) | Gap: %.0f (%.2f%%) | Evals: %d | Time: %.1f s\n",
              run, best_cov, cov_pct, gap, gap_pct, true_evals, dart_time))
  cat(sprintf("   Saved: dart_run_%02d.csv  (%d rows x %d cols)\n\n",
              run, nrow(history), ncol(history)))
  
  dart_summary_rows[[run]] <- data.frame(
    method         = "DART-CBO",
    run            = run,
    best_coverage  = best_cov,
    coverage_pct   = cov_pct,
    gap_from_glpk  = gap,
    gap_pct        = gap_pct,
    true_evals     = true_evals,
    time_seconds   = dart_time,
    feasible_evals = feas_n,
    feasible_rate  = feas_r
  )
}

dart_summary <- do.call(rbind, dart_summary_rows)

# =============================================================================
# 4. COMBINED SUMMARY CSVs
# =============================================================================
summary_all <- rbind(glpk_result_summary  <- data.frame(
  method         = "GLPK",
  run            = 1,
  best_coverage  = glpk_obj,
  coverage_pct   = glpk_pct,
  gap_from_glpk  = 0,
  gap_pct        = 0,
  true_evals     = NA,
  time_seconds   = glpk_time,
  feasible_evals = NA,
  feasible_rate  = NA
),
ga_result_summary,
dart_summary)

method_stats <- summary_all %>%
  group_by(method) %>%
  summarise(
    n_runs            = n(),
    mean_coverage     = mean(best_coverage,  na.rm = TRUE),
    sd_coverage       = sd(best_coverage,    na.rm = TRUE),
    mean_coverage_pct = mean(coverage_pct,   na.rm = TRUE),
    sd_coverage_pct   = sd(coverage_pct,     na.rm = TRUE),
    mean_gap_pct      = mean(gap_pct,        na.rm = TRUE),
    sd_gap_pct        = sd(gap_pct,          na.rm = TRUE),
    mean_true_evals   = mean(true_evals,     na.rm = TRUE),
    mean_time_s       = mean(time_seconds,   na.rm = TRUE),
    sd_time_s         = sd(time_seconds,     na.rm = TRUE),
    mean_feas_rate    = mean(feasible_rate,  na.rm = TRUE),
    .groups = "drop"
  )

write.csv(summary_all,
          file.path(output_dir, "summary_all_runs.csv"),
          row.names = FALSE)

write.csv(method_stats,
          file.path(output_dir, "summary_method_stats.csv"),
          row.names = FALSE)

# =============================================================================
# 5. CONSOLE REPORT
# =============================================================================
cat("\n", strrep("=", 65), "\n")
cat("FINAL COMPARISON REPORT\n")
cat(strrep("=", 65), "\n\n")

cat(sprintf("%-12s  %8s  %8s  %8s  %8s  %8s\n",
            "Method", "MeanCov%", "SD", "MeanGap%", "AvgEvals", "AvgTime(s)"))
cat(strrep("-", 65), "\n")

for (i in seq_len(nrow(method_stats))) {
  r <- method_stats[i, ]
  cat(sprintf("%-12s  %8.2f  %8.2f  %8.2f  %8.0f  %8.1f\n",
              r$method,
              r$mean_coverage_pct,
              r$sd_coverage_pct,
              r$mean_gap_pct,
              r$mean_true_evals,
              r$mean_time_s))
}

cat(sprintf("\nGLPK optimal (baseline): %.0f people (%.2f%%)\n", glpk_obj, glpk_pct))
cat(sprintf("\nOutput directory: %s/\n", output_dir))
cat("  glpk_result.csv            (1 row:  metrics + x1..x16)\n")
cat("  ga_result.csv              (all GA evals: history + x1..x16 per row)\n")
cat(sprintf("  dart_run_01.csv .. dart_run_%02d.csv\n", N_RUNS))
cat("                             (n_init+n_iter rows: history + x1..x16 per row)\n")
cat("  summary_all_runs.csv\n")
cat("  summary_method_stats.csv\n")