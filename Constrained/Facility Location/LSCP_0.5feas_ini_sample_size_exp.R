# =============================================================================
# LSCP — DART-CBO (dartMachine package) — Initial Sample Size Experiment
# Tests n_init = 5, 10, 20 with 10 instances each
# Saves full results (including x vectors) as .rds per instance
#
# Output structure:
#   LSCP_0.5feas_ini_exp/ini_sample_set/
#     ini5/   instance_01.rds ... instance_10.rds
#     ini10/  instance_01.rds ... instance_10.rds
#     ini20/  instance_01.rds ... instance_10.rds
# =============================================================================

options(java.parameters = "-Xmx8g")
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
# FIXED SETTINGS
# =============================================================================
n_iter    <- 250
num_trees <- 100
penalty_w <-  2

service_radius <- 5000

# Experiment grid
ini_values  <- c(5, 10, 20)
n_instances <- 10

base_output_dir <- "LSCP_0.5feas_ini_exp/ini_sample_set"

# =============================================================================
# DATA + COVERAGE MATRIX  (loaded once)
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

cat(sprintf("Problem: %d demand points, %d candidate facilities\n\n", n_demand, n_vars))

# =============================================================================
# LSCP HELPERS
# =============================================================================
lscp_obj       <- function(x) sum(x)
lscp_feasible  <- function(x) all(as.vector(A %*% x) >= 1)
lscp_violation <- function(x) sum(pmax(0, 1 - as.vector(A %*% x)))

normalize_01 <- function(x) {
  r <- range(x)
  if (diff(r) == 0) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

get_f_best <- function(y, feas) {
  idx <- which(as.logical(feas))
  if (!length(idx)) return(list(value = NULL, idx = NULL))
  bi <- idx[which.min(y[idx])]
  list(value = y[bi], idx = bi)
}

# =============================================================================
# GLPK — reference optimal (computed once)
# =============================================================================
cat("--- GLPK (reference) ---\n")
lp_model <- MIPModel() %>%
  add_variable(x[j], j = 1:n_vars, type = "binary") %>%
  add_constraint(sum_expr(A[i, j] * x[j], j = 1:n_vars) >= 1, i = 1:n_demand) %>%
  set_objective(sum_expr(x[j], j = 1:n_vars), sense = "min")
lp_res   <- solve_model(lp_model, with_ROI(solver = "glpk"))
glpk_obj <- lp_res$objective_value
cat(sprintf("GLPK optimal: %d facilities\n\n", glpk_obj))

# =============================================================================
# SURROGATE FITTING
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
# ACQUISITION FUNCTION FACTORY
# Returns a closure that captures the current environment state
# =============================================================================
make_acq_fn <- function(models_env) {
  function(x_vec) {
    x_in <- as.data.frame(matrix(as.numeric(x_vec), nrow = 1))
    colnames(x_in) <- col_names
    
    tryCatch({
      post_obj  <- bart_machine_get_posterior(models_env$models$obj, new_data = x_in)
      obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
      mu        <- mean(obj_draws, na.rm = TRUE)
      sigma     <- sd(obj_draws,   na.rm = TRUE)
      if (is.na(mu))                     mu    <- 0
      if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
      
      p_feas <- if (is.null(models_env$models$feas)) {
        0.5
      } else {
        as.numeric(predict(models_env$models$feas, new_data = x_in, type = "prob"))
      }
      p_feas <- max(min(p_feas, 1), 0)
      if (is.na(p_feas)) p_feas <- 0.5
      
      post_viol  <- bart_machine_get_posterior(models_env$models$viol, new_data = x_in)
      viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
      pred_viol  <- max(0, mean(viol_draws, na.rm = TRUE))
      if (is.na(pred_viol)) pred_viol <- 0
      
      f_best_norm <- models_env$f_best_norm
      current_pw  <- models_env$current_pw
      
      if (is.null(f_best_norm)) {
        acq <- p_feas - current_pw * pred_viol + 0.01 * (-mu)
      } else {
        z   <- (f_best_norm - mu) / sigma
        EI  <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
        acq <- EI * p_feas - current_pw * pred_viol
      }
      
      if (is.na(acq) || is.nan(acq)) -1e6 else acq
      
    }, error = function(e) -1e6)
  }
}

# =============================================================================
# SINGLE RUN FUNCTION
# Returns a list with history (data.frame) and X_all (matrix of all solutions)
# =============================================================================
run_single_instance <- function(n_init, instance_seed) {
  
  cat(sprintf("\n  >> n_init=%d | instance seed=%d\n", n_init, instance_seed))
  
  # ---- Initial design -------------------------------------------------------
  set.seed(instance_seed)
  X        <- matrix(rbinom(n_init * n_vars, 1, 0.5), nrow = n_init)
  X_df     <- as.data.frame(X); colnames(X_df) <- col_names
  y        <- apply(X, 1, lscp_obj)
  feas_bin <- as.integer(apply(X, 1, lscp_feasible))
  viols    <- apply(X, 1, lscp_violation)
  
  cat(sprintf("     Initial: %d feasible / %d\n", sum(feas_bin), n_init))
  
  # ---- Initial surrogates ---------------------------------------------------
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
  if (is.null(models$feas))
    cat("     [feas model] p_feas = 0.5 until first feasible found\n")
  
  # ---- Shared mutable environment for acquisition function ------------------
  models_env <- new.env(parent = emptyenv())
  models_env$models     <- models
  
  fb_info     <- get_f_best(y, feas_bin)
  f_best      <- fb_info$value
  y_range     <- max(max(y) - min(y), 1e-8)
  models_env$f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
  models_env$current_pw  <- penalty_w
  
  acq_fn <- make_acq_fn(models_env)
  
  infeasible_streak <- 0
  true_evals        <- n_init
  
  # ---- History (includes x vectors) ----------------------------------------
  history <- data.frame(
    iteration     = rep(0, n_init),
    y             = y,
    feasible      = feas_bin,
    violation     = viols,
    f_best_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals    = seq_len(n_init),
    as.data.frame(X)
  )
  colnames(history)[(ncol(history) - n_vars + 1):ncol(history)] <- col_names
  
  cat(sprintf("     %-6s %-6s %-5s %-8s %-6s\n", "iter", "y", "feas", "viol", "best"))
  cat(sprintf("     %s\n", strrep("-", 36)))
  
  # ---- BO loop --------------------------------------------------------------
  for (iter in seq_len(n_iter)) {
    
    if (infeasible_streak > 3)
      models_env$current_pw <- min(models_env$current_pw * 1.5, 20)
    
    ga_res <- ga(
      type     = "binary",
      fitness  = acq_fn,
      nBits    = n_vars,
      popSize  = 100,
      maxiter  = 40,
      run      = 20,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = iter * 1000 + instance_seed   # unique per (iter, instance)
    )
    x_new <- as.numeric(ga_res@solution[1, ])
    
    y_new      <- lscp_obj(x_new)
    feas_new   <- as.integer(lscp_feasible(x_new))
    viol_new   <- lscp_violation(x_new)
    true_evals <- true_evals + 1
    
    if (feas_new == 0) {
      infeasible_streak <- infeasible_streak + 1
    } else {
      infeasible_streak <- 0
      models_env$current_pw <- max(models_env$current_pw * 0.9, penalty_w)
    }
    
    X        <- rbind(X, x_new)
    X_df     <- as.data.frame(X); colnames(X_df) <- col_names
    y        <- c(y, y_new)
    feas_bin <- c(feas_bin, feas_new)
    viols    <- c(viols, viol_new)
    
    fb_info     <- get_f_best(y, feas_bin)
    f_best      <- fb_info$value
    y_range     <- max(max(y) - min(y), 1e-8)
    models_env$f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
    
    # Build history row with x vector
    new_row <- data.frame(
      iteration     = iter,
      y             = y_new,
      feasible      = feas_new,
      violation     = viol_new,
      f_best_so_far = ifelse(is.null(f_best), NA, f_best),
      true_evals    = true_evals,
      as.data.frame(matrix(x_new, nrow = 1))
    )
    colnames(new_row)[(ncol(new_row) - n_vars + 1):ncol(new_row)] <- col_names
    history <- rbind(history, new_row)
    
    cat(sprintf("     %-6d %-6d %-5d %-8.0f %-6s\n",
                iter, y_new, feas_new, viol_new,
                ifelse(is.null(f_best), "NA", as.character(f_best))))
    
    # Refit surrogates on all accumulated data
    y_norm    <- normalize_01(y)
    viol_norm <- normalize_01(viols)
    models_env$models <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
  }
  
  # ---- Summary --------------------------------------------------------------
  cat(sprintf("\n     GLPK optimal : %d\n", glpk_obj))
  cat(sprintf("     DART-CBO best: %s\n",
              ifelse(is.null(f_best), "NA", as.character(f_best))))
  cat(sprintf("     Feasible     : %d / %d (%.0f%%)\n",
              sum(history$feasible, na.rm = TRUE),
              nrow(history),
              100 * mean(history$feasible, na.rm = TRUE)))
  
  list(
    n_init       = n_init,
    instance_seed= instance_seed,
    glpk_optimal = glpk_obj,
    f_best       = f_best,
    history      = history,       # data.frame: metrics + x1..x16 per row
    X_all        = X              # raw matrix of all evaluated solutions
  )
}

# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("STARTING EXPERIMENT\n")
cat(sprintf("  n_init values : %s\n", paste(ini_values, collapse = ", ")))
cat(sprintf("  instances each: %d\n", n_instances))
cat(sprintf("  n_iter        : %d\n", n_iter))
cat(strrep("=", 60), "\n\n")

# Seeds: instance i gets seed (ini_idx * 100 + i) so seeds never clash across groups
# e.g. ini5: 101-110 | ini10: 201-210 | ini20: 301-310

for (ini_idx in seq_along(ini_values)) {
  
  n_init   <- ini_values[ini_idx]
  out_dir  <- file.path(base_output_dir, paste0("ini", n_init))
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  cat(sprintf("\n%s\n n_init = %d  -->  %s\n%s\n",
              strrep("-", 55), n_init, out_dir, strrep("-", 55)))
  
  for (inst in seq_len(n_instances)) {
    
    inst_seed <- ini_idx * 100 + inst   # e.g. 101, 102 ... 110 for ini5
    
    result <- run_single_instance(n_init = n_init, instance_seed = inst_seed)
    
    out_file <- file.path(out_dir, sprintf("instance_%02d.rds", inst))
    saveRDS(result, file = out_file)
    cat(sprintf("  Saved: %s\n", out_file))
  }
  
  cat(sprintf("\n  Completed all %d instances for n_init=%d\n", n_instances, n_init))
}

cat("\n", strrep("=", 60), "\n")
cat("EXPERIMENT COMPLETE\n")
cat(sprintf("Results saved under: %s\n", base_output_dir))
cat(strrep("=", 60), "\n")