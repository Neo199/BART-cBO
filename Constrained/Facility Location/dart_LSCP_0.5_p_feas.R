# =============================================================================
# LSCP — DART-CBO (dartMachine package) — standalone test script
# Pure black-box: initial design is fully random, no feasible seed
#
# Feasibility model strategy:
#   - p_feas = 0.5 (max uncertainty) until the first feasible point is found
#   - Once both classes exist, fit classifier on ALL data accumulated so far
#   - Refit every BO iteration from that point forward
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
n_init    <-  20    # full run: 10
n_iter    <- 250    # full run: 250
num_trees <- 100    # full run: 100
penalty_w <-  2

service_radius <- 5000
output_dir     <- "BART-CBO/Constrained/dartMachine/Facility Location/results_LSCP_DART"
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
# GLPK — reference only, not used in the BO search
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
# INITIAL DESIGN — fully random
# =============================================================================
set.seed(42)
X        <- matrix(rbinom(n_init * n_vars, 1, 0.5), nrow = n_init)
X_df     <- as.data.frame(X); colnames(X_df) <- col_names
y        <- apply(X, 1, lscp_obj)
feas_bin <- as.integer(apply(X, 1, lscp_feasible))
viols    <- apply(X, 1, lscp_violation)

cat(sprintf("Initial design: %d feasible / %d\n", sum(feas_bin), n_init))
cat(sprintf("Objectives    : %s\n", paste(y, collapse = ", ")))
cat(sprintf("Violations    : %s\n\n", paste(viols, collapse = ", ")))

# =============================================================================
# FIT SURROGATE MODELS
# feas_model is NULL until both classes have been observed.
# obj and viol models are always fitted — they don't need two classes.
# =============================================================================
fit_dart_models <- function(X_df, y_norm, feas_bin, viol_norm) {
  
  obj_model  <- bartMachine(X = X_df, y = y_norm,   num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  viol_model <- bartMachine(X = X_df, y = viol_norm, num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  
  # Only fit once both 0 and 1 have been seen
  feas_model <- if (length(unique(feas_bin)) == 2) {
    cat("  [feas model] fitting classifier on all data so far\n")
    bartMachine(X = X_df,
                y = factor(feas_bin, levels = c("0", "1")),
                num_trees = num_trees, alpha = 1, verbose = FALSE)
  } else {
    NULL   # not yet — acq_fn will use p_feas = 0.5
  }
  
  list(obj = obj_model, feas = feas_model, viol = viol_model)
}

y_norm    <- normalize_01(y)
viol_norm <- normalize_01(viols)

cat("Fitting initial DART surrogates...\n")
models <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
if (is.null(models$feas))
  cat("  [feas model] no feasible points yet — p_feas = 0.5 until first feasible found\n")
cat("Surrogates fitted OK\n\n")

# =============================================================================
# ACQUISITION FUNCTION
# f_best_norm, current_pw, and models are read from the enclosing environment
# and updated each iteration before acq_fn is called.
# =============================================================================
acq_fn <- function(x_vec) {
  x_in <- as.data.frame(matrix(as.numeric(x_vec), nrow = 1))
  colnames(x_in) <- col_names
  
  tryCatch({
    # Objective posterior
    post_obj  <- bart_machine_get_posterior(models$obj, new_data = x_in)
    obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
    mu        <- mean(obj_draws, na.rm = TRUE)
    sigma     <- sd(obj_draws,   na.rm = TRUE)
    if (is.na(mu))                     mu    <- 0
    if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
    
    # Feasibility probability:
    #   - NULL model  => p_feas = 0.5 (max uncertainty, no feasible seen yet)
    #   - fitted model => use posterior prediction from classifier
    p_feas <- if (is.null(models$feas)) {
      0.5
    } else {
      as.numeric(predict(models$feas, new_data = x_in, type = "prob"))
    }
    p_feas <- max(min(p_feas, 1), 0)
    if (is.na(p_feas)) p_feas <- 0.5
    
    # Violation posterior
    post_viol  <- bart_machine_get_posterior(models$viol, new_data = x_in)
    viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
    pred_viol  <- max(0, mean(viol_draws, na.rm = TRUE))
    if (is.na(pred_viol)) pred_viol <- 0
    
    # No feasible point seen yet: explore toward feasible + low cost
    if (is.null(f_best_norm)) {
      acq <- p_feas - current_pw * pred_viol + 0.01 * (-mu)
      
      # Feasible point exists: standard EI x P(feasible) - penalty x violation
    } else {
      z   <- (f_best_norm - mu) / sigma
      EI  <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
      acq <- EI * p_feas - current_pw * pred_viol
    }
    
    if (is.na(acq) || is.nan(acq)) -1e6 else acq
    
  }, error = function(e) -1e6)
}

# =============================================================================
# BO LOOP
# =============================================================================
fb_info     <- get_f_best(y, feas_bin)
f_best      <- fb_info$value
y_range     <- max(max(y) - min(y), 1e-8)
f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL

current_pw        <- penalty_w
infeasible_streak <- 0
true_evals        <- n_init

history <- data.frame(
  iteration     = rep(0, n_init),
  y             = y,
  feasible      = feas_bin,
  violation     = viols,
  f_best_so_far = ifelse(is.null(f_best), NA, f_best),
  true_evals    = seq_len(n_init)
)

cat("Starting BO loop...\n\n")
cat(sprintf("%-6s %-6s %-5s %-8s %-6s\n", "iter", "y", "feas", "viol", "best"))
cat(strrep("-", 38), "\n")

for (iter in seq_len(n_iter)) {
  
  if (infeasible_streak > 3) current_pw <- min(current_pw * 1.5, 20)
  
  ga_res <- ga(
    type     = "binary",
    fitness  = acq_fn,
    nBits    = n_vars,
    popSize  = 100,    # full run: 200
    maxiter  = 40,    # full run: 40
    run      = 20,     # full run: 20
    keepBest = TRUE,
    monitor  = FALSE,
    seed     = iter
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
    current_pw        <- max(current_pw * 0.9, penalty_w)
  }
  
  X        <- rbind(X, x_new)
  X_df     <- as.data.frame(X); colnames(X_df) <- col_names
  y        <- c(y, y_new)
  feas_bin <- c(feas_bin, feas_new)
  viols    <- c(viols, viol_new)
  
  fb_info     <- get_f_best(y, feas_bin)
  f_best      <- fb_info$value
  y_range     <- max(max(y) - min(y), 1e-8)
  f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
  
  history <- rbind(history, data.frame(
    iteration     = iter,
    y             = y_new,
    feasible      = feas_new,
    violation     = viol_new,
    f_best_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals    = true_evals
  ))
  
  cat(sprintf("%-6d %-6d %-5d %-8.0f %-6s\n",
              iter, y_new, feas_new, viol_new,
              ifelse(is.null(f_best), "NA", as.character(f_best))))
  
  # Refit all surrogates on full accumulated data
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
}

# =============================================================================
# RESULTS
# =============================================================================
cat("\n", strrep("=", 45), "\n")
cat("DART-CBO RESULTS\n")
cat(strrep("=", 45), "\n")
cat(sprintf("GLPK optimal    : %d facilities\n", glpk_obj))
cat(sprintf("DART-CBO best   : %s facilities\n",
            ifelse(is.null(f_best), "NA (no feasible found)", as.character(f_best))))
if (!is.null(f_best))
  cat(sprintf("Gap to optimal  : %d\n", f_best - glpk_obj))
cat(sprintf("Feasible evals  : %d / %d (%.0f%%)\n",
            sum(history$feasible, na.rm = TRUE),
            nrow(history),
            100 * mean(history$feasible, na.rm = TRUE)))
cat(sprintf("Total true evals: %d\n", true_evals))

write.csv(history, file.path(output_dir, "dart_history.csv"), row.names = FALSE)
cat(sprintf("\nHistory saved to %s/dart_history.csv\n", output_dir))

# X contains all evaluated solutions (n_init + n_iter rows)
# history has the matching metrics row by row

x_df_full <- as.data.frame(X)
colnames(x_df_full) <- col_names  # already defined in your session

# Combine with history
history_with_x <- cbind(history, x_df_full)

write.csv(history_with_x, 
          file.path(output_dir, "dart_history_1feasini_with_x.csv"), 
          row.names = FALSE)
cat("Saved history with x vectors.\n")
