# =============================================================================
# MCLP — DART-CBO (bartMachine package)
#
# Problem:  Maximise population covered by at most max_facilities open sites.
#
# Acquisition function: EI for MAXIMISATION x P(feasible) - penalty x violation
#
#   EI+(x) = (mu - f_best+) * Phi(z) + sigma * phi(z)
#   where z = (mu - f_best+) / sigma
#   and f_best+ = best feasible coverage seen so far
#
#   No sign flip is used anywhere — the surrogate is trained on raw (positive)
#   coverage values and f_best+ is the running maximum.
#
# Feasibility model strategy (identical to LSCP companion script):
#   - p_feas = 0.5 until both classes (feasible / infeasible) are observed.
#   - Once both classes exist, fit a BART classifier on ALL data so far.
#   - Refit every BO iteration from that point forward.
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
n_init         <-  10    # initial random evaluations
n_iter         <- 250    # BO iterations
num_trees      <- 100    # BART trees
penalty_w      <-   5    # initial violation-penalty weight
max_facilities <-   4    # MCLP budget constraint
service_radius <- 5000

output_dir <- "results_MCLP_DART"
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

# Population vector — matched to row order of A
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
# MCLP HELPERS  (all in natural units — no sign flip)
# =============================================================================

# Objective: population covered (POSITIVE, to MAXIMISE)
mclp_coverage <- function(x) {
  coverage_vec <- pmin(as.vector(A %*% x), 1)
  sum(coverage_vec * population)
}

# Feasible iff facility count <= max_facilities
mclp_feasible <- function(x) as.integer(sum(x) <= max_facilities)

# Soft constraint violation = max(0, sum(x) - max_facilities)
mclp_violation <- function(x) max(0L, sum(x) - max_facilities)

normalize_01 <- function(x) {
  r <- range(x)
  if (diff(r) == 0) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

# Best feasible coverage so far (MAXIMUM over feasible evaluations)
get_f_best <- function(cov, feas) {
  idx <- which(as.logical(feas))
  if (!length(idx)) return(list(value = NULL, idx = NULL))
  bi <- idx[which.max(cov[idx])]
  list(value = cov[bi], idx = bi)
}

# =============================================================================
# GLPK — exact reference solution
# =============================================================================
cat("--- GLPK (reference) ---\n")

lp_model <- MIPModel() %>%
  add_variable(x[j], j = 1:n_vars,   type = "binary") %>%
  add_variable(z[i], i = 1:n_demand, type = "binary") %>%
  add_constraint(z[i] <= sum_expr(A[i, j] * x[j], j = 1:n_vars), i = 1:n_demand) %>%
  add_constraint(sum_expr(x[j], j = 1:n_vars) <= max_facilities) %>%
  set_objective(sum_expr(population[i] * z[i], i = 1:n_demand), sense = "max")

lp_res   <- solve_model(lp_model, with_ROI(solver = "glpk"))
glpk_obj <- lp_res$objective_value
cat(sprintf("GLPK optimal coverage: %.0f people (%.1f%%)\n\n",
            glpk_obj, 100 * glpk_obj / total_pop))

# =============================================================================
# INITIAL DESIGN — fully random
# =============================================================================
set.seed(42)
X        <- matrix(rbinom(n_init * n_vars, 1, 0.5), nrow = n_init)
X_df     <- as.data.frame(X); colnames(X_df) <- col_names

# All quantities stay positive / natural scale
y        <- apply(X, 1, mclp_coverage)   # population covered (positive)
feas_bin <- apply(X, 1, mclp_feasible)
viols    <- apply(X, 1, mclp_violation)

cat(sprintf("Initial design: %d feasible / %d\n", sum(feas_bin), n_init))
cat(sprintf("Coverage (pop): %s\n",  paste(round(y), collapse = ", ")))
cat(sprintf("Violations    : %s\n\n", paste(viols,   collapse = ", ")))

# =============================================================================
# FIT SURROGATE MODELS
# =============================================================================
fit_dart_models <- function(X_df, y_norm, feas_bin, viol_norm) {
  
  # Objective surrogate — trained on normalised coverage (higher = better)
  obj_model  <- bartMachine(X = X_df, y = y_norm,   num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  
  # Violation surrogate
  viol_model <- bartMachine(X = X_df, y = viol_norm, num_trees = num_trees,
                            alpha = 1, verbose = FALSE)
  
  # Feasibility classifier — only once both classes are observed
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

y_norm    <- normalize_01(y)
viol_norm <- normalize_01(viols)

cat("Fitting initial DART surrogates...\n")
models <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
if (is.null(models$feas))
  cat("  [feas model] both classes not yet observed — p_feas = 0.5\n")
cat("Surrogates fitted OK\n\n")

# =============================================================================
# ACQUISITION FUNCTION — EI for MAXIMISATION
#
# For a maximisation surrogate with posterior mean mu and std sigma:
#
#   z      = (mu - f_best_norm) / sigma
#   EI+(x) = (mu - f_best_norm) * Phi(z) + sigma * phi(z)
#
# This is the mirror image of the standard (minimisation) EI formula:
#   - Standard EI:  improvement = f_best - mu  (want mu LOW)
#   - EI+ here:     improvement = mu - f_best  (want mu HIGH)
#
# Full acquisition:
#   acq = EI+(x) * p_feas - penalty * pred_viol
#
# When no feasible point exists yet, exploration-only fallback:
#   acq = 10 * p_feas - penalty * pred_viol + 0.01 * mu
#
# f_best_norm, current_pw, and models are updated each iteration before
# acq_fn is called.
# =============================================================================
acq_fn <- function(x_vec) {
  x_in <- as.data.frame(matrix(as.numeric(x_vec), nrow = 1))
  colnames(x_in) <- col_names
  
  tryCatch({
    # ---- Objective posterior (normalised coverage scale) ----
    post_obj  <- bart_machine_get_posterior(models$obj, new_data = x_in)
    obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
    mu        <- mean(obj_draws, na.rm = TRUE)
    sigma     <- sd(obj_draws,   na.rm = TRUE)
    if (is.na(mu))                     mu    <- 0
    if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
    
    # ---- Feasibility probability ----
    p_feas <- if (is.null(models$feas)) {
      0.5
    } else {
      as.numeric(predict(models$feas, new_data = x_in, type = "prob"))
    }
    p_feas <- max(min(p_feas, 1), 0)
    if (is.na(p_feas)) p_feas <- 0.5
    
    # ---- Violation posterior ----
    post_viol  <- bart_machine_get_posterior(models$viol, new_data = x_in)
    viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
    pred_viol  <- max(0, mean(viol_draws, na.rm = TRUE))
    if (is.na(pred_viol)) pred_viol <- 0
    
    # ---- Acquisition value ----
    if (is.null(f_best_norm)) {
      # No feasible point yet — exploration fallback
      # 0.01 * mu softly steers toward high-coverage regions
      acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * mu
      
    } else {
      # EI for MAXIMISATION
      # f_best_norm and mu are both on the normalised [0, 1] coverage scale
      z  <- (mu - f_best_norm) / sigma
      EI <- max((mu - f_best_norm) * pnorm(z) + sigma * dnorm(z), 0)
      acq <- EI * p_feas - current_pw * pred_viol
    }
    
    if (is.na(acq) || is.nan(acq)) -1e6 else acq
    
  }, error = function(e) -1e6)
}

# =============================================================================
# BO LOOP
# =============================================================================
fb_info     <- get_f_best(y, feas_bin)
f_best      <- fb_info$value                # best coverage seen (raw scale)
y_range     <- max(max(y) - min(y), 1e-8)
f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL

current_pw        <- penalty_w
infeasible_streak <- 0
true_evals        <- n_init

history <- data.frame(
  iteration       = rep(0, n_init),
  coverage        = y,
  feasible        = feas_bin,
  violation       = viols,
  best_cov_so_far = ifelse(is.null(f_best), NA, f_best),
  true_evals      = seq_len(n_init)
)

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
    popSize  = 50,      # full run: 200
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
  
  history <- rbind(history, data.frame(
    iteration       = iter,
    coverage        = cov_new,
    feasible        = feas_new,
    violation       = viol_new,
    best_cov_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals      = true_evals
  ))
  
  cat(sprintf("%-6d %-10.0f %-5d %-8.0f %-10s\n",
              iter, cov_new, feas_new, viol_new,
              ifelse(is.null(f_best), "NA", as.character(round(f_best)))))
  
  # Refit all surrogates on full accumulated data
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_dart_models(X_df, y_norm, feas_bin, viol_norm)
}

# =============================================================================
# RESULTS
# =============================================================================
cat("\n", strrep("=", 50), "\n")
cat("DART-CBO MCLP RESULTS\n")
cat(strrep("=", 50), "\n")

cat(sprintf("GLPK optimal coverage : %.0f people (%.1f%%)\n",
            glpk_obj, 100 * glpk_obj / total_pop))
cat(sprintf("DART-CBO best coverage: %s\n",
            ifelse(is.null(f_best),
                   "NA (no feasible found)",
                   sprintf("%.0f people (%.1f%%)", f_best, 100 * f_best / total_pop))))
if (!is.null(f_best))
  cat(sprintf("Gap from optimal      : %.0f people (%.1f%%)\n",
              glpk_obj - f_best,
              100 * (glpk_obj - f_best) / glpk_obj))

cat(sprintf("Feasible evals        : %d / %d (%.0f%%)\n",
            sum(history$feasible, na.rm = TRUE),
            nrow(history),
            100 * mean(history$feasible, na.rm = TRUE)))
cat(sprintf("Total true evals      : %d\n", true_evals))

write.csv(history, file.path(output_dir, "dart_mclp_history.csv"), row.names = FALSE)
cat(sprintf("\nHistory saved to %s/dart_mclp_history.csv\n", output_dir))