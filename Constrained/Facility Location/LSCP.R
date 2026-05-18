# =============================================================================
# Constrained Bayesian Optimisation for LSCP
# Comparing: BART-CBO, DART-CBO, GA, GLPK
#
# BART  via BART package  : wbart (regression), pbart (classification)
#   - posterior prediction on new data uses pwbart() / predict.pbart()
# DART  via dartMachine   : bart_machine_get_posterior / predict(..., type="prob")
#
# Author: adapted from Niyati Seth's LSCP + knapsack BART-CBO code
# Date  : December 2025
# =============================================================================

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

# =============================================================================
# LIBRARIES
# =============================================================================
library(tidyr)
library(dplyr)
library(ggplot2)
library(BART)          # wbart, pbart, pwbart, predict.pbart
library(dartMachine)   # dartMachine, bart_machine_get_posterior
library(GA)
library(ompr)
library(ompr.roi)
library(ROI)
library(ROI.plugin.glpk)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
n_iter          <- 250
n_init          <- 10
n_instances     <- 10
service_radius  <- 5000
base_output_dir <- "results_LSCP_comparison"

if (!dir.exists(base_output_dir)) dir.create(base_output_dir, recursive = TRUE)

# =============================================================================
# LOAD DATA
# =============================================================================
demand_data     <- read.csv("PhD-Data/SF_data/SF_demand_205_centroid_uniform_weight.csv")
facility_loc    <- read.csv("PhD-Data/SF_data/SF_store_site_16_longlat.csv")
distance_matrix <- read.csv("PhD-Data/SF_data/SF_network_distance_candidateStore_16_censusTract_205_new.csv")

# =============================================================================
# COVERAGE MATRIX
# =============================================================================
distance_matrix$covered <- as.integer(distance_matrix$distance <= service_radius)

A_df <- distance_matrix %>%
  dplyr::select(DestinationName, name, covered) %>%
  tidyr::pivot_wider(
    names_from  = name,
    values_from = covered,
    values_fill = list(covered = 0)
  )

A        <- as.matrix(A_df[, -1])
n_demand <- nrow(A)
n_vars   <- ncol(A)

cat(sprintf("Problem: %d demand points, %d candidate facilities\n", n_demand, n_vars))

# =============================================================================
# LSCP HELPERS
# =============================================================================
lscp_obj <- function(x) sum(x)

lscp_feasible <- function(x) all(as.vector(A %*% x) >= 1)

lscp_violation <- function(x) sum(pmax(0, 1 - as.vector(A %*% x)))

lscp_prob <- function(x_mat) apply(x_mat, 1, lscp_obj)

# =============================================================================
# GENERAL UTILITIES
# =============================================================================
normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

get_f_best <- function(y_vec, feas_vec) {
  # MINIMISATION: smallest feasible objective value
  feas_idx <- which(as.logical(feas_vec))
  if (length(feas_idx) == 0) return(list(value = NULL, idx = NULL))
  best_idx <- feas_idx[which.min(y_vec[feas_idx])]
  list(value = y_vec[best_idx], idx = best_idx)
}

random_binary <- function(n, p) matrix(rbinom(n * p, 1, 0.5), nrow = n, ncol = p)

# Inner GA to maximise an acquisition function over binary space
maximise_acq_ga <- function(acq_fn, n_bits, seed_val) {
  ga(
    type     = "binary",
    fitness  = acq_fn,
    nBits    = n_bits,
    popSize  = 200,
    maxiter  = 40,
    run      = 20,
    keepBest = TRUE,
    monitor  = FALSE,
    seed     = seed_val
  )@solution[1, ]
}

# =============================================================================
# GLPK EXACT SOLVER
# =============================================================================
run_glpk <- function() {
  cat("Running GLPK exact solver...\n")
  t0 <- Sys.time()
  
  model_lp <- MIPModel() %>%
    add_variable(x[j], j = 1:n_vars, type = "binary") %>%
    add_constraint(sum_expr(A[i, j] * x[j], j = 1:n_vars) >= 1, i = 1:nrow(A)) %>%
    set_objective(sum_expr(x[j], j = 1:n_vars), sense = "min")
  
  result <- solve_model(model_lp, with_ROI(solver = "glpk"))
  t1     <- Sys.time()
  
  list(
    objective    = result$objective_value,
    solution     = as.numeric(result$solution),
    time_seconds = as.numeric(difftime(t1, t0, units = "secs"))
  )
}

# =============================================================================
# GA BENCHMARK
# =============================================================================
run_ga_lscp <- function(seed_val) {
  set.seed(seed_val)
  eval_count <- 0L
  
  fitness_ga <- function(x) {
    eval_count <<- eval_count + 1L
    x <- round(x)
    penalty <- sum(pmax(0, 1 - as.vector(A %*% x))) * 1000
    -(lscp_obj(x) + penalty)      # GA maximises, LSCP minimises
  }
  
  t0 <- Sys.time()
  ga_res <- ga(type = "binary", fitness = fitness_ga,
               nBits = n_vars, run = 100, monitor = FALSE)
  t1 <- Sys.time()
  
  best_x <- round(ga_res@solution[1, ])
  list(
    x_best          = best_x,
    f_best          = lscp_obj(best_x),
    feasible        = lscp_feasible(best_x),
    true_eval_count = eval_count,
    time_seconds    = as.numeric(difftime(t1, t0, units = "secs"))
  )
}

# =============================================================================
# BART-CBO
# -----------------------------------------------------------------------------
# BART package prediction API
#
#   Regression (wbart):
#     - Fit  : wbart(x.train, y.train, ...)
#     - Predict on new data: pwbart(x.test, treedraws, mu)
#         x.test     : numeric matrix [n_new x p]
#         treedraws  : model$treedraws
#         mu         : model$mu  (training-mean offset stored by wbart)
#       Returns matrix [ndpost x n_new]; rows = posterior draws
#
#   Classification (pbart):
#     - Fit  : pbart(x.train, y.train, ...)   y in {0,1} or {-1,1}
#     - Predict on new data: predict(pbart_obj, newdata = x.test)
#       Returns list with $prob.test [ndpost x n_new]
#       Column means = posterior P(y=1 | x)
# =============================================================================

build_bart_models <- function(X_mat, y_norm, feas_bin, viol_norm) {
  
  bart_obj <- wbart(
    x.train    = X_mat,
    y.train    = y_norm,
    ndpost     = 200,
    nskip      = 100,
    printevery = 1e6
  )
  
  bart_feas <- pbart(
    x.train    = X_mat,
    y.train    = feas_bin,   # integer 0/1; pbart recodes internally
    ndpost     = 200,
    nskip      = 100,
    printevery = 1e6
  )
  
  bart_viol <- wbart(
    x.train    = X_mat,
    y.train    = viol_norm,
    ndpost     = 200,
    nskip      = 100,
    printevery = 1e6
  )
  
  list(obj = bart_obj, feas = bart_feas, viol = bart_viol)
}

# Regression posterior draws for a single new point via pwbart
bart_reg_draws <- function(model, x_new) {
  x_mat <- matrix(as.numeric(x_new), nrow = 1)
  # pwbart returns [ndpost x 1]; drop to vector
  as.numeric(pwbart(x_mat, model$treedraws, mu = model$mu)[, 1])
}

# Feasibility probability posterior draws via predict.pbart
bart_feas_draws <- function(model, x_new) {
  x_mat <- matrix(as.numeric(x_new), nrow = 1)
  pred  <- predict(model, newdata = x_mat)
  # prob.test is [ndpost x 1]; drop to vector of P(feasible) draws
  as.numeric(pred$prob.test[, 1])
}

bart_acq_fn <- function(x_vec, models, f_best_norm, penalty_weight) {
  
  # ── Objective posterior ─────────────────────────────────────────────────
  obj_draws <- bart_reg_draws(models$obj, x_vec)
  mu        <- mean(obj_draws)
  sigma     <- sd(obj_draws)
  if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
  
  # EI for MINIMISATION
  EI <- if (is.null(f_best_norm)) {
    -mu
  } else {
    z  <- (f_best_norm - mu) / sigma
    ei <- (f_best_norm - mu) * pnorm(z) + sigma * dnorm(z)
    max(ei, 0)
  }
  
  # ── Feasibility probability ─────────────────────────────────────────────
  p_feas <- mean(bart_feas_draws(models$feas, x_vec))
  p_feas <- max(min(p_feas, 1), 0)
  if (is.na(p_feas)) p_feas <- 0.5
  
  # ── Violation posterior ─────────────────────────────────────────────────
  viol_draws <- bart_reg_draws(models$viol, x_vec)
  pred_viol  <- max(0, mean(viol_draws))
  if (is.na(pred_viol)) pred_viol <- 0
  
  # ── Combined acquisition ────────────────────────────────────────────────
  acq <- if (is.null(f_best_norm)) {
    10 * p_feas - penalty_weight * pred_viol + 0.01 * (-mu)
  } else {
    EI * p_feas - penalty_weight * pred_viol
  }
  
  if (is.na(acq) || is.nan(acq)) return(-1e6)
  acq
}

run_bart_cbo <- function(seed_val, penalty_weight = 5, verbose = TRUE) {
  set.seed(seed_val)
  
  X          <- random_binary(n_init, n_vars)
  y          <- lscp_prob(X)
  feas_bin   <- as.integer(apply(X, 1, lscp_feasible))
  viols      <- apply(X, 1, lscp_violation)
  true_evals <- n_init
  
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  
  f_best_info <- get_f_best(y, feas_bin)
  f_best      <- f_best_info$value
  
  history_df <- data.frame(
    iteration     = rep(0, n_init),
    y             = y,
    feasible      = feas_bin,
    violation     = viols,
    f_best_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals    = seq_len(n_init)
  )
  
  current_pw        <- penalty_weight
  infeasible_streak <- 0
  models            <- build_bart_models(X, y_norm, feas_bin, viol_norm)
  
  for (iter in seq_len(n_iter)) {
    
    if (infeasible_streak > 3) current_pw <- min(current_pw * 1.5, 20)
    
    y_range         <- max(max(y) - min(y), 1e-8)
    f_best_norm_cur <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
    
    acq    <- function(xv) bart_acq_fn(xv, models, f_best_norm_cur, current_pw)
    x_next <- as.numeric(maximise_acq_ga(acq, n_vars,
                                         sample.int(.Machine$integer.max, 1)))
    
    y_next     <- lscp_obj(x_next)
    feas_next  <- as.integer(lscp_feasible(x_next))
    viol_next  <- lscp_violation(x_next)
    true_evals <- true_evals + 1
    
    if (feas_next == 0) {
      infeasible_streak <- infeasible_streak + 1
    } else {
      infeasible_streak <- 0
      current_pw        <- max(current_pw * 0.9, penalty_weight)
    }
    
    X        <- rbind(X, x_next)
    y        <- c(y, y_next)
    feas_bin <- c(feas_bin, feas_next)
    viols    <- c(viols, viol_next)
    
    y_norm    <- normalize_01(y)
    viol_norm <- normalize_01(viols)
    
    f_best_info <- get_f_best(y, feas_bin)
    f_best      <- f_best_info$value
    
    history_df <- rbind(history_df, data.frame(
      iteration     = iter,
      y             = y_next,
      feasible      = feas_next,
      violation     = viol_next,
      f_best_so_far = ifelse(is.null(f_best), NA, f_best),
      true_evals    = true_evals
    ))
    
    models <- build_bart_models(X, y_norm, feas_bin, viol_norm)
    
    if (verbose && iter %% 25 == 0) {
      cat(sprintf("  [BART] Iter %3d | evals: %d | feas: %d/%d | best: %s\n",
                  iter, true_evals, sum(feas_bin), length(feas_bin),
                  ifelse(is.null(f_best), "NA", f_best)))
    }
  }
  
  list(
    history          = history_df,
    X                = X,
    y                = y,
    feas             = feas_bin,
    f_best           = f_best,
    x_best           = if (!is.null(f_best_info$idx)) X[f_best_info$idx, ] else NULL,
    n_feasible       = sum(feas_bin),
    n_total          = length(feas_bin),
    feasibility_rate = mean(feas_bin),
    true_eval_count  = true_evals
  )
}

# =============================================================================
# DART-CBO
# -----------------------------------------------------------------------------
# dartMachine prediction API (mirrors bartMachine):
#
#   Regression:
#     bart_machine_get_posterior(model, new_data)
#       new_data : data.frame with same column names as training X
#       Returns  : list with $y_hat_posterior_samples [n_new x ndpost]
#                  so for a single row: [1 x ndpost] -> drop to vector
#
#   Classification:
#     predict(model, new_data, type = "prob")
#       Returns scalar P(y = 1 | x) for a single-row new_data
# =============================================================================

build_dart_models <- function(X_df, y_norm, feas_factor, viol_norm) {
  dart_obj  <- bartMachine(X = X_df, y = y_norm,     num_trees = 100, verbose = FALSE)
  dart_feas <- bartMachine(X = X_df, y = feas_factor, num_trees = 100, verbose = FALSE)
  dart_viol <- bartMachine(X = X_df, y = viol_norm,  num_trees = 100, verbose = FALSE)
  list(obj = dart_obj, feas = dart_feas, viol = dart_viol)
}

dart_acq_fn <- function(x_vec) {
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
      acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * (-mu)
      
      # Feasible point exists: standard EI x P(feasible) - penalty x violation
    } else {
      z   <- (f_best_norm - mu) / sigma
      EI  <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
      acq <- EI * p_feas - current_pw * pred_viol
    }
    
    if (is.na(acq) || is.nan(acq)) -1e6 else acq
    
  }, error = function(e) -1e6)
}

run_dart_cbo <- function(seed_val, penalty_weight = 5, verbose = TRUE) {
  set.seed(seed_val)
  col_names <- paste0("x", seq_len(n_vars))
  
  X          <- random_binary(n_init, n_vars)
  X_df       <- as.data.frame(X); colnames(X_df) <- col_names
  y          <- lscp_prob(X)
  feas_bin   <- as.integer(apply(X, 1, lscp_feasible))
  viols      <- apply(X, 1, lscp_violation)
  true_evals <- n_init
  
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  
  f_best_info <- get_f_best(y, feas_bin)
  f_best      <- f_best_info$value
  
  history_df <- data.frame(
    iteration     = rep(0, n_init),
    y             = y,
    feasible      = feas_bin,
    violation     = viols,
    f_best_so_far = ifelse(is.null(f_best), NA, f_best),
    true_evals    = seq_len(n_init)
  )
  
  current_pw        <- penalty_weight
  infeasible_streak <- 0
  models            <- build_dart_models(X_df, y_norm, as.factor(feas_bin), viol_norm)
  
  for (iter in seq_len(n_iter)) {
    
    if (infeasible_streak > 3) current_pw <- min(current_pw * 1.5, 20)
    
    y_range         <- max(max(y) - min(y), 1e-8)
    f_best_norm_cur <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
    
    acq    <- function(xv) dart_acq_fn(xv, models, col_names, f_best_norm_cur, current_pw)
    x_next <- as.numeric(maximise_acq_ga(acq, n_vars,
                                         sample.int(.Machine$integer.max, 1)))
    
    y_next     <- lscp_obj(x_next)
    feas_next  <- as.integer(lscp_feasible(x_next))
    viol_next  <- lscp_violation(x_next)
    true_evals <- true_evals + 1
    
    if (feas_next == 0) {
      infeasible_streak <- infeasible_streak + 1
    } else {
      infeasible_streak <- 0
      current_pw        <- max(current_pw * 0.9, penalty_weight)
    }
    
    X        <- rbind(X, x_next)
    X_df     <- as.data.frame(X); colnames(X_df) <- col_names
    y        <- c(y, y_next)
    feas_bin <- c(feas_bin, feas_next)
    viols    <- c(viols, viol_next)
    
    y_norm    <- normalize_01(y)
    viol_norm <- normalize_01(viols)
    
    f_best_info <- get_f_best(y, feas_bin)
    f_best      <- f_best_info$value
    
    history_df <- rbind(history_df, data.frame(
      iteration     = iter,
      y             = y_next,
      feasible      = feas_next,
      violation     = viol_next,
      f_best_so_far = ifelse(is.null(f_best), NA, f_best),
      true_evals    = true_evals
    ))
    
    models <- build_dart_models(X_df, y_norm, as.factor(feas_bin), viol_norm)
    
    if (verbose && iter %% 25 == 0) {
      cat(sprintf("  [DART] Iter %3d | evals: %d | feas: %d/%d | best: %s\n",
                  iter, true_evals, sum(feas_bin), length(feas_bin),
                  ifelse(is.null(f_best), "NA", f_best)))
    }
  }
  
  list(
    history          = history_df,
    X                = X,
    y                = y,
    feas             = feas_bin,
    f_best           = f_best,
    x_best           = if (!is.null(f_best_info$idx)) X[f_best_info$idx, ] else NULL,
    n_feasible       = sum(feas_bin),
    n_total          = length(feas_bin),
    feasibility_rate = mean(feas_bin),
    true_eval_count  = true_evals
  )
}

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
instance_seeds <- 1000 + seq_len(n_instances)
penalty_weight <- 5

glpk_result <- run_glpk()
cat(sprintf("\nGLPK optimal: %d  (%.2f sec)\n\n",
            glpk_result$objective, glpk_result$time_seconds))

all_results <- data.frame()

for (inst in seq_len(n_instances)) {
  seed_val <- instance_seeds[inst]
  cat(sprintf("\n========== Instance %d / %d  (seed = %d) ==========\n",
              inst, n_instances, seed_val))
  
  cat("  Running BART-CBO...\n")
  t0        <- Sys.time()
  bart      <- run_bart_cbo(seed_val, penalty_weight = penalty_weight, verbose = TRUE)
  bart_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  write.csv(bart$history,
            file.path(base_output_dir, sprintf("bart_history_inst%02d.csv", inst)),
            row.names = FALSE)
  
  cat("  Running DART-CBO...\n")
  t0        <- Sys.time()
  dart      <- run_dart_cbo(seed_val, penalty_weight = penalty_weight, verbose = TRUE)
  dart_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  write.csv(dart$history,
            file.path(base_output_dir, sprintf("dart_history_inst%02d.csv", inst)),
            row.names = FALSE)
  
  cat("  Running GA...\n")
  ga_res <- run_ga_lscp(seed_val)
  
  row <- data.frame(
    instance             = inst,
    seed                 = seed_val,
    glpk_objective       = glpk_result$objective,
    glpk_time_sec        = glpk_result$time_seconds,
    
    bart_best            = ifelse(is.null(bart$f_best), NA, bart$f_best),
    bart_n_feasible      = bart$n_feasible,
    bart_feasibility_pct = bart$feasibility_rate * 100,
    bart_true_evals      = bart$true_eval_count,
    bart_time_sec        = bart_time,
    bart_gap             = ifelse(is.null(bart$f_best), NA,
                                  bart$f_best - glpk_result$objective),
    
    dart_best            = ifelse(is.null(dart$f_best), NA, dart$f_best),
    dart_n_feasible      = dart$n_feasible,
    dart_feasibility_pct = dart$feasibility_rate * 100,
    dart_true_evals      = dart$true_eval_count,
    dart_time_sec        = dart_time,
    dart_gap             = ifelse(is.null(dart$f_best), NA,
                                  dart$f_best - glpk_result$objective),
    
    ga_best              = ga_res$f_best,
    ga_feasible          = as.integer(ga_res$feasible),
    ga_true_evals        = ga_res$true_eval_count,
    ga_time_sec          = ga_res$time_seconds,
    ga_gap               = ga_res$f_best - glpk_result$objective
  )
  
  all_results <- rbind(all_results, row)
  
  cat(sprintf("  BART: %s | DART: %s | GA: %d | GLPK: %d\n",
              ifelse(is.na(row$bart_best), "NA", row$bart_best),
              ifelse(is.na(row$dart_best), "NA", row$dart_best),
              row$ga_best, row$glpk_objective))
}

# =============================================================================
# SAVE + AGGREGATE
# =============================================================================
write.csv(all_results,
          file.path(base_output_dir, "comparison_all_instances.csv"),
          row.names = FALSE)

methods <- c("bart", "dart", "ga")
agg <- lapply(methods, function(m) {
  data.frame(
    method               = toupper(m),
    mean_best            = mean(all_results[[paste0(m, "_best")]],       na.rm = TRUE),
    sd_best              = sd(all_results[[paste0(m, "_best")]],         na.rm = TRUE),
    mean_gap             = mean(all_results[[paste0(m, "_gap")]],        na.rm = TRUE),
    sd_gap               = sd(all_results[[paste0(m, "_gap")]],          na.rm = TRUE),
    mean_evals           = mean(all_results[[paste0(m, "_true_evals")]], na.rm = TRUE),
    mean_time_sec        = mean(all_results[[paste0(m, "_time_sec")]],   na.rm = TRUE),
    n_feasible_instances = if (m == "ga") sum(all_results$ga_feasible)
    else sum(!is.na(all_results[[paste0(m, "_best")]]))
  )
})
agg_df <- rbind(
  data.frame(method = "GLPK", mean_best = glpk_result$objective, sd_best = 0,
             mean_gap = 0, sd_gap = 0, mean_evals = NA,
             mean_time_sec = glpk_result$time_seconds,
             n_feasible_instances = n_instances),
  do.call(rbind, agg)
)

write.csv(agg_df,
          file.path(base_output_dir, "comparison_aggregated.csv"),
          row.names = FALSE)

cat("\n=========================================================\n")
cat("        FINAL COMPARISON: LSCP OPTIMISATION\n")
cat("=========================================================\n\n")
print(agg_df, digits = 3, row.names = FALSE)

# =============================================================================
# CONVERGENCE PLOT
# =============================================================================
load_histories <- function(method_name, n) {
  lapply(seq_len(n), function(i) {
    df <- read.csv(file.path(base_output_dir,
                             sprintf("%s_history_inst%02d.csv", method_name, i)))
    df$instance <- i
    df$method   <- toupper(method_name)
    df
  })
}

all_hist <- do.call(rbind, c(load_histories("bart", n_instances),
                             load_histories("dart", n_instances)))
all_hist <- all_hist[!is.na(all_hist$f_best_so_far), ]

avg_hist <- all_hist %>%
  group_by(method, iteration) %>%
  summarise(mean_best = mean(f_best_so_far, na.rm = TRUE),
            sd_best   = sd(f_best_so_far,   na.rm = TRUE),
            .groups   = "drop")

p_conv <- ggplot(avg_hist, aes(x = iteration, y = mean_best,
                               colour = method, fill = method)) +
  geom_ribbon(aes(ymin = mean_best - sd_best, ymax = mean_best + sd_best),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = glpk_result$objective, linetype = "dashed",
             colour = "black", linewidth = 0.8) +
  annotate("text", x = n_iter * 0.6, y = glpk_result$objective + 0.15,
           label = sprintf("GLPK optimal = %d", glpk_result$objective),
           colour = "black", size = 3.5) +
  scale_colour_manual(values = c(BART = "#0072B2", DART = "#D55E00")) +
  scale_fill_manual(values   = c(BART = "#0072B2", DART = "#D55E00")) +
  labs(title    = "LSCP: BART-CBO vs DART-CBO Convergence",
       subtitle = sprintf("Mean +/- SD over %d instances | Budget = %d evals",
                          n_instances, n_init + n_iter),
       x = "BO Iteration", y = "Best Feasible Objective (# Facilities)",
       colour = "Method", fill = "Method") +
  theme_bw(base_size = 13) +
  theme(legend.position = "top")

ggsave(file.path(base_output_dir, "convergence_plot.pdf"),
       plot = p_conv, width = 8, height = 5)

cat("\nResults saved to:", base_output_dir, "\n")
cat("Convergence plot saved.\n")