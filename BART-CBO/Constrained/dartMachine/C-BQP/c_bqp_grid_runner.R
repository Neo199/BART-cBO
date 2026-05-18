# -------------------------------------------------------------
# HPC Array Worker: Constrained BQP Grid Search
# Sweeps penalty_w in {2, 5, 10} and n_init in {5, 10, 20}
#
# instance_id maps to (configs vary slowest, method next, rep fastest):
#   config 1 (pw=2,  ni=5):  1..N_REPS = BART-BO,  N_REPS+1..2*N_REPS = GA
#   config 2 (pw=2,  ni=10): 2*N_REPS+1..3*N_REPS  = BART-BO, ...
#   ... and so on for all 9 configs
#
# Total array size = N_CONFIGS * 2 * N_REPS = 9 * 2 * 10 = 180
# Example SLURM: --array=1-180
# -------------------------------------------------------------

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

# ── Parse instance ID ────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || is.na(as.integer(args[1]))) {
  instance_id <- 1L
  message("No instance ID supplied, defaulting to instance_id = 1")
} else {
  instance_id <- as.integer(args[1])
}
cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Starting instance", instance_id, "\n")

# ── Grid parameters ──────────────────────────────────────────
PENALTY_WEIGHTS <- c(2, 5, 10)
INIT_SAMPLES    <- c(5, 10, 20)

grid      <- expand.grid(penalty_w = PENALTY_WEIGHTS, n_init = INIT_SAMPLES)
grid      <- grid[order(grid$penalty_w, grid$n_init), ]
rownames(grid) <- NULL
N_CONFIGS <- nrow(grid)    # 9

# ── Fixed global parameters ──────────────────────────────────
N_REPS      <- 10
p           <- 100
alpha       <- 1
k_budget    <- 70
n_iter      <- 250
folder_name <- "c_bqp_grid_results"
dir.create(folder_name, showWarnings = FALSE, recursive = TRUE)

# ── Decode instance → config + method + rep ──────────────────
total_instances <- N_CONFIGS * 2L * N_REPS    # 180
if (instance_id < 1L || instance_id > total_instances) {
  stop(sprintf("instance_id %d out of range [1, %d]", instance_id, total_instances))
}

tmp        <- instance_id - 1L
rep_id     <- (tmp %% N_REPS) + 1L
tmp        <- tmp %/% N_REPS
method_idx <- (tmp %% 2L) + 1L                # 1 = BART-BO, 2 = GA
cfg_idx    <- (tmp %/% 2L) + 1L

cfg       <- grid[cfg_idx, ]
n_init    <- cfg$n_init
penalty_w <- cfg$penalty_w
method    <- c("BART-BO", "GA")[method_idx]
seed      <- 1000L * cfg_idx + 100L * method_idx + rep_id

cat(sprintf("Config %d/%d | penalty_w=%.0f | n_init=%d | Method: %s | Rep: %d | Seed: %d\n",
            cfg_idx, N_CONFIGS, penalty_w, n_init, method, rep_id, seed))
set.seed(seed)

# ── Problem helpers ──────────────────────────────────────────
quad_mat <- function(n_vars, alpha) {
  K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
  decay <- outer(1:n_vars, 1:n_vars, Vectorize(K))
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  Q * decay
}

Q <- quad_mat(p, alpha)

eval_counter <- 0L

true_f <- function(x) {
  eval_counter <<- eval_counter + 1L
  as.numeric(t(as.numeric(x)) %*% Q %*% as.numeric(x))
}
true_f_batch <- function(X_mat) apply(X_mat, 1, true_f)

bqp_feasible  <- function(x) as.integer(sum(x) <= k_budget)
bqp_violation <- function(x) max(0L, as.integer(sum(x)) - k_budget)

normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

get_f_best_feasible <- function(y, feas) {
  idx <- which(as.logical(feas))
  if (!length(idx)) return(list(value = NULL, idx = NULL))
  bi <- idx[which.min(y[idx])]
  list(value = y[bi], idx = bi)
}

# ── Per-method timing ────────────────────────────────────────
t_start <- proc.time()

# ============================================================
# METHOD 1: Constrained BART-BO
# ============================================================
run_bart_bo <- function() {
  library(dartMachine)
  library(GA)
  
  # ── Initial design ─────────────────────────────────────────
  X        <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)
  X_df     <- as.data.frame(X)
  colnames(X_df) <- paste0("x", 1:p)
  
  y        <- true_f_batch(X)
  feas_bin <- apply(X, 1, bqp_feasible)
  viols    <- apply(X, 1, bqp_violation)
  
  cat(sprintf("  Init: %d feasible / %d points | best feasible obj = %s\n",
              sum(feas_bin), n_init,
              ifelse(any(feas_bin == 1),
                     round(min(y[feas_bin == 1]), 4), "none")))
  
  # ── Surrogate fitter ──────────────────────────────────────
  fit_surrogates <- function(X_df, y_norm, feas_bin, viol_norm,
                             num_trees = 100) {
    obj_model  <- bartMachine(X = X_df, y = y_norm,
                              num_trees = num_trees, verbose = FALSE)
    viol_model <- if (var(viol_norm) > 1e-10) {
      bartMachine(X = X_df, y = viol_norm,
                  num_trees = num_trees, verbose = FALSE)
    } else NULL
    feas_model <- if (length(unique(feas_bin)) == 2) {
      bartMachine(X         = X_df,
                  y         = factor(feas_bin, levels = c("0", "1")),
                  num_trees = num_trees,
                  verbose   = FALSE)
    } else NULL
    list(obj = obj_model, viol = viol_model, feas = feas_model)
  }
  
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_surrogates(X_df, y_norm, feas_bin, viol_norm)
  
  # ── Track best feasible ────────────────────────────────────
  fb_info       <- get_f_best_feasible(y, feas_bin)
  f_best        <- fb_info$value
  best_idx      <- fb_info$idx
  best_x_so_far <- if (!is.null(best_idx)) X[best_idx, ] else NULL
  
  f_best_norm <- if (!is.null(f_best))
    (f_best - min(y)) / max(max(y) - min(y), 1e-8) else NULL
  
  current_pw        <- penalty_w
  infeasible_streak <- 0L
  best_trace        <- rep(NA_real_, n_iter)
  
  # ── Acquisition function ──────────────────────────────────
  acq_fn <- function(x_vec) {
    if (is.list(x_vec)) x_vec <- unlist(x_vec)
    x_vec <- as.numeric(x_vec)
    x_in  <- as.data.frame(matrix(x_vec, nrow = 1))
    colnames(x_in) <- colnames(X_df)
    
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
        pv <- predict(models$feas, new_data = x_in, type = "prob")
        max(min(as.numeric(pv), 1), 0)
      }
      if (is.na(p_feas)) p_feas <- 0.5
      
      pred_viol <- if (is.null(models$viol)) {
        0
      } else {
        post_viol  <- bart_machine_get_posterior(models$viol, new_data = x_in)
        viol_draws <- as.numeric(post_viol$y_hat_posterior_samples[1, ])
        max(0, mean(viol_draws, na.rm = TRUE))
      }
      if (is.na(pred_viol)) pred_viol <- 0
      
      if (is.null(f_best_norm)) {
        acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * (1 - mu)
      } else {
        z  <- (f_best_norm - mu) / sigma
        EI <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
        acq <- EI * p_feas - current_pw * pred_viol
      }
      
      if (is.na(acq) || is.nan(acq)) -1e6 else acq
    }, error = function(e) -1e6)
  }
  
  # ── BO loop ────────────────────────────────────────────────
  for (iter in seq_len(n_iter)) {
    
    if (infeasible_streak > 3)
      current_pw <- min(current_pw * 1.5, 50)
    
    GA_res <- ga(
      type     = "binary",
      nBits    = p,
      fitness  = acq_fn,
      popSize  = 60,
      maxiter  = 20,
      run      = 40,
      keepBest = TRUE,
      monitor  = FALSE,
      seed     = iter * rep_id
    )
    
    x_next    <- as.numeric(GA_res@solution[1, ])
    y_next    <- true_f(x_next)
    feas_next <- bqp_feasible(x_next)
    viol_next <- bqp_violation(x_next)
    
    if (feas_next == 0) {
      infeasible_streak <- infeasible_streak + 1L
    } else {
      infeasible_streak <- 0L
      current_pw        <- max(current_pw * 0.9, penalty_w)
    }
    
    X        <- rbind(X, x_next)
    X_df     <- as.data.frame(X)
    colnames(X_df) <- paste0("x", 1:p)
    y        <- c(y, y_next)
    feas_bin <- c(feas_bin, feas_next)
    viols    <- c(viols, viol_next)
    
    fb_info     <- get_f_best_feasible(y, feas_bin)
    f_best      <- fb_info$value
    best_idx    <- fb_info$idx
    if (!is.null(best_idx)) best_x_so_far <- X[best_idx, ]
    
    y_range     <- max(max(y) - min(y), 1e-8)
    f_best_norm <- if (!is.null(f_best)) (f_best - min(y)) / y_range else NULL
    
    best_trace[iter] <- if (!is.null(f_best)) f_best else NA_real_
    
    y_norm    <- normalize_01(y)
    viol_norm <- normalize_01(viols)
    models    <- fit_surrogates(X_df, y_norm, feas_bin, viol_norm)
    
    if (iter %% 50 == 0 || iter == 1)
      cat(sprintf("  Iter %3d | obj=%.4f | feas=%d | viol=%d | best=%s | pw=%.2f\n",
                  iter, y_next, feas_next, viol_next,
                  ifelse(is.null(f_best), "NA", sprintf("%.4f", f_best)),
                  current_pw))
  }
  
  list(
    method      = "BART-BO",
    rep         = rep_id,
    seed        = seed,
    penalty_w   = penalty_w,
    n_init      = n_init,
    cfg_idx     = cfg_idx,
    best        = f_best,
    best_x      = best_x_so_far,
    feasible    = !is.null(f_best),
    budget_used = if (!is.null(best_x_so_far)) sum(best_x_so_far) else NA,
    evals       = eval_counter,
    best_trace  = best_trace,     # length n_iter, NA until first feasible
    ga_trace    = NULL,
    feas_rate   = mean(feas_bin)
  )
}

# ============================================================
# METHOD 2: Penalty GA
# ============================================================
run_penalty_ga <- function() {
  library(GA)
  
  ga_fitness <- function(x_vec) {
    x    <- as.numeric(x_vec)
    obj  <- true_f(x)
    viol <- bqp_violation(x)
    -(obj + 1e3 * viol)
  }
  
  GA_final <- ga(
    type    = "binary",
    fitness = ga_fitness,
    nBits   = p,
    monitor = FALSE
  )
  
  ga_x    <- as.integer(GA_final@solution[1, ])
  ga_obj  <- as.numeric(t(ga_x) %*% Q %*% ga_x)
  ga_feas <- bqp_feasible(ga_x)
  
  # ── Extract per-generation best feasible objective ────────
  # GA@summary["max"] is the best (most positive) fitness per generation.
  # Fitness = -(obj + 1e3*viol), so if the best individual in a generation
  # is feasible (viol=0), then -fitness = obj  (<< 1e3).
  # Mark infeasible-generation bests as NA; forward-filled in aggregate.
  ga_summary   <- GA_final@summary
  gen_best_raw <- as.numeric(ga_summary[, "max"])
  gen_best_obj <- ifelse(-gen_best_raw < 1e3, -gen_best_raw, NA_real_)
  
  cat(sprintf("  GA done | obj=%.4f | feas=%d | sum(x)=%d | evals=%d | gens=%d\n",
              ga_obj, ga_feas, sum(ga_x), eval_counter, length(gen_best_obj)))
  
  list(
    method      = "GA",
    rep         = rep_id,
    seed        = seed,
    penalty_w   = penalty_w,
    n_init      = n_init,
    cfg_idx     = cfg_idx,
    best        = ga_obj,
    best_x      = ga_x,
    feasible    = as.logical(ga_feas),
    budget_used = sum(ga_x),
    evals       = eval_counter,
    best_trace  = NULL,
    ga_trace    = gen_best_obj,   # length n_generations, NA = infeasible gen
    feas_rate   = NA
  )
}

# ── Dispatch ─────────────────────────────────────────────────
result <- switch(method,
                 "BART-BO"    = run_bart_bo(),
                 "GA" = run_penalty_ga(),
                 stop("Unknown method: ", method)
)

result$elapsed_sec <- as.numeric((proc.time() - t_start)["elapsed"])

# ── Save result ───────────────────────────────────────────────
out_file <- file.path(
  folder_name,
  sprintf("instance_%03d_pw%02d_ni%02d_%s_rep%02d.rds",
          instance_id,
          as.integer(penalty_w),
          as.integer(n_init),
          gsub("-", "", gsub("Penalty-", "PenaltyGA_", method)),
          rep_id)
)
saveRDS(result, file = out_file)

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    "- Finished instance", instance_id,
    sprintf("| pw=%.0f | ni=%d | %s rep %d | best=%s | evals=%s | %.1fs\n",
            penalty_w, n_init, method, rep_id,
            ifelse(is.null(result$best) || is.na(result$best),
                   "NA", sprintf("%.6f", result$best)),
            ifelse(is.na(result$evals), "N/A", as.character(result$evals)),
            result$elapsed_sec))