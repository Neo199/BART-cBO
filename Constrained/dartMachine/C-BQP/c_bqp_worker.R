# ------------------------------------------------------------- 
# HPC Array Worker: Constrained Binary Quadratic Programming 
# Run via array job: Rscript c_bqp_worker.R <instance_id> 
# 
# instance_id maps to: 
#   1..N_REPS            -> Constrained BART-BO  (rep 1..N_REPS) 
#   N_REPS+1..2*N_REPS   -> Penalty GA           (rep 1..N_REPS) 
# 
# Total array size = 2 * N_REPS 
# Example SLURM: --array=1-20  (for N_REPS=10) 
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

# ── Global parameters ──────────────────────────────────────── 
N_REPS      <- 10          # replications per stochastic method 
p           <- 50         # number of binary variables 
alpha       <- 1            # correlation length for Q matrix 
k_budget    <- 40          # cardinality constraint: sum(x) <= k_budget 
n_init      <- 10          # initial design size 
n_iter      <- 250         # BO iterations 
penalty_w   <- 5            # initial violation-penalty weight 
folder_name <- "c_bqp_result_n50" 
dir.create(folder_name, showWarnings = FALSE, recursive = TRUE) 

# ── Decode instance → method + rep ────────────────────────── 
total_instances <- 2L * N_REPS    # BART-BO reps + GA reps 
if (instance_id < 1L || instance_id > total_instances) { 
  stop(sprintf("instance_id %d out of range [1, %d]", instance_id, total_instances)) 
} 

if (instance_id <= N_REPS) { 
  method     <- "BART-BO" 
  rep_id     <- instance_id 
  method_idx <- 1L 
} else { 
  method     <- "Penalty-GA" 
  rep_id     <- instance_id - N_REPS 
  method_idx <- 2L 
} 

seed <- 100 * method_idx + rep_id    # unique, reproducible 
cat(sprintf("Method: %s | Rep: %d | Seed: %d\n", method, rep_id, seed)) 
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
  
  # ── Surrogate fitter ───────────────────────────────────────── 
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
  
  # ── Track best feasible ───────────────────────────────────── 
  fb_info       <- get_f_best_feasible(y, feas_bin) 
  f_best        <- fb_info$value 
  best_idx      <- fb_info$idx 
  best_x_so_far <- if (!is.null(best_idx)) X[best_idx, ] else NULL 
  
  f_best_norm <- if (!is.null(f_best)) 
    (f_best - min(y)) / max(max(y) - min(y), 1e-8) else NULL 
  
  current_pw        <- penalty_w 
  infeasible_streak <- 0L 
  best_trace        <- rep(NA_real_, n_iter) 
  
  # ── Acquisition function ────────────────────────────────────── 
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
  
  # ── BO loop ───────────────────────────────────────────────── 
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
    method        = "BART-BO", 
    rep           = rep_id, 
    seed          = seed, 
    best          = f_best, 
    best_x        = best_x_so_far, 
    feasible      = !is.null(f_best), 
    budget_used   = if (!is.null(best_x_so_far)) sum(best_x_so_far) else NA, 
    evals         = eval_counter, 
    best_trace    = best_trace, 
    feas_rate     = mean(feas_bin) 
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
    -(obj + 1e3 * viol)    # GA maximises; negate and penalise 
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
  
  # ── Extract per-generation best feasible objective ────────────
  # GA@summary is a matrix with columns: mean, max, etc. — one row per generation.
  # We recover the true objective for each generation's best individual by
  # un-negating and un-penalising: if penalty term is 0 (feasible), -fitness = obj.
  # For safety we store the raw best fitness and transform it,
  # treating infeasible gens as NA (will be forward-filled in the aggregate script).
  ga_summary   <- GA_final@summary                         # matrix: n_gen x cols
  gen_best_raw <- as.numeric(ga_summary[, "max"])          # best fitness per gen (negated penalised)
  
  # Convert: obj = -fitness when feasible (viol=0), else NA
  # A generation's best is likely feasible if -fitness < 1e3 (penalty kicks in at 1e3 * viol)
  gen_best_obj <- ifelse(-gen_best_raw < 1e3,              # heuristic feasibility guard
                         -gen_best_raw,
                         NA_real_)
  
  cat(sprintf("  GA done | obj=%.4f | feas=%d | sum(x)=%d | evals=%d | gens=%d\n", 
              ga_obj, ga_feas, sum(ga_x), eval_counter, length(gen_best_obj))) 
  
  list( 
    method      = "Penalty-GA", 
    rep         = rep_id, 
    seed        = seed, 
    best        = ga_obj, 
    best_x      = ga_x, 
    feasible    = as.logical(ga_feas), 
    budget_used = sum(ga_x), 
    evals       = eval_counter, 
    best_trace  = NULL,          # BO-style trace not applicable for GA
    ga_trace    = gen_best_obj,  # per-generation best feasible objective
    feas_rate   = NA 
  ) 
} 

# ── Dispatch ──────────────────────────────────────────────── 
result <- switch(method, 
                 "BART-BO"    = run_bart_bo(), 
                 "Penalty-GA" = run_penalty_ga(), 
                 stop("Unknown method: ", method) 
) 

result$elapsed_sec <- as.numeric((proc.time() - t_start)["elapsed"]) 

# ── Save result ──────────────────────────────────────────────── 
out_file <- file.path( 
  folder_name, 
  sprintf("instance_%03d_%s_rep%02d.rds", 
          instance_id, 
          gsub("-", "", gsub("Penalty-", "PenaltyGA_", method)), 
          rep_id) 
) 
saveRDS(result, file = out_file) 

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), 
    "- Finished instance", instance_id, 
    sprintf("| %s rep %d | best=%s | evals=%s | %.1fs\n", 
            method, rep_id, 
            ifelse(is.null(result$best) || is.na(result$best), 
                   "NA", sprintf("%.6f", result$best)), 
            ifelse(is.na(result$evals), "N/A", as.character(result$evals)), 
            result$elapsed_sec))