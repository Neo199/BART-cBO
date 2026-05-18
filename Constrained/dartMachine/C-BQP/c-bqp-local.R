# =============================================================
# LOCAL RSTUDIO TEST SCRIPT - CONSOLE OUTPUT ONLY
# =============================================================

options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")

# ── Test controls ─────────────────────────────────────────────
TEST_METHOD <- "BART-BO"   # change to "Penalty-GA" to test that

# ── Scaled-down parameters ────────────────────────────────────
p         <- 100
alpha     <- 1
k_budget  <- 70
n_init    <- 5
n_iter    <- 5
penalty_w <- 5

# ── Hardcode instance/method ──────────────────────────────────
rep_id     <- 1L
method_idx <- if (TEST_METHOD == "BART-BO") 1L else 2L
method     <- TEST_METHOD
seed       <- 100 * method_idx + rep_id

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Starting local test\n")
cat(sprintf("Method: %s | Rep: %d | Seed: %d\n", method, rep_id, seed))
set.seed(seed)

# ── Problem helpers ───────────────────────────────────────────
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

t_start <- proc.time()

# ============================================================
# METHOD 1: BART-BO
# ============================================================
run_bart_bo <- function() {
  library(dartMachine)
  library(GA)
  
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
  
  fit_surrogates <- function(X_df, y_norm, feas_bin, viol_norm, num_trees = 50) {
    obj_model <- bartMachine(X = X_df, y = y_norm,
                             num_trees = num_trees, verbose = FALSE)
    
    # When all violations are zero, v_hat_k(x) = 0 for all x.
    # Acquisition reduces to: f_a(x) = eta * p_feas(x)  [paper eq.]
    # No surrogate needed — we handle this in acq_fn by returning 0 directly.
    viol_model <- if (var(viol_norm) > 1e-10) {
      bartMachine(X = X_df, y = viol_norm,
                  num_trees = num_trees, verbose = FALSE)
    } else NULL
    
    feas_model <- if (length(unique(feas_bin)) == 2) {
      bartMachine(X = X_df, y = factor(feas_bin, levels = c("0","1")),
                  num_trees = num_trees, verbose = FALSE)
    } else NULL
    
    list(obj = obj_model, viol = viol_model, feas = feas_model)
  }
  
  y_norm    <- normalize_01(y)
  viol_norm <- normalize_01(viols)
  models    <- fit_surrogates(X_df, y_norm, feas_bin, viol_norm)
  
  fb_info       <- get_f_best_feasible(y, feas_bin)
  f_best        <- fb_info$value
  best_idx      <- fb_info$idx
  best_x_so_far <- if (!is.null(best_idx)) X[best_idx, ] else NULL
  f_best_norm   <- if (!is.null(f_best))
    (f_best - min(y)) / max(max(y) - min(y), 1e-8) else NULL
  
  current_pw        <- penalty_w
  infeasible_streak <- 0L
  best_trace        <- rep(NA_real_, n_iter)
  
  acq_fn <- function(x_vec) {
    if (is.list(x_vec)) x_vec <- unlist(x_vec)
    x_vec <- as.numeric(x_vec)
    x_in  <- as.data.frame(matrix(x_vec, nrow = 1))
    colnames(x_in) <- colnames(X_df)
    
    tryCatch({
      post_obj  <- bart_machine_get_posterior(models$obj, new_data = x_in)
      obj_draws <- as.numeric(post_obj$y_hat_posterior_samples[1, ])
      mu    <- mean(obj_draws, na.rm = TRUE)
      sigma <- sd(obj_draws,   na.rm = TRUE)
      if (is.na(mu))                     mu    <- 0
      if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
      
      p_feas <- if (is.null(models$feas)) {
        0.5
      } else {
        pv <- predict(models$feas, new_data = x_in, type = "prob")
        max(min(as.numeric(pv), 1), 0)
      }
      if (is.na(p_feas)) p_feas <- 0.5
      
      # No violations observed yet => v_hat_k(x) = 0
      # => penalty term lambda * sum_k v_hat_k(x) vanishes
      # => f_a(x) = eta * p_feas(x)          [paper eq.]
      pred_viol <- if (is.null(models$viol)) {
        0
      } else {
        pv2 <- bart_machine_get_posterior(models$viol, new_data = x_in)
        max(0, mean(as.numeric(pv2$y_hat_posterior_samples[1, ]), na.rm = TRUE))
      }
      if (is.na(pred_viol)) pred_viol <- 0
      
      if (is.null(f_best_norm)) {
        # No feasible solution yet: f_a(x) = eta * p_feas(x) - lambda * v_hat(x)
        acq <- 10 * p_feas - current_pw * pred_viol + 0.01 * (1 - mu)
      } else {
        # Feasible solution exists: switch to EI-based acquisition
        z  <- (f_best_norm - mu) / sigma
        EI <- max((f_best_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
        acq <- EI * p_feas - current_pw * pred_viol
      }
      
      if (is.na(acq) || is.nan(acq)) -1e6 else acq
    }, error = function(e) -1e6)
  }
  
  for (iter in seq_len(n_iter)) {
    if (infeasible_streak > 3)
      current_pw <- min(current_pw * 1.5, 50)
    
    GA_res <- ga(type = "binary", nBits = p, fitness = acq_fn,
                 popSize = 20, maxiter = 10, run = 10,
                 keepBest = TRUE, monitor = FALSE, seed = iter * rep_id)
    
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
    
    cat(sprintf("  Iter %d | obj=%.4f | feas=%d | viol=%d | best=%s | pw=%.2f | viol_model=%s\n",
                iter, y_next, feas_next, viol_next,
                ifelse(is.null(f_best), "NA", sprintf("%.4f", f_best)),
                current_pw,
                ifelse(is.null(models$viol), "zero (paper eq.)", "fitted")))
  }
  
  list(method = "BART-BO", rep = rep_id, seed = seed,
       best = f_best, best_x = best_x_so_far,
       feasible = !is.null(f_best),
       budget_used = if (!is.null(best_x_so_far)) sum(best_x_so_far) else NA,
       evals = eval_counter, best_trace = best_trace,
       feas_rate = mean(feas_bin))
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
  
  GA_final <- ga(type = "binary", fitness = ga_fitness,
                 nBits = p, monitor = FALSE)
  
  ga_x    <- as.integer(GA_final@solution[1, ])
  ga_obj  <- as.numeric(t(ga_x) %*% Q %*% ga_x)
  ga_feas <- bqp_feasible(ga_x)
  
  cat(sprintf("  GA done | obj=%.4f | feas=%d | sum(x)=%d | evals=%d\n",
              ga_obj, ga_feas, sum(ga_x), eval_counter))
  
  list(method = "Penalty-GA", rep = rep_id, seed = seed,
       best = ga_obj, best_x = ga_x,
       feasible = as.logical(ga_feas),
       budget_used = sum(ga_x), evals = eval_counter,
       best_trace = NULL, feas_rate = NA)
}

# ── Run ───────────────────────────────────────────────────────
result <- switch(method,
                 "BART-BO"    = run_bart_bo(),
                 "Penalty-GA" = run_penalty_ga(),
                 stop("Unknown method: ", method))

result$elapsed_sec <- as.numeric((proc.time() - t_start)["elapsed"])

# ── Print everything to console ───────────────────────────────
cat("\n============================================================\n")
cat("RESULTS SUMMARY\n")
cat("============================================================\n")
cat(sprintf("Method:      %s\n",  result$method))
cat(sprintf("Feasible:    %s\n",  result$feasible))
cat(sprintf("Best obj:    %.6f\n", result$best))
cat(sprintf("Budget used: %s / %d\n", result$budget_used, k_budget))
cat(sprintf("Evals:       %d\n",  result$evals))
cat(sprintf("Feas rate:   %s\n",  ifelse(is.na(result$feas_rate), "NA",
                                         sprintf("%.1f%%", result$feas_rate * 100))))
cat(sprintf("Time:        %.1fs\n", result$elapsed_sec))
cat("\nBest x found:\n")
print(result$best_x)
cat("\nBest trace (feasible obj per iter):\n")
print(result$best_trace)
cat("============================================================\n")
