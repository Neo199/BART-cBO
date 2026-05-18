# -------------------------------------------------------------
# HPC Array Worker: Binary Quadratic Programming Comparison
# Run via array job: Rscript bqp_worker.R <instance_id>
#
# instance_id maps to:
#   1..N_REPS          -> DART-BO   (rep 1..N_REPS)
#   N_REPS+1..2*N_REPS -> BART-BO   (rep 1..N_REPS)
#   2*N_REPS+1..3*N_REPS -> Pure GA (rep 1..N_REPS)
#
# Total array size = 3 * N_REPS
# Example SGE/SLURM: --array=1-30  (for N_REPS=10)
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
N_REPS      <- 10L          # replications per method
p           <- 100L         # number of binary variables
alpha       <- 1
n_init      <- 20L
n_iter      <- 250L
folder_name <- "bqp_results"
dir.create(folder_name, showWarnings = FALSE, recursive = TRUE)

# ── Decode instance → method + rep ──────────────────────────
total_instances <- 3L * N_REPS
if (instance_id < 1L || instance_id > total_instances) {
  stop(sprintf("instance_id %d out of range [1, %d]", instance_id, total_instances))
}

method_idx <- ((instance_id - 1L) %/% N_REPS) + 1L   # 1=DART, 2=BART, 3=GA
rep_id     <- ((instance_id - 1L) %%  N_REPS) + 1L
method     <- c("DART-BO", "BART-BO", "Pure-GA")[method_idx]
seed       <- 1000L * method_idx + rep_id              # unique, reproducible

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

normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# ── Per-method timing ────────────────────────────────────────
t_start <- proc.time()

# ============================================================
# METHOD 1: DART-BO
# ============================================================
run_dart <- function() {
  library(dartMachine)
  library(GA)

  X     <- matrix(rbinom(n_init * p, 1, 0.5), n_init, p)
  y     <- true_f_batch(X)
  X_df  <- data.frame(X)
  y_norm <- normalize_01(y)

  dart_fit   <- bartMachine(X_df, y_norm, num_trees = 100)
  f_best     <- min(y)
  x_best     <- X[which.min(y), ]
  best_trace <- numeric(n_iter)

  cat("DART Init: best =", round(f_best, 4), "\n")

  for (iter in seq_len(n_iter)) {
    fitness_fn <- function(x) {
      x_in  <- as.data.frame(matrix(as.numeric(x), 1))
      colnames(x_in) <- colnames(X_df)
      post  <- bart_machine_get_posterior(dart_fit, x_in)
      pred  <- as.numeric(post$y_hat_posterior_samples)
      mu    <- mean(pred, na.rm = TRUE)
      sigma <- sd(pred,   na.rm = TRUE)
      if (is.na(mu))                       mu    <- 0
      if (is.na(sigma) || sigma < 1e-10)  sigma <- 1e-6
      f_norm <- (f_best - min(y)) / (max(y) - min(y) + 1e-12)
      z <- (f_norm - mu) / sigma
      max((f_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
    }

    GA_res <- ga(type = "binary", nBits = p, fitness = fitness_fn,
                 popSize = 200, maxiter = 40, run = 20, keepBest = TRUE,
                 suggestions = matrix(x_best, nrow = 1), monitor = FALSE)

    x_next <- as.numeric(GA_res@solution[1, ])
    y_next <- true_f(x_next)
    X      <- rbind(X, x_next)
    y      <- c(y, y_next)
    if (y_next < f_best) { f_best <- y_next; x_best <- x_next }

    X_df   <- data.frame(X)
    y_norm <- normalize_01(y)
    f_best <- min(y)
    dart_fit <- bartMachine(X_df, y_norm, num_trees = 100)
    best_trace[iter] <- f_best
    if (iter %% 50 == 0 || iter == 1)
      cat(sprintf("  DART Iter %3d: best = %8.4f\n", iter, f_best))
  }

  list(method = "DART-BO", rep = rep_id, seed = seed,
       best = f_best, evals = eval_counter, best_trace = best_trace)
}

# ============================================================
# METHOD 2: BART-BO
# ============================================================
run_bart <- function() {
  library(BART)
  library(GA)

  X      <- matrix(rbinom(n_init * p, 1, 0.5), n_init, p)
  y      <- true_f_batch(X)
  y_norm <- normalize_01(y)

  bart_fit <- wbart(X, y_norm, ndpost = 1000, nskip = 200,
                    ntree = 100, printevery = 100000)
  f_best     <- min(y)
  x_best     <- X[which.min(y), ]
  best_trace <- numeric(n_iter)

  cat("BART Init: best =", round(f_best, 4), "\n")

  for (iter in seq_len(n_iter)) {
    fitness_fn <- function(x) {
      x_mat <- matrix(as.numeric(x), 1)
      pred  <- predict(bart_fit, x_mat)
      if (nrow(pred) < ncol(pred)) pred <- t(pred)
      mu    <- mean(pred)
      sigma <- sd(pred)
      if (is.na(mu))                       mu    <- 0
      if (is.na(sigma) || sigma < 1e-10)  sigma <- 1e-6
      f_norm <- (f_best - min(y)) / (max(y) - min(y) + 1e-12)
      z <- (f_norm - mu) / sigma
      max((f_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
    }

    GA_res <- ga(type = "binary", nBits = p, fitness = fitness_fn,
                 popSize = 200, maxiter = 40, run = 20, keepBest = TRUE,
                 suggestions = matrix(x_best, nrow = 1), monitor = FALSE)

    x_next <- as.numeric(GA_res@solution[1, ])
    y_next <- true_f(x_next)
    X      <- rbind(X, x_next)
    y      <- c(y, y_next)
    if (y_next < f_best) { f_best <- y_next; x_best <- x_next }

    y_norm <- normalize_01(y)
    f_best <- min(y)
    bart_fit <- wbart(X, y_norm, ndpost = 1000, nskip = 200,
                      ntree = 100, printevery = 100000)
    best_trace[iter] <- f_best
    if (iter %% 50 == 0 || iter == 1)
      cat(sprintf("  BART Iter %3d: best = %8.4f\n", iter, f_best))
  }

  list(method = "BART-BO", rep = rep_id, seed = seed,
       best = f_best, evals = eval_counter, best_trace = best_trace)
}

# ============================================================
# METHOD 3: Pure GA
# ============================================================
run_ga <- function() {
  library(GA)

  GA_final <- ga(type = "binary", fitness = function(x) -true_f(x),
                 nBits = p, monitor = FALSE)

  list(method = "Pure-GA", rep = rep_id, seed = seed,
       best = -GA_final@fitnessValue, evals = eval_counter,
       best_trace = NULL)
}

# ── Dispatch ─────────────────────────────────────────────────
result <- switch(method,
  "DART-BO" = run_dart(),
  "BART-BO" = run_bart(),
  "Pure-GA" = run_ga(),
  stop("Unknown method: ", method)
)

result$elapsed_sec <- as.numeric((proc.time() - t_start)["elapsed"])

# ── Save result ───────────────────────────────────────────────
out_file <- file.path(folder_name,
                      sprintf("instance_%03d_%s_rep%02d.rds",
                              instance_id, gsub("-", "", method), rep_id))
saveRDS(result, file = out_file)

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    "- Finished instance", instance_id,
    sprintf("| %s rep %d | best=%.6f | evals=%d | %.1fs\n",
            method, rep_id, result$best, result$evals, result$elapsed_sec))
