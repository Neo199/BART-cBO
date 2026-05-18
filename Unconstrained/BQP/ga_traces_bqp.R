# -------------------------------------------------------------
# GA PATCH WORKER: Unconstrained BQP — re-run Pure-GA reps
#                  to capture per-generation traces.
#
# Run this INSTEAD OF re-running the full bqp_worker.R for GA.
# Existing DART-BO / BART-BO .rds files are untouched.
#
# Usage (local, all reps at once):
#   Rscript bqp_ga_patch.R
#
# Usage (SLURM array, one rep per job):
#   Rscript bqp_ga_patch.R <rep_id>   (rep_id in 1..N_REPS)
#   SLURM: --array=1-10
# -------------------------------------------------------------

# ── Parse optional rep_id ─────────────────────────────────────
args   <- commandArgs(trailingOnly = TRUE)
run_all <- length(args) == 0 || is.na(as.integer(args[1]))

if (run_all) {
  rep_ids <- 1:10          # adjust N_REPS here if needed
  message("No rep_id supplied — running all GA reps locally.")
} else {
  rep_ids <- as.integer(args[1])
}

# ── Parameters  (must match original bqp_worker.R exactly) ───
N_REPS      <- 10L
p           <- 100L
alpha       <- 1
folder_name <- "BART-CBO/Unconstrained/BQP/bqp_results"
method_idx  <- 3L          # Pure-GA is method 3 in the original script

dir.create(folder_name, showWarnings = FALSE, recursive = TRUE)

# ── Problem helpers ───────────────────────────────────────────
quad_mat <- function(n_vars, alpha) {
  K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
  decay <- outer(1:n_vars, 1:n_vars, Vectorize(K))
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  Q * decay
}

# ── Run one GA rep ────────────────────────────────────────────
run_ga_patch <- function(rep_id) {
  seed <- 1000L * method_idx + rep_id       # identical to original
  set.seed(seed)
  
  Q <- quad_mat(p, alpha)                   # regenerate same Q
  
  eval_counter <- 0L
  true_f <- function(x) {
    eval_counter <<- eval_counter + 1L
    as.numeric(t(as.numeric(x)) %*% Q %*% as.numeric(x))
  }
  
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      sprintf("- Starting Pure-GA rep %d (seed=%d)\n", rep_id, seed))
  
  t_start  <- proc.time()
  library(GA)
  
  GA_final <- ga(
    type    = "binary",
    fitness = function(x) -true_f(x),
    nBits   = p,
    monitor = FALSE
  )
  
  # ── Per-generation best objective ──────────────────────────
  # GA@summary["max"] = best (least negative) fitness per generation.
  # Fitness = -obj, so obj = -fitness.  No penalty, so all gens are feasible.
  gen_best_obj <- -as.numeric(GA_final@summary[, "max"])
  
  elapsed <- as.numeric((proc.time() - t_start)["elapsed"])
  
  result <- list(
    method      = "Pure-GA",
    rep         = rep_id,
    seed        = seed,
    best        = -GA_final@fitnessValue,   # scalar, matches original
    evals       = eval_counter,
    best_trace  = NULL,                     # not applicable for GA
    ga_trace    = gen_best_obj,             # NEW: per-generation best obj
    elapsed_sec = elapsed
  )
  
  # ── Save: overwrite original file (same naming convention) ─
  instance_id <- 2L * N_REPS + rep_id      # original instance_id for GA reps
  out_file <- file.path(
    folder_name,
    sprintf("instance_%03d_%s_rep%02d.rds",
            instance_id, "PureGA", rep_id)
  )
  saveRDS(result, file = out_file)
  
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      sprintf("- Saved rep %d | best=%.6f | evals=%d | gens=%d | %.1fs\n",
              rep_id, result$best, result$evals,
              length(gen_best_obj), elapsed))
}

# ── Dispatch ──────────────────────────────────────────────────
for (rid in rep_ids) run_ga_patch(rid)
cat("All done.\n")