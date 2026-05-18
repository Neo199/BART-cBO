# =============================================================================
# GA TRACE COLLECTION — 0/1 Knapsack
# Reproduces benchmark GA runs (same seeds) and captures obj vs generation
# No BART-BO machinery needed — results are seed-reproducible.
# =============================================================================
suppressPackageStartupMessages(library(GA))

# =============================================================================
# EXPERIMENT GRID  — must match worker script exactly
# =============================================================================
penalty_weights <- c(0.1, 1, 2, 5, 10)
n_instances     <- 10
n_iter          <- 250   # kept for reference; GA uses its own defaults

grid <- expand.grid(
  penalty_weight = penalty_weights,
  instance       = seq_len(n_instances)
)

# =============================================================================
# PROBLEM SETUP (24-item knapsack)
# =============================================================================
p       <- 24L
weights <- c(382745, 799601, 909247, 729069, 467902,  44328,  34610, 698150,
             823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
             323046, 446298, 931161,  31385, 496951, 264724, 224916, 169684)
values  <- c(825594, 1677009, 1676628, 1523970, 943972,  97426,  69666, 1296457,
             1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538,
             675367,  853655, 1826027,   65731, 901489,  577243,  466257,  369261)
W       <- 6404180L

true_f       <- function(x) sum(values * x)
constraint_fn <- function(x) sum(weights * x) <= W

# =============================================================================
# GA with per-generation trace capture via monitor callback
# =============================================================================
run_ga_with_trace <- function(seed_val) {
  set.seed(seed_val)
  
  # Store best feasible value found so far each generation
  trace <- numeric(0)
  
  running_best <- -Inf   # best feasible value seen across all generations
  
  monitor_fn <- function(obj, ...) {
    # obj is the GA object mid-run; obj@population and obj@fitness are current gen
    pop     <- obj@population
    fit_raw <- obj@fitness   # these are the raw fitness values (or -Inf if infeasible)
    
    # best feasible this generation
    feas_vals <- fit_raw[is.finite(fit_raw) & fit_raw > -Inf]
    gen_best  <- if (length(feas_vals) > 0) max(feas_vals) else NA_real_
    
    # update running best (cumulative max — makes a monotone trace)
    if (!is.na(gen_best) && gen_best > running_best)
      running_best <<- gen_best
    
    trace <<- c(trace, running_best)
  }
  
  fitness_fn <- function(x_vec) {
    x <- as.numeric(x_vec)
    if (constraint_fn(x)) true_f(x) else -Inf
  }
  
  GA_final <- ga(
    type    = "binary",
    fitness = fitness_fn,
    nBits   = p,
    monitor = monitor_fn   # <-- captures per-generation info
  )
  
  list(
    trace   = trace,           # length = number of generations run
    x_best  = GA_final@solution[1, ],
    f_best  = GA_final@fitnessValue
  )
}

# =============================================================================
# RUN ALL INSTANCES  (penalty_weight doesn't affect benchmark GA, but we
# replicate the same seed logic so traces match worker script exactly)
# =============================================================================
output_dir <- "knapsack_ga_traces"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

all_traces <- list()

for (i in seq_len(nrow(grid))) {
  pw       <- grid$penalty_weight[i]
  inst     <- grid$instance[i]
  seed_val <- 1000L + inst
  
  cat(sprintf("Running GA  |  task %3d / %d  |  penalty=%.1f  |  instance=%d  |  seed=%d\n",
              i, nrow(grid), pw, inst, seed_val))
  
  result <- run_ga_with_trace(seed_val)
  
  # Save per-instance trace CSV
  trace_df <- data.frame(
    generation     = seq_along(result$trace),
    best_feasible  = result$trace,
    penalty_weight = pw,
    instance       = inst,
    seed           = seed_val
  )
  
  pw_dir <- file.path(output_dir, sprintf("penalty_%.1f", pw))
  dir.create(pw_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(trace_df,
            file.path(pw_dir, sprintf("ga_trace_instance_%02d.csv", inst)),
            row.names = FALSE)
  
  all_traces[[i]] <- trace_df
}

# =============================================================================
# COMBINED CSV — all traces in one file for easy plotting
# =============================================================================
combined <- do.call(rbind, all_traces)
write.csv(combined,
          file.path(output_dir, "all_ga_traces.csv"),
          row.names = FALSE)

cat(sprintf("\nDone. Combined trace saved to: %s/all_ga_traces.csv\n", output_dir))
cat(sprintf("Individual traces in: %s/penalty_*/\n", output_dir))

# =============================================================================
# QUICK BASE-R PLOT  (one panel per penalty weight — lines = instances)
# =============================================================================
pdf(file.path(output_dir, "ga_traces_plot.pdf"), width = 12, height = 8)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

for (pw in penalty_weights) {
  sub <- combined[combined$penalty_weight == pw, ]
  instances <- unique(sub$instance)
  
  # determine axis limits across all instances for this pw
  xlim <- range(sub$generation, na.rm = TRUE)
  ylim <- range(sub$best_feasible[is.finite(sub$best_feasible)], na.rm = TRUE)
  
  plot(NULL, xlim = xlim, ylim = ylim,
       xlab = "Generation", ylab = "Best feasible objective",
       main = sprintf("penalty_weight = %.1f", pw))
  
  cols <- rainbow(length(instances))
  for (j in seq_along(instances)) {
    inst_data <- sub[sub$instance == instances[j], ]
    lines(inst_data$generation, inst_data$best_feasible,
          col = cols[j], lwd = 1.2)
  }
  legend("bottomright", legend = paste0("inst ", instances),
         col = cols, lty = 1, lwd = 1.2, cex = 0.6, bty = "n")
}
dev.off()

cat(sprintf("Plot saved to: %s/ga_traces_plot.pdf\n", output_dir))