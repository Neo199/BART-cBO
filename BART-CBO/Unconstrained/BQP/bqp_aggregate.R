# =============================================================================
# AGGREGATION + PLOTS — Unconstrained BQP (DART-BO vs BART-BO vs Pure-GA)
# Run after all array jobs + bqp_ga_patch.R complete:
#   Rscript bqp_aggregate.R
# =============================================================================

library(ggplot2)
library(dplyr)
library(zoo)
library(patchwork)

folder_name <- "BART-CBO/Unconstrained/BQP/bqp_results"
plot_dir    <- file.path(folder_name, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# =============================================================================
# 1.  LOAD ALL RESULT FILES
# =============================================================================
rds_files <- list.files(folder_name, pattern = "\\.rds$", full.names = TRUE)
rds_files <- rds_files[!grepl("convergence_traces\\.rds$", rds_files)]
if (length(rds_files) == 0) stop("No .rds files found in: ", folder_name)
cat("Found", length(rds_files), "result files\n\n")
results <- lapply(rds_files, readRDS)

# =============================================================================
# 2.  FLATTEN TO SUMMARY DATA FRAME
# =============================================================================
summary_df <- do.call(rbind, lapply(results, function(r) {
  if (!is.list(r) || is.data.frame(r)) return(NULL)
  data.frame(
    method      = r$method,
    rep         = r$rep,
    seed        = r$seed,
    best        = r$best,
    evals       = r$evals,
    elapsed_sec = r$elapsed_sec,
    stringsAsFactors = FALSE
  )
}))
summary_df <- summary_df[order(summary_df$method, summary_df$rep), ]

# =============================================================================
# 3.  CONSOLE SUMMARY  (preserved from original)
# =============================================================================
methods <- unique(summary_df$method)
agg <- do.call(rbind, lapply(methods, function(m) {
  d <- summary_df[summary_df$method == m, ]
  data.frame(
    Method    = m,
    N_reps    = nrow(d),
    Mean_best = round(mean(d$best),  4),
    SD_best   = round(sd(d$best),    4),
    Min_best  = round(min(d$best),   4),
    Max_best  = round(max(d$best),   4),
    Mean_eval = round(mean(d$evals), 1),
    Mean_sec  = round(mean(d$elapsed_sec), 1),
    stringsAsFactors = FALSE
  )
}))

cat("========================================\n")
cat("FINAL COMPARISON SUMMARY\n")
cat("========================================\n\n")
print(agg, row.names = FALSE)

ga_row  <- agg[agg$Method == "Pure-GA", ]
bo_rows <- agg[agg$Method != "Pure-GA", ]
cat("\n--- Relative to Pure-GA baseline ---\n")
for (i in seq_len(nrow(bo_rows))) {
  m     <- bo_rows$Method[i]
  gap   <- (bo_rows$Mean_best[i] - ga_row$Mean_best) / abs(ga_row$Mean_best) * 100
  e_pct <- 100 * bo_rows$Mean_eval[i] / ga_row$Mean_eval
  if (gap < 0) {
    cat(sprintf("%s: %.2f%% better objective, %.1f%% of GA evaluations\n",
                m, abs(gap), e_pct))
  } else {
    cat(sprintf("%s: %.2f%% worse objective, %.1f%% of GA evaluations\n",
                m, gap, e_pct))
  }
}

# =============================================================================
# 4.  OKABE-ITO PALETTE
# =============================================================================
col_dart <- "#0072B2"   # blue
col_bart <- "#009E73"   # green
col_ga   <- "#E69F00"   # amber

# =============================================================================
# 5.  PLOT 1 (TOP) — DART-BO & BART-BO convergence traces
# =============================================================================
bo_traces <- Filter(function(r) !is.null(r$best_trace), results)

bo_trace_df <- do.call(rbind, lapply(bo_traces, function(r) {
  data.frame(
    method = r$method,
    rep    = r$rep,
    iter   = seq_along(r$best_trace),
    best   = r$best_trace,
    stringsAsFactors = FALSE
  )
}))

bo_agg <- bo_trace_df |>
  group_by(method, iter) |>
  summarise(
    mean_best = mean(best, na.rm = TRUE),
    sd_best   = sd(best,   na.rm = TRUE),
    .groups   = "drop"
  ) |>
  mutate(method_f = factor(method, levels = c("DART-BO", "BART-BO")))

# =============================================================================
# 6.  PLOT 2 (BOTTOM) — Pure-GA per-generation trace
#     If ga_trace is missing from saved files (old runs pre-patch), regenerate
#     GA traces inline using the same seeds. Fast enough to run locally.
# =============================================================================
ga_traces <- Filter(function(r) !is.null(r$ga_trace) && r$method == "Pure-GA",
                    results)

if (length(ga_traces) == 0) {
  message("ga_trace not found in saved files — regenerating GA traces inline (fast).")
  library(GA)
  
  # Parameters must match bqp_worker.R exactly
  p_bqp      <- 100L
  alpha_bqp  <- 1
  method_idx <- 3L   # Pure-GA is method 3
  
  quad_mat_local <- function(n_vars, alpha) {
    K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
    decay <- outer(1:n_vars, 1:n_vars, Vectorize(K))
    Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
    Q * decay
  }
  
  ga_traces <- lapply(1:10, function(rep_id) {
    seed <- 1000L * method_idx + rep_id
    set.seed(seed)
    Q_local   <- quad_mat_local(p_bqp, alpha_bqp)
    true_f_local <- function(x) as.numeric(t(as.numeric(x)) %*% Q_local %*% as.numeric(x))
    
    GA_final <- ga(
      type    = "binary",
      fitness = function(x) -true_f_local(x),
      nBits   = p_bqp,
      monitor = FALSE
    )
    
    # Fitness = -obj, no penalty — obj = -fitness for every generation
    gen_best_obj <- -as.numeric(GA_final@summary[, "max"])
    cat(sprintf("  GA rep %d done | gens=%d | best=%.4f\n",
                rep_id, length(gen_best_obj), -GA_final@fitnessValue))
    
    list(method   = "Pure-GA",
         rep      = rep_id,
         ga_trace = gen_best_obj)
  })
}

ga_trace_df <- do.call(rbind, lapply(ga_traces, function(r) {
  data.frame(
    rep  = r$rep,
    gen  = seq_along(r$ga_trace),
    best = r$ga_trace,
    stringsAsFactors = FALSE
  )
}))

# Forward-fill within each rep (defensive — GA minimises without constraints
# so NAs are not expected, but safe to carry forward just in case)
ga_trace_df <- ga_trace_df |>
  group_by(rep) |>
  mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
  ungroup()

ga_agg <- ga_trace_df |>
  group_by(gen) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    .groups   = "drop"
  )

# =============================================================================
# 7.  SHARED Y-AXIS BREAKS  (interval of 5, computed from combined range)
# =============================================================================
y_all    <- c(bo_agg$mean_best - bo_agg$sd_best,
              bo_agg$mean_best + bo_agg$sd_best,
              ga_agg$mean_best - ga_agg$sd_best,
              ga_agg$mean_best + ga_agg$sd_best)
y_range  <- range(y_all, na.rm = TRUE)
y_breaks <- seq(floor(y_range[1] / 5) * 5,
                ceiling(y_range[2] / 5) * 5,
                by = 5)

# =============================================================================
# 8.  BUILD PLOTS
# =============================================================================
p_bo <- ggplot(bo_agg, aes(x = iter, colour = method_f, fill = method_f)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              alpha = 0.20, colour = NA) +
  geom_line(aes(y = mean_best), linewidth = 0.9) +
  scale_colour_manual(
    name   = "Method",
    values = c("DART-BO" = col_dart, "BART-BO" = col_bart),
    labels = c("DART-BO" = "Dirichlet BART-BO", "BART-BO" = "Standard BART-BO")
  ) +
  scale_fill_manual(
    name   = "Method",
    values = c("DART-BO" = col_dart, "BART-BO" = col_bart),
    labels = c("DART-BO" = "Dirichlet BART-BO", "BART-BO" = "Standard BART-BO")
  ) +
  scale_y_continuous(breaks = y_breaks) +
  labs(
    title    = "Dirichlet BART-BO & Standard BART-BO — Best Objective vs. BO Iteration",
    subtitle = "Mean \u00b1 1 SD across 10 reps (ribbon).",
    x        = "BO Iteration",
    y        = "Best Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )
 
p_ga <- ggplot(ga_agg, aes(x = gen)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              fill = col_ga, alpha = 0.25) +
  geom_line(aes(y = mean_best), colour = col_ga, linewidth = 0.9) +
  scale_y_continuous(breaks = y_breaks) +
  labs(
    title    = "GA — Best Objective vs. Generation",
    subtitle = "Mean \u00b1 1 SD across 10 reps (ribbon).",
    x        = "Generation",
    y        = "Best Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = col_ga),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    panel.grid.minor = element_blank()
  )

# =============================================================================
# 9.  COMBINE AND SAVE
# =============================================================================
p_combined <- p_bo / p_ga +
  plot_annotation(
    title   = "Unconstrained BQP — Dirichlet BART-BO & Standard BART-BO vs GA",
    theme   = theme(
      plot.title   = element_text(face = "bold", size = 15, hjust = 0.5),
      plot.caption = element_text(colour = "grey50", size = 9)
    )
  )

ggsave(file.path(plot_dir, "bqp_convergence.pdf"),
       p_combined, width = 9, height = 10, device = "pdf")
cat("\nSaved: bqp_convergence.pdf\n")

# =============================================================================
# 10.  SAVE CSVs + CONVERGENCE TRACE RDS  (original behaviour preserved)
# =============================================================================
if (length(bo_traces) > 0) {
  trace_df <- do.call(rbind, lapply(bo_traces, function(r) {
    data.frame(method = r$method, rep = r$rep,
               iter   = seq_along(r$best_trace),
               best   = r$best_trace,
               stringsAsFactors = FALSE)
  }))
  saveRDS(trace_df, file.path(folder_name, "convergence_traces.rds"))
  cat("Saved: convergence_traces.rds\n")
}

write.csv(summary_df, file.path(folder_name, "results_individual.csv"), row.names = FALSE)
write.csv(agg,        file.path(folder_name, "results_summary.csv"),    row.names = FALSE)
cat("Saved CSVs to:", folder_name, "\n")