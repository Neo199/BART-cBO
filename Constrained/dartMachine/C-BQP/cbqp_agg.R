# =============================================================================
# LOCAL POST-PROCESSING — Constrained BQP (DART-CBO vs GA)
# Run after all array jobs complete:  Rscript c_bqp_aggregate.R
# =============================================================================

library(ggplot2)
library(dplyr)
library(zoo)
library(patchwork)

folder_name <- "c_bqp_result_n100"
plot_dir    <- file.path(folder_name, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# =============================================================================
# 1.  LOAD ALL RESULT FILES
# =============================================================================
rds_files <- list.files(folder_name, pattern = "\\.rds$", full.names = TRUE)
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
    best        = ifelse(is.null(r$best)        || is.na(r$best),        NA_real_, r$best),
    feasible    = ifelse(is.null(r$feasible)    || is.na(r$feasible),    NA,       r$feasible),
    budget_used = ifelse(is.null(r$budget_used) || is.na(r$budget_used), NA_real_, r$budget_used),
    evals       = ifelse(is.null(r$evals)       || is.na(r$evals),       NA_real_, r$evals),
    feas_rate   = ifelse(is.null(r$feas_rate)   || is.na(r$feas_rate),   NA_real_, r$feas_rate),
    elapsed_sec = r$elapsed_sec,
    stringsAsFactors = FALSE
  )
}))
summary_df <- summary_df[order(summary_df$method, summary_df$rep), ]

# =============================================================================
# 3.  COLORBLIND-SAFE PALETTE  (Okabe-Ito)
# =============================================================================
col_dartcbo <- "#0072B2"   # blue
col_ga      <- "#E69F00"   # amber

# =============================================================================
# 4.  PLOT 1 (TOP) — DART-CBO convergence trace
#     best_trace: length = n_iter, NA until first feasible found, then filled
# =============================================================================
bo_traces <- Filter(function(r) !is.null(r$best_trace), results)

bo_trace_df <- do.call(rbind, lapply(bo_traces, function(r) {
  data.frame(
    rep  = r$rep,
    iter = seq_along(r$best_trace),
    best = r$best_trace,
    stringsAsFactors = FALSE
  )
}))

# Forward-fill within each rep so ribbon covers all iterations once feasible found
bo_trace_df <- bo_trace_df |>
  group_by(rep) |>
  mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
  ungroup()

bo_agg <- bo_trace_df |>
  group_by(iter) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    .groups   = "drop"
  )

ga_traces <- Filter(function(r) !is.null(r$ga_trace), results)

if (length(ga_traces) == 0) {
  stop(paste(
    "No GA generation traces found (r$ga_trace is NULL in all GA results).",
    "Re-run the workers with the updated c_bqp_worker.R that saves ga_trace."
  ))
}

ga_trace_df_raw <- do.call(rbind, lapply(ga_traces, function(r) {
  trace <- r$ga_trace
  data.frame(
    rep  = r$rep,
    gen  = seq_along(trace),
    best = trace,
    stringsAsFactors = FALSE
  )
}))

# Forward-fill within each rep (generations before first feasible remain NA,
# ribbon will only cover generations where at least one rep has a value)
ga_trace_df_raw <- ga_trace_df_raw |>
  group_by(rep) |>
  mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
  ungroup()

ga_agg <- ga_trace_df_raw |>
  group_by(gen) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    n_obs     = sum(!is.na(best_filled)),
    .groups   = "drop"
  ) |>
  filter(n_obs > 0)   # drop leading gens where no rep has a feasible best yet

# =============================================================================
# SHARED Y-AXIS LIMITS  (computed across both BO and GA aggregates)
# =============================================================================
y_min <- min(
  bo_agg$mean_best - bo_agg$sd_best,
  ga_agg$mean_best - ga_agg$sd_best,
  na.rm = TRUE
)
y_max <- max(
  bo_agg$mean_best + bo_agg$sd_best,
  ga_agg$mean_best + ga_agg$sd_best,
  na.rm = TRUE
)
y_pad  <- 0.04 * (y_max - y_min)   # 4% breathing room
y_lims <- c(y_min - y_pad, y_max + y_pad)

# Shared y-axis breaks: interval of 5 across the union of both traces
y_all    <- c(bo_agg$mean_best - bo_agg$sd_best,
              bo_agg$mean_best + bo_agg$sd_best,
              ga_agg$mean_best - ga_agg$sd_best,
              ga_agg$mean_best + ga_agg$sd_best)
y_range  <- range(y_all, na.rm = TRUE)
y_breaks <- seq(floor(y_range[1] / 5) * 5,
                ceiling(y_range[2] / 5) * 5,
                by = 5)

p_bo <- ggplot(bo_agg, aes(x = iter)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              fill = col_dartcbo, alpha = 0.25) +
  geom_line(aes(y = mean_best),
            colour = col_dartcbo, linewidth = 0.9) +
  scale_y_continuous(breaks = y_breaks) +
  labs(
    title    = "BART-CBO — Best Feasible Objective vs. BO Iteration",
    subtitle = "Mean \u00b1 1 SD across 10 reps (ribbon). Carried forward from first feasible solution.",
    x        = "BO Iteration",
    y        = "Best Feasible Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  coord_cartesian(ylim = y_lims) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = col_dartcbo),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    panel.grid.minor = element_blank()
  )

# =============================================================================
# 5.  PLOT 2 (BOTTOM) — GA per-generation best feasible objective
#     ga_trace: length = n_generations, NA for generations with no feasible best
# =============================================================================

p_ga <- ggplot(ga_agg, aes(x = gen)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              fill = col_ga, alpha = 0.25) +
  geom_line(aes(y = mean_best),
            colour = col_ga, linewidth = 0.9) +
  scale_y_continuous(breaks = y_breaks) +
  labs(
    title    = "GA — Best Feasible Objective vs. Generation",
    subtitle = "Mean \u00b1 1 SD across 10 reps (ribbon). Carried forward from first feasible generation.",
    x        = "Generation",
    y        = "Best Feasible Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  coord_cartesian(ylim = y_lims) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = col_ga),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    panel.grid.minor = element_blank()
  )

# =============================================================================
# 6.  COMBINE INTO 1-COL × 2-ROW GRID AND SAVE
# =============================================================================
p_combined <- p_bo / p_ga +
  plot_annotation(
    title   = "Constrained BQP — BART-CBO vs GA",
    theme   = theme(
      plot.title   = element_text(face = "bold", size = 15, hjust = 0.5),
      plot.caption = element_text(colour = "grey50", size = 9)
    )
  )

ggsave(
  file.path(plot_dir, "c_bqp_combined.pdf"),
  p_combined,
  width  = 9,
  height = 10,
  device = "pdf"
)

cat("Saved combined plot to:", file.path(plot_dir, "c_bqp_combined.pdf"), "\n")
