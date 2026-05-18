# =============================================================================
# LOCAL POST-PROCESSING — Constrained BQP Grid Search
# Sweeps penalty_w in {2, 5, 10} x n_init in {5, 10, 20}
# Run after all 180 array jobs complete:  Rscript c_bqp_grid_aggregate.R
#
# Produces:
#   plots/grid_bo_convergence.pdf   — DART-CBO traces, faceted by n_init,
#                                     coloured by penalty_w  (1 col × 3 rows)
#   plots/grid_ga_convergence.pdf   — GA generation traces, same facet layout
#   plots/grid_final_best_boxplot.pdf — final-best-obj boxplot, DART-CBO vs GA
#                                       faceted by n_init × penalty_w
# =============================================================================

library(ggplot2)
library(dplyr)
library(zoo)
library(patchwork)

folder_name <- "c_bqp_grid_results"
plot_dir    <- file.path(folder_name, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
# penalty_w levels: 2, 5, 10  →  sky-blue, amber, vermillion
OI <- c("2"  = "#56B4E9",
        "5"  = "#E69F00",
        "10" = "#D55E00")

# method colours (used in boxplot)
col_bo <- "#0072B2"
col_ga <- "#009E73"

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
    penalty_w   = r$penalty_w,
    n_init      = r$n_init,
    cfg_idx     = r$cfg_idx,
    best        = ifelse(is.null(r$best)        || is.na(r$best),        NA_real_, r$best),
    feasible    = ifelse(is.null(r$feasible)    || is.na(r$feasible),    NA,       r$feasible),
    budget_used = ifelse(is.null(r$budget_used) || is.na(r$budget_used), NA_real_, r$budget_used),
    evals       = ifelse(is.null(r$evals)       || is.na(r$evals),       NA_real_, r$evals),
    feas_rate   = ifelse(is.null(r$feas_rate)   || is.na(r$feas_rate),   NA_real_, r$feas_rate),
    elapsed_sec = r$elapsed_sec,
    stringsAsFactors = FALSE
  )
}))
summary_df <- summary_df[order(summary_df$method, summary_df$penalty_w,
                               summary_df$n_init,  summary_df$rep), ]

# Config label for facets
summary_df$pw_label   <- paste0("pw = ", summary_df$penalty_w)
summary_df$ni_label   <- paste0("n_init = ", summary_df$n_init)
summary_df$pw_f       <- factor(summary_df$penalty_w, levels = c(2, 5, 10))
summary_df$ni_f       <- factor(summary_df$n_init,    levels = c(5, 10, 20))

cat("Summary of loaded results:\n")
print(table(summary_df$method, summary_df$penalty_w, summary_df$n_init))
cat("\n")

# =============================================================================
# 3.  HELPER: build trace aggregation from a list of result objects
#     trace_field  — "best_trace" (BO) or "ga_trace" (GA)
#     x_label      — "iter" or "gen"
# =============================================================================
build_trace_agg <- function(res_list, trace_field, x_label) {
  do.call(rbind, lapply(res_list, function(r) {
    tr <- r[[trace_field]]
    if (is.null(tr) || length(tr) == 0) return(NULL)
    data.frame(
      penalty_w = r$penalty_w,
      n_init    = r$n_init,
      rep       = r$rep,
      x         = seq_along(tr),
      best      = tr,
      stringsAsFactors = FALSE
    )
  }))
}

# =============================================================================
# 4.  DART-CBO CONVERGENCE TRACES
# =============================================================================
bo_results <- Filter(function(r) !is.null(r$best_trace), results)

bo_raw <- build_trace_agg(bo_results, "best_trace", "iter")

bo_raw <- bo_raw |>
  group_by(penalty_w, n_init, rep) |>
  mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
  ungroup()

bo_agg <- bo_raw |>
  group_by(penalty_w, n_init, x) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    .groups   = "drop"
  ) |>
  mutate(
    pw_f = factor(penalty_w, levels = c(2, 5, 10)),
    ni_f = factor(n_init,    levels = c(5, 10, 20),
                  labels = c("n_init = 5", "n_init = 10", "n_init = 20"))
  )

p_bo_conv <- ggplot(bo_agg, aes(x = x, colour = pw_f, fill = pw_f)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              alpha = 0.18, colour = NA) +
  geom_line(aes(y = mean_best), linewidth = 0.85) +
  facet_wrap(~ ni_f, ncol = 1, scales = "free_y") +
  scale_colour_manual(
    name   = "Penalty w",
    values = OI,
    labels = c("2", "5", "10")
  ) +
  scale_fill_manual(
    name   = "Penalty w",
    values = OI,
    labels = c("2", "5", "10")
  ) +
  labs(
    title    = "DART-CBO — Best Feasible Objective vs. BO Iteration",
    subtitle = "Mean \u00b1 1 SD across 10 reps. Rows = n_init. Colours = penalty_w.",
    x        = "BO Iteration",
    y        = "Best Feasible Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = col_bo),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92"),
    strip.text       = element_text(face = "bold")
  )

ggsave(file.path(plot_dir, "grid_bo_convergence.pdf"),
       p_bo_conv, width = 9, height = 12, device = "pdf")
cat("Saved: grid_bo_convergence.pdf\n")

# =============================================================================
# 5.  GA CONVERGENCE TRACES
# =============================================================================
ga_results <- Filter(function(r) !is.null(r$ga_trace), results)

if (length(ga_results) == 0) {
  warning(paste(
    "No GA generation traces found (r$ga_trace is NULL in all GA results).",
    "Re-run the workers with the updated c_bqp_grid_worker.R."
  ))
} else {
  
  ga_raw <- build_trace_agg(ga_results, "ga_trace", "gen")
  
  ga_raw <- ga_raw |>
    group_by(penalty_w, n_init, rep) |>
    mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
    ungroup()
  
  ga_agg <- ga_raw |>
    group_by(penalty_w, n_init, x) |>
    summarise(
      mean_best = mean(best_filled, na.rm = TRUE),
      sd_best   = sd(best_filled,   na.rm = TRUE),
      n_obs     = sum(!is.na(best_filled)),
      .groups   = "drop"
    ) |>
    filter(n_obs > 0) |>
    mutate(
      pw_f = factor(penalty_w, levels = c(2, 5, 10)),
      ni_f = factor(n_init,    levels = c(5, 10, 20),
                    labels = c("n_init = 5", "n_init = 10", "n_init = 20"))
    )
  
  p_ga_conv <- ggplot(ga_agg, aes(x = x, colour = pw_f, fill = pw_f)) +
    geom_ribbon(aes(ymin = mean_best - sd_best,
                    ymax = mean_best + sd_best),
                alpha = 0.18, colour = NA) +
    geom_line(aes(y = mean_best), linewidth = 0.85) +
    facet_wrap(~ ni_f, ncol = 1, scales = "free_y") +
    scale_colour_manual(
      name   = "Penalty w",
      values = OI,
      labels = c("2", "5", "10")
    ) +
    scale_fill_manual(
      name   = "Penalty w",
      values = OI,
      labels = c("2", "5", "10")
    ) +
    labs(
      title    = "GA — Best Feasible Objective vs. Generation",
      subtitle = "Mean \u00b1 1 SD across 10 reps. Rows = n_init. Colours = penalty_w.",
      x        = "Generation",
      y        = "Best Feasible Objective (x\u2019Qx)"
    ) +
    theme_bw(base_size = 13) +
    theme(
      plot.title       = element_text(face = "bold", size = 13, colour = col_ga),
      plot.subtitle    = element_text(colour = "grey40", size = 10),
      legend.position  = "bottom",
      legend.title     = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92"),
      strip.text       = element_text(face = "bold")
    )
  
  ggsave(file.path(plot_dir, "grid_ga_convergence.pdf"),
         p_ga_conv, width = 9, height = 12, device = "pdf")
  cat("Saved: grid_ga_convergence.pdf\n")
}

# =============================================================================
# 6.  FINAL BEST OBJECTIVE — BOXPLOT  (DART-CBO vs GA)
#     Faceted by n_init (rows) x penalty_w (columns)
# =============================================================================
box_df <- summary_df[!is.na(summary_df$best), ]
box_df$method_f <- factor(box_df$method,
                          levels = c("BART-BO", "GA"),
                          labels = c("DART-CBO", "GA"))

p_box <- ggplot(box_df,
                aes(x = method_f, y = best,
                    fill = method_f, colour = method_f)) +
  geom_boxplot(alpha = 0.4, outlier.shape = 21,
               outlier.size = 1.8, linewidth = 0.6) +
  geom_jitter(width = 0.12, size = 1.5, alpha = 0.6) +
  facet_grid(ni_f ~ pw_f,
             labeller = labeller(
               ni_f = label_value,
               pw_f = function(x) paste0("pw = ", x)
             )) +
  scale_fill_manual(values   = c("DART-CBO" = col_bo, "GA" = col_ga),
                    guide    = "none") +
  scale_colour_manual(values = c("DART-CBO" = col_bo, "GA" = col_ga),
                      guide  = "none") +
  labs(
    title    = "Final Best Feasible Objective: DART-CBO vs GA",
    subtitle = "Each panel = one (n_init, penalty_w) config. Points = individual reps.",
    x        = NULL,
    y        = "Best Feasible Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92"),
    strip.text       = element_text(face = "bold", size = 10),
    axis.text.x      = element_text(size = 10)
  )

ggsave(file.path(plot_dir, "grid_final_best_boxplot.pdf"),
       p_box, width = 10, height = 9, device = "pdf")
cat("Saved: grid_final_best_boxplot.pdf\n")

# =============================================================================
# 7.  PRINT SUMMARY TABLE
# =============================================================================
cat("\n── Mean final best feasible objective ──\n")
summary_df |>
  filter(!is.na(best)) |>
  group_by(method, penalty_w, n_init) |>
  summarise(
    mean_best  = round(mean(best),  4),
    sd_best    = round(sd(best),    4),
    feas_n     = sum(!is.na(best)),
    mean_evals = round(mean(evals, na.rm = TRUE)),
    .groups    = "drop"
  ) |>
  arrange(method, penalty_w, n_init) |>
  as.data.frame() |>
  print()