# =============================================================================
# LOCAL POST-PROCESSING — Constrained BQP (DART-CBO vs GA)
# Run after all array jobs complete:  Rscript c_bqp_aggregate.R
# =============================================================================

library(ggplot2)
library(dplyr)
library(zoo)

folder_name <- "BART-CBO/Constrained/dartMachine/C-BQP/c_bqp_result_n50"
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
# 3.  CONVERGENCE TRACES  (DART-CBO only — GA has no trace)
# =============================================================================
traces <- Filter(function(r) !is.null(r$best_trace), results)

trace_df <- do.call(rbind, lapply(traces, function(r) {
  data.frame(
    method = r$method,
    rep    = r$rep,
    iter   = seq_along(r$best_trace),
    best   = r$best_trace,
    stringsAsFactors = FALSE
  )
}))

# Forward-fill NAs (no feasible found yet at that iteration)
trace_df <- trace_df |>
  group_by(method, rep) |>
  mutate(best_filled = zoo::na.locf(best, na.rm = FALSE)) |>
  ungroup()

conv_agg <- trace_df |>
  group_by(iter) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    .groups   = "drop"
  )

# =============================================================================
# 4.  GA REFERENCE STATS
# =============================================================================
ga_df   <- summary_df[summary_df$method == "Penalty-GA" & !is.na(summary_df$best), ]
ga_mean <- mean(ga_df$best, na.rm = TRUE)
ga_sd   <- sd(ga_df$best,   na.rm = TRUE)

# =============================================================================
# 5.  COLOUR / STYLE PALETTE  (matches knapsack script)
# =============================================================================
col_dartcbo <- "#0072B2"   # blue  — matches "Penalty = 10" knapsack anchor
col_ga      <- "#D55E00"   # orange-red — identical to knapsack GA colour

# =============================================================================
# PLOT 1 — Convergence trace: mean ± 1 SD  +  GA reference
# =============================================================================
p_conv <- ggplot(conv_agg, aes(x = iter)) +
  # GA ±1 SD band
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = ga_mean - ga_sd,
           ymax = ga_mean + ga_sd,
           fill = col_ga, alpha = 0.15) +
  # DART-CBO ±1 SD ribbon
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              fill = col_dartcbo, alpha = 0.20) +
  # DART-CBO mean line
  geom_line(aes(y = mean_best, colour = "DART-CBO"),
            linewidth = 0.9) +
  # GA mean dashed line
  geom_hline(aes(colour = "GA"),
             yintercept = ga_mean,
             linetype   = "dashed",
             linewidth  = 0.8) +
  scale_colour_manual(
    name   = "Method",
    values = c("DART-CBO" = col_dartcbo,
               "GA"       = col_ga)
  ) +
  labs(
    title    = "Constrained BQP Convergence Trace",
    subtitle = "Mean best feasible objective \u00b1 1 SD across 10 reps. GA shown as final mean \u00b1 1 SD (dashed).",
    x        = "BO Iteration",
    y        = "Best Feasible Objective (x\u2019Qx)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14),
    plot.subtitle    = element_text(colour = "grey40", size = 11),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# =============================================================================
# PLOT 2 — Evaluation count comparison: DART-CBO vs GA
# =============================================================================
eval_df <- summary_df[!is.na(summary_df$evals), ]
eval_df$label <- ifelse(eval_df$method == "BART-BO", "DART-CBO", "GA")

eval_agg <- eval_df |>
  group_by(label) |>
  summarise(
    mean_evals = mean(evals),
    sd_evals   = sd(evals),
    .groups    = "drop"
  )

p_evals <- ggplot(eval_agg,
                  aes(x = label, y = mean_evals, fill = label)) +
  geom_col(width = 0.5, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_evals - sd_evals,
                    ymax = mean_evals + sd_evals),
                width = 0.15, linewidth = 0.7) +
  geom_jitter(data = eval_df,
              aes(x = label, y = evals, fill = label),
              shape = 21, width = 0.08, size = 2, alpha = 0.7,
              inherit.aes = FALSE) +
  scale_fill_manual(values = c("DART-CBO" = col_dartcbo,
                               "GA"       = col_ga),
                    guide  = "none") +
  labs(
    title    = "True Objective Evaluations: DART-CBO vs GA",
    subtitle = "Bar = mean \u00b1 1 SD across 10 reps. Points = individual reps.",
    x        = NULL,
    y        = "Number of f(x) evaluations"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14),
    plot.subtitle    = element_text(colour = "grey40", size = 11),
    legend.position  = "none",
    panel.grid.minor = element_blank()
  )

# =============================================================================
# SAVE BOTH PLOTS TO A SINGLE PDF
# =============================================================================
ggsave(file.path(plot_dir, "c_bqp_convergence.pdf"),
       p_conv,  width = 9, height = 5.5, device = "pdf")

ggsave(file.path(plot_dir, "c_bqp_evalcount.pdf"),
       p_evals, width = 9, height = 5.5, device = "pdf")