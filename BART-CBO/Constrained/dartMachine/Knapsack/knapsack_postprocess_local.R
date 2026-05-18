# =============================================================================
# LOCAL POST-PROCESSING — Constrained BART-BO Knapsack Experiments
# Run this on your local machine after downloading knapsack_results_GA/
#
## TRUE OPTIMUM (known solution)
## true_optimum <- 13549094
# =============================================================================

library(ggplot2)
library(dplyr)
library(zoo)
library(patchwork)
library(scales)

base_dir        <- "BART-CBO/Constrained/dartMachine/Knapsack/knapsack_results_GA"
penalty_weights <- c(0.1, 1, 2, 5, 10)
n_instances     <- 10
true_optimum    <- 13549094   # GLPK exact solution

# =============================================================================
# 1.  COLLECT ALL PER-TASK SUMMARIES
# =============================================================================
cat("Collecting per-task summaries...\n")

all_summaries <- lapply(penalty_weights, function(pw) {
  pw_dir <- file.path(base_dir, sprintf("penalty_%.1f", pw))
  rows   <- lapply(seq_len(n_instances), function(inst) {
    rds_path <- file.path(pw_dir, sprintf("summary_instance_%02d.rds", inst))
    csv_path <- file.path(pw_dir, sprintf("summary_instance_%02d.csv", inst))
    if (file.exists(rds_path)) {
      readRDS(rds_path)
    } else if (file.exists(csv_path)) {
      read.csv(csv_path)
    } else {
      warning(sprintf("Missing results: pw=%.1f  inst=%d", pw, inst))
      NULL
    }
  })
  do.call(rbind, Filter(Negate(is.null), rows))
})

summary_df <- do.call(rbind, Filter(Negate(is.null), all_summaries))
summary_df$penalty_weight <- factor(summary_df$penalty_weight,
                                    levels = sort(unique(summary_df$penalty_weight)))

cat(sprintf("Loaded %d / %d task summaries\n",
            nrow(summary_df), length(penalty_weights) * n_instances))

write.csv(summary_df,
          file.path(base_dir, "all_summaries.csv"),
          row.names = FALSE)
cat("Saved: all_summaries.csv\n\n")

# =============================================================================
# 2.  AGGREGATED TABLE (mean ± SD per penalty weight)
# =============================================================================
cat("Computing aggregated statistics...\n")

agg_df <- summary_df |>
  group_by(penalty_weight) |>
  summarise(
    n_tasks                  = n(),
    n_with_feasible          = sum(!is.na(bart_best)),
    bart_best_mean           = mean(bart_best,            na.rm = TRUE),
    bart_best_sd             = sd(bart_best,              na.rm = TRUE),
    ga_best_mean             = mean(ga_best,              na.rm = TRUE),
    ga_best_sd               = sd(ga_best,                na.rm = TRUE),
    optimality_gap_mean      = mean(optimality_gap,       na.rm = TRUE),
    optimality_gap_sd        = sd(optimality_gap,         na.rm = TRUE),
    feasibility_pct_mean     = mean(feasibility_pct,      na.rm = TRUE),
    feasibility_pct_sd       = sd(feasibility_pct,        na.rm = TRUE),
    bart_true_evals_mean     = mean(bart_true_eval_count, na.rm = TRUE),
    bart_true_evals_sd       = sd(bart_true_eval_count,   na.rm = TRUE),
    ga_true_evals_mean       = mean(ga_true_eval_count,   na.rm = TRUE),
    ga_true_evals_sd         = sd(ga_true_eval_count,     na.rm = TRUE),
    bart_wall_sec_mean       = mean(bart_wall_sec,        na.rm = TRUE),
    .groups = "drop"
  )

write.csv(agg_df,
          file.path(base_dir, "comparison_aggregated.csv"),
          row.names = FALSE)

cat("\n--- AGGREGATED RESULTS ---\n")
print(as.data.frame(agg_df), digits = 3)

# =============================================================================
# 3.  PLOTS
# =============================================================================
plot_dir <- file.path(base_dir, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# Okabe-Ito palette — penalty weights + references
cb_palette <- c(
  "Penalty = 0.1"       = "#E69F00",
  "Penalty = 1"         = "#56B4E9",
  "Penalty = 2"         = "#009E73",
  "Penalty = 5"         = "#CC79A7",
  "Penalty = 10"        = "#0072B2",
  "True Optimum (GLPK)" = "#000000",
  "GA"                  = "#D55E00"
)

# =============================================================================
# PLOT 1 (TOP) — build conv_agg FIRST
# =============================================================================
cat("\nLoading iteration-level data for BART-CBO convergence curves...\n")

iter_data <- lapply(penalty_weights, function(pw) {
  pw_dir <- file.path(base_dir, sprintf("penalty_%.1f", pw))
  inst_frames <- lapply(seq_len(n_instances), function(inst) {
    fp <- file.path(pw_dir, sprintf("instance_%02d.csv", inst))
    if (!file.exists(fp)) return(NULL)
    df <- read.csv(fp)
    df$iteration      <- seq_len(nrow(df))
    df$penalty_weight <- pw
    df$instance       <- inst
    df$run_best <- {
      best <- NA_real_
      sapply(seq_len(nrow(df)), function(i) {
        if (!is.na(df$feasible[i]) && as.logical(df$feasible[i])) {
          if (is.na(best) || df$y[i] > best) best <<- df$y[i]
        }
        best
      })
    }
    df[, c("iteration", "penalty_weight", "instance", "run_best")]
  })
  do.call(rbind, Filter(Negate(is.null), inst_frames))
})

iter_df <- do.call(rbind, Filter(Negate(is.null), iter_data))
iter_df$penalty_weight <- factor(iter_df$penalty_weight,
                                 levels = sort(unique(iter_df$penalty_weight)))

iter_df <- iter_df |>
  group_by(penalty_weight, instance) |>
  mutate(run_best = zoo::na.locf(run_best, na.rm = FALSE)) |>
  ungroup()

conv_agg <- iter_df |>
  group_by(penalty_weight, iteration) |>
  summarise(mean_best = mean(run_best, na.rm = TRUE),
            sd_best   = sd(run_best,   na.rm = TRUE),
            .groups   = "drop") |>
  mutate(method = factor(paste("Penalty =", as.character(penalty_weight)),
                         levels = names(cb_palette)))

# =============================================================================
# PLOT 2 (BOTTOM) — build ga_agg SECOND
# =============================================================================
cat("Loading GA generation traces...\n")

ga_trace_dir <- "BART-CBO/Constrained/dartMachine/Knapsack/knapsack_ga_traces"
ga_pw_dir    <- file.path(ga_trace_dir, "penalty_0.1")

ga_trace_df <- do.call(rbind, lapply(seq_len(n_instances), function(inst) {
  fp <- file.path(ga_pw_dir, sprintf("ga_trace_instance_%02d.csv", inst))
  if (!file.exists(fp)) {
    warning(sprintf("Missing GA trace: %s", fp))
    return(NULL)
  }
  read.csv(fp)
}))

if (is.null(ga_trace_df) || nrow(ga_trace_df) == 0) {
  stop("No GA trace files found.")
}

ga_trace_df <- ga_trace_df |>
  group_by(instance) |>
  mutate(best_filled = zoo::na.locf(best_feasible, na.rm = FALSE)) |>
  ungroup()

ga_agg <- ga_trace_df |>
  group_by(generation) |>
  summarise(
    mean_best = mean(best_filled, na.rm = TRUE),
    sd_best   = sd(best_filled,   na.rm = TRUE),
    n_obs     = sum(!is.na(best_filled)),
    .groups   = "drop"
  ) |>
  filter(n_obs > 0)

# =============================================================================
# SHARED Y-AXIS — NOW both conv_agg and ga_agg exist
# =============================================================================
shared_y_all <- c(
  conv_agg$mean_best - conv_agg$sd_best,
  conv_agg$mean_best + conv_agg$sd_best,
  ga_agg$mean_best   - ga_agg$sd_best,
  ga_agg$mean_best   + ga_agg$sd_best,
  true_optimum
)
shared_y_rng  <- range(shared_y_all, na.rm = TRUE)
shared_breaks <- seq(
  floor(shared_y_rng[1]   / 5e5) * 5e5,
  ceiling(shared_y_rng[2] / 5e5) * 5e5,
  by = 5e5
)
shared_limits <- c(shared_y_rng[1] - 1e4, shared_y_rng[2] + 1e4)

# =============================================================================
# BUILD p_bo — using shared scale INSIDE the ggplot call
# =============================================================================
p_bo <- ggplot(conv_agg,
               aes(x = iteration, colour = method, fill = method)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              alpha = 0.18, colour = NA) +
  geom_line(aes(y = mean_best), linewidth = 0.9) +
  geom_hline(aes(yintercept = true_optimum, colour = "True Optimum (GLPK)"),
             linetype = "solid", linewidth = 0.8) +
  scale_colour_manual(name = "Method", values = cb_palette) +
  scale_fill_manual(name   = "Method", values = cb_palette, guide = "none") +
  scale_y_continuous(breaks = shared_breaks, limits = shared_limits,   # <-- shared
                     labels = scales::comma) +
  labs(
    title    = "BART-CBO — Best Feasible Objective vs. BO Iteration",
    subtitle = "Mean \u00b1 1 SD across 10 instances (ribbon). Carried forward from first feasible solution.",
    x        = "BO Iteration",
    y        = "Best Feasible Objective Value"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = "#0072B2"),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# =============================================================================
# BUILD p_ga — using shared scale INSIDE the ggplot call
# =============================================================================

p_ga <- ggplot(ga_agg, aes(x = generation)) +
  geom_ribbon(aes(ymin = mean_best - sd_best,
                  ymax = mean_best + sd_best),
              fill = "#D55E00", alpha = 0.25) +
  geom_line(aes(y = mean_best, colour = "GA"), linewidth = 0.9) +
  geom_hline(aes(yintercept = true_optimum, colour = "True Optimum (GLPK)"),   # legend entry
             linetype = "solid", linewidth = 0.8) +
  scale_colour_manual(name = "Method", values = cb_palette) +
  scale_y_continuous(breaks = shared_breaks, limits = shared_limits,
                     labels = scales::comma) +
  labs(
    title    = "GA — Best Feasible Objective vs. Generation",
    subtitle = "Mean \u00b1 1 SD across 10 instances (ribbon). Carried forward from first feasible generation.",
    x        = "Generation",
    y        = "Best Feasible Objective Value"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, colour = "#D55E00"),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )
# =============================================================================
# COMBINE AND SAVE
# =============================================================================
p_combined <- (p_bo / p_ga) +
  plot_annotation(
    title   = "Knapsack — BART-CBO vs GA",
    theme   = list(                          # <-- pass a named list, not theme()
      plot.title   = element_text(face = "bold", size = 15, hjust = 0.5),
      plot.caption = element_text(colour = "grey50", size = 9)
    )
  )

ggsave(file.path(plot_dir, "knapsack_convergence_curves.pdf"),
       p_combined, width = 9, height = 10, device = "pdf")
cat("Saved: knapsack_convergence_curves.pdf\n")

# =============================================================================
# 4.  PRINT FINAL INTERPRETATION TABLE
# =============================================================================
cat("\n=============================================================================\n")
cat("FINAL SUMMARY TABLE\n")
cat("=============================================================================\n")
for (i in seq_len(nrow(agg_df))) {
  cat(sprintf("\nPenalty Weight %.1f:\n",    agg_df$penalty_weight[i]))
  cat(sprintf("  Feasibility:              %.1f%% ± %.1f%%\n",
              agg_df$feasibility_pct_mean[i], agg_df$feasibility_pct_sd[i]))
  cat(sprintf("  BART-BO Objective:        %.0f ± %.0f\n",
              agg_df$bart_best_mean[i],       agg_df$bart_best_sd[i]))
  cat(sprintf("  Optimality Gap:           %.2f%% ± %.2f%%\n",
              agg_df$optimality_gap_mean[i],  agg_df$optimality_gap_sd[i]))
  cat(sprintf("  BART-BO true evals:       %.1f ± %.1f\n",
              agg_df$bart_true_evals_mean[i], agg_df$bart_true_evals_sd[i]))
  cat(sprintf("  GA true evals:            %.1f ± %.1f\n",
              agg_df$ga_true_evals_mean[i],   agg_df$ga_true_evals_sd[i]))
  cat(sprintf("  BART-BO wall time (mean): %.0f s\n",
              agg_df$bart_wall_sec_mean[i]))
  cat(sprintf("  Instances with feasible:  %d/%d\n",
              agg_df$n_with_feasible[i], n_instances))
}

cat("\n=============================================================================\n")
cat("POST-PROCESSING COMPLETE\n")
cat("=============================================================================\n")
cat(sprintf("Output files:\n"))
cat(sprintf("  %s/all_summaries.csv\n",         base_dir))
cat(sprintf("  %s/comparison_aggregated.csv\n", base_dir))
cat(sprintf("  %s/plots/\n",                    base_dir))

# =============================================================================
# 5.  LATEX TABLE
# =============================================================================
library(xtable)

ga_mean_global <- mean(summary_df$ga_best, na.rm = TRUE)

table_df <- agg_df |>
  mutate(
    pw              = as.numeric(as.character(penalty_weight)),
    bart_cell       = paste0(
      round(bart_best_mean / 1e6, 2), " $\\pm$ ",
      round(bart_best_sd   / 1e6, 2), "$\\times 10^6$"),
    feas_cell       = paste0(
      round(feasibility_pct_mean, 1), " $\\pm$ ",
      round(feasibility_pct_sd,   1)),
    ga_cell         = paste0(round(ga_mean_global / 1e6, 2), "$\\times 10^6$"),
    gap_true        = round((true_optimum - bart_best_mean) / true_optimum * 100, 2),
    gap_ga          = round((ga_mean_global - bart_best_mean) / ga_mean_global * 100, 2),
    bart_evals_cell = paste0(
      round(bart_true_evals_mean, 0), " $\\pm$ ",
      round(bart_true_evals_sd,   0)),
    ga_evals_cell   = paste0(
      round(ga_true_evals_mean, 0), " $\\pm$ ",
      round(ga_true_evals_sd,   0))
  ) |>
  select(
    `Penalty weight`      = pw,
    `BART-BO optimum`     = bart_cell,
    `Feasible (% +/- sd)` = feas_cell,
    `GA mean`             = ga_cell,
    `% gap from true`     = gap_true,
    `% gap from GA`       = gap_ga,
    `BART-BO evals`       = bart_evals_cell,
    `GA evals`            = ga_evals_cell
  )

cat("\n=============================================================================\n")
cat("RESULTS TABLE\n")
cat("=============================================================================\n\n")
print(as.data.frame(table_df), row.names = FALSE)

xt <- xtable(
  table_df,
  caption = "Knapsack results across penalty weights. True optimum (GLPK) = 13,549,094.",
  label   = "tab:knapsack_results",
  align   = c("r", "r", "c", "c", "c", "c", "c", "c", "c")
)
print(xt, sanitize.text.function = identity)