# -------------------------------------------------------------
# BQP Results: Convergence Plot + LaTeX Summary Table
# Requires: ggplot2, dplyr, xtable
# Run AFTER bqp_aggregate.R has completed
# -------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(xtable)

folder_name <- "BART-CBO/Unconstrained/BQP/bqp_results"

# ── 1. Load data ──────────────────────────────────────────────

# Aggregated per-rep summary (for xtable)
summary_df <- read.csv(file.path(folder_name, "results_individual.csv"),
                       stringsAsFactors = FALSE)

# Aggregated method-level summary (for xtable)
agg <- read.csv(file.path(folder_name, "results_summary.csv"),
                stringsAsFactors = FALSE)

# Convergence traces
trace_path <- file.path(folder_name, "convergence_traces.rds")
if (!file.exists(trace_path)) stop("convergence_traces.rds not found. Run bqp_aggregate.R first.")
trace_df <- readRDS(trace_path)

# ── 2. Convergence trace plot ─────────────────────────────────

# Compute per-method mean & ribbon (mean ± 1 SD) across reps at each iteration
trace_summary <- trace_df %>%
  group_by(method, iter) %>%
  summarise(
    mean_best = mean(best),
    sd_best   = sd(best),
    .groups   = "drop"
  ) %>%
  mutate(
    ymin = mean_best - sd_best,
    ymax = mean_best + sd_best
  )

# Only DART-BO and BART-BO in trace_summary (no changes needed there)
trace_summary$method <- factor(trace_summary$method, levels = c("DART-BO", "BART-BO"))

ga_mean <- agg$Mean_best[agg$Method == "Pure-GA"]
ga_sd   <- agg$SD_best[agg$Method == "Pure-GA"]

p_trace <- ggplot(trace_summary, aes(x = iter, colour = method)) +
  # GA SD band
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = ga_mean - ga_sd, ymax = ga_mean + ga_sd,
           fill = "#CC79A7", alpha = 0.20) +
  # Draw each ribbon separately so fill colour is unambiguous
  geom_ribbon(data = subset(trace_summary, method == "DART-BO"),
              aes(ymin = ymin, ymax = ymax, fill = method),
              alpha = 0.20, colour = NA) +
  geom_ribbon(data = subset(trace_summary, method == "BART-BO"),
              aes(ymin = ymin, ymax = ymax, fill = method),
              alpha = 0.20, colour = NA) +
  geom_line(aes(y = mean_best), linewidth = 0.9) +
  geom_hline(aes(yintercept = ga_mean, colour = "Pure-GA"),
             linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(
    name   = "Method",
    values = c("Pure-GA" = "#CC79A7", "DART-BO" = "#56B4E9", "BART-BO" = "#009E73"),
    labels = c("Pure-GA" = "GA", "DART-BO" = "Dirichlet BART-BO", "BART-BO" = "Standard BART-BO")
  ) +
  scale_fill_manual(
    name   = "Method",
    values = c("Pure-GA" = "#CC79A7", "DART-BO" = "#56B4E9", "BART-BO" = "#009E73"),
    labels = c("Pure-GA" = "GA", "DART-BO" = "Dirichlet BART-BO", "BART-BO" = "Standard BART-BO"),
    guide  = "none"  
  ) +
  labs(
    title    = "BQP Convergence Traces",
    subtitle = "Mean best objective ± 1 SD across replications. Pure-GA shown as final mean ± 1 SD (dashed).",
    x        = "Iteration",
    y        = "Best Objective Value"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14),
    plot.subtitle    = element_text(colour = "grey40", size = 11),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )
ggsave(
  filename = file.path(folder_name, "bqp_convergence_traces.pdf"),
  plot     = p_trace,
  width    = 8, height = 5, device = "pdf"
)

cat("Saved convergence plot to:", folder_name, "\n")

# ── 3. LaTeX summary table via xtable ─────────────────────────

# Rename columns to clean LaTeX-friendly labels
# Convert time to minutes and round all values to 2 decimal places
agg_tex <- agg %>%
  mutate(Mean_sec = round(Mean_sec / 60, 2)) %>%   # seconds -> minutes
  rename(
    "Method"          = Method,
    "Reps"            = N_reps,
    "Mean Obj."       = Mean_best,
    "SD Obj."         = SD_best,
    "Min Obj."        = Min_best,
    "Max Obj."        = Max_best,
    "Mean Evals"      = Mean_eval,
    "Mean Time (min)" = Mean_sec    # updated label
  )

xt <- xtable(
  agg_tex,
  caption = "BQP benchmark: per-method summary over replications.
             Bold values indicate the best (lowest) mean objective.",
  label   = "tab:bqp_summary",
  digits  = c(0, 0, 0, 2, 2, 2, 2, 2, 2)  # 0 = row index, then 2 d.p. for all columns
)

# Mark the best (minimum) Mean Obj. row
best_idx <- which.min(agg_tex[["Mean Obj."]])

atr <- list()
atr$pos     <- list(best_idx - 1)
atr$command <- c("\\rowcolor{gray!15}")

print(
  xt,
  file                   = file.path(folder_name, "bqp_summary_table.tex"),
  include.rownames       = FALSE,
  booktabs               = TRUE,
  caption.placement      = "top",
  add.to.row             = atr,
  sanitize.text.function = identity,
  comment                = FALSE
)
cat("Saved LaTeX table to:", file.path(folder_name, "bqp_summary_table.tex"), "\n")

# ── 4. Preview table in console ───────────────────────────────
cat("\n========================================\n")
cat("SUMMARY TABLE PREVIEW\n")
cat("========================================\n\n")
print(agg, row.names = FALSE)
