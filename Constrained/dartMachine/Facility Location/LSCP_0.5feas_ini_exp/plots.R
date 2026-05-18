# =============================================================================
# LSCP DART-CBO — Objective Trace + OSM Gradient Maps + LaTeX Summary Table
#
# Produces:
#   1. plots/lscp_objective_trace.pdf      — f_best mean ± SD over BO iterations
#   2. plots/lscp_map_ini5.pdf             — OSM gradient map, n_init = 5
#   3. plots/lscp_map_ini10.pdf            — OSM gradient map, n_init = 10
#   4. plots/lscp_map_ini20.pdf            — OSM gradient map, n_init = 20
#   5. plots/lscp_maps_combined.pdf        — all 3 maps side by side (30 x 10 in)
#   6. plots/lscp_summary_table.tex        — xtable LaTeX summary
#
# Map gradient logic
#   X_all from each instance_XX.rds contains EVERY evaluated solution (not
#   just the best). All rows across 10 instances are pooled per n_init group.
#   Facility gradient = proportion of pooled rows where that facility is selected.
#   Demand gradient   = proportion of pooled rows where that demand point is covered.
#   This mirrors the MCLP gradient approach but uses the full evaluation pool
#   rather than one best-x per run.
#
# Colorblind-safe palette: Okabe-Ito throughout (same as MCLP code).
# =============================================================================

library(sf)
library(ggplot2)
library(osmdata)
library(dplyr)
library(tidyr)
library(patchwork)
library(xtable)

# ---- Paths ------------------------------------------------------------------
base_dir    <- "BART-CBO/Constrained/dartMachine/Facility Location/LSCP_0.5feas_ini_exp/ini_sample_set"
data_dir    <- "PhD-Data/SF_data"
out_dir     <- "BART-CBO/Constrained/dartMachine/Facility Location/LSCP_0.5feas_ini_exp/plots"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

ini_values  <- c(5, 10, 20)
n_instances <- 10

# ---- Okabe-Ito palette ------------------------------------------------------
okabe_ito    <- c("#000000","#E69F00","#56B4E9","#009E73",
                  "#F0E442","#0072B2","#D55E00","#CC79A7")
demand_pal   <- c(okabe_ito[7], okabe_ito[5], okabe_ito[4])  # vermillion→yellow→bl-green
facility_pal <- c("#FFFFFF",    okabe_ito[3], okabe_ito[6])  # white→sky blue→blue
trace_pal    <- c(okabe_ito[7], okabe_ito[6], okabe_ito[4])  # vermillion, blue, bl-green (Okabe-Ito)

# =============================================================================
# 1.  Load spatial / problem data  (once)
# =============================================================================
demand_data     <- read.csv(file.path(data_dir,
                                      "SF_demand_205_centroid_uniform_weight.csv"))
facility_loc    <- read.csv(file.path(data_dir,
                                      "SF_store_site_16_longlat.csv"))
distance_matrix <- read.csv(file.path(data_dir,
                                      "SF_network_distance_candidateStore_16_censusTract_205_new.csv"))

distance_matrix$covered <- as.integer(distance_matrix$distance <= 5000)

A_df <- distance_matrix %>%
  dplyr::select(DestinationName, name, covered) %>%
  tidyr::pivot_wider(names_from  = name,
                     values_from = covered,
                     values_fill = list(covered = 0))

A         <- as.matrix(A_df[, -1])
n_demand  <- nrow(A)
n_vars    <- ncol(A)

cat(sprintf("Problem: %d demand points, %d candidate facilities\n\n",
            n_demand, n_vars))

# =============================================================================
# 2.  Shared bounding box + OSM streets  (fetched once)
# =============================================================================
all_pts  <- st_union(
  st_geometry(st_as_sf(demand_data,  coords = c("long", "lat"), crs = 4326)),
  st_geometry(st_as_sf(facility_loc, coords = c("long", "lat"), crs = 4326))
)
bbox_raw <- st_bbox(all_pts)
pad_x    <- (bbox_raw["xmax"] - bbox_raw["xmin"]) * 0.05
pad_y    <- (bbox_raw["ymax"] - bbox_raw["ymin"]) * 0.05
bbox_exp <- bbox_raw
bbox_exp[c("xmin","xmax")] <- bbox_exp[c("xmin","xmax")] + c(-pad_x,  pad_x)
bbox_exp[c("ymin","ymax")] <- bbox_exp[c("ymin","ymax")] + c(-pad_y,  pad_y)

cat("Fetching OSM street data (once for all maps)...\n")

osm_cache_path <- file.path(out_dir, "osm_streets_cache.rds")

fetch_osm_streets <- function(bbox, max_tries = 5, wait_sec = 10) {
  for (attempt in seq_len(max_tries)) {
    cat(sprintf("  OSM attempt %d / %d ...\n", attempt, max_tries))
    result <- tryCatch({
      opq(bbox = bbox, timeout = 120) %>%
        add_osm_feature(key = "highway") %>%
        osmdata_sf() %>%
        `[[`("osm_lines")
    }, error = function(e) {
      cat(sprintf("  Attempt %d failed: %s\n", attempt, conditionMessage(e)))
      NULL
    })
    if (!is.null(result) && nrow(result) > 0) return(result)
    if (attempt < max_tries) {
      cat(sprintf("  Waiting %d seconds before retry...\n", wait_sec))
      Sys.sleep(wait_sec)
    }
  }
  NULL
}

# Use cache if available, otherwise fetch
if (file.exists(osm_cache_path)) {
  cat("  Loading OSM streets from cache...\n")
  streets <- readRDS(osm_cache_path)
} else {
  streets <- fetch_osm_streets(bbox_exp)
  if (!is.null(streets)) {
    saveRDS(streets, osm_cache_path)
    cat(sprintf("  OSM streets cached to: %s\n", osm_cache_path))
  } else {
    cat("  WARNING: OSM fetch failed after all retries. Maps will render without streets.\n")
    streets <- NULL
  }
}
cat("OSM streets ready.\n\n")

# =============================================================================
# 3.  Load all instance results
#       — history_df  for the trace plot
#       — summary_df  for the xtable
#       — map_data    for gradient maps (pooled X_all per n_init)
# =============================================================================
all_history <- list()
all_summary <- list()
map_data    <- list()

for (n_init in ini_values) {
  
  dir_path <- file.path(base_dir, paste0("ini", n_init))
  X_pool   <- NULL   # all evaluated x vectors pooled across instances
  
  for (inst in seq_len(n_instances)) {
    rds_file <- file.path(dir_path, sprintf("instance_%02d.rds", inst))
    if (!file.exists(rds_file)) {
      warning(sprintf("Missing: %s", rds_file)); next
    }
    
    res <- readRDS(rds_file)
    h   <- res$history
    
    # Extract best feasible x vector for this instance
    h_feas <- h[h$feasible == 1, ]
    if (nrow(h_feas) > 0) {
      best_row <- h_feas[which.min(h_feas$y), ]
      x_cols   <- paste0("x", seq_len(n_vars))
      best_x   <- as.integer(best_row[1, x_cols])
      X_pool   <- rbind(X_pool, best_x)
    } else {
      warning(sprintf("No feasible solution found in n_init=%d, instance=%d", n_init, inst))
    }
    
    # History (for trace)
    h$n_init   <- n_init
    h$instance <- inst
    h <- h %>%
      dplyr::arrange(true_evals) %>%
      dplyr::mutate(cum_feas_rate = cummean(as.numeric(feasible)))
    all_history[[length(all_history) + 1]] <- h
    
    # Per-instance summary
    all_summary[[length(all_summary) + 1]] <- data.frame(
      n_init       = n_init,
      instance     = inst,
      glpk_optimal = res$glpk_optimal,
      f_best       = ifelse(is.null(res$f_best), NA_real_, res$f_best),
      gap          = ifelse(is.null(res$f_best), NA_real_,
                            res$f_best - res$glpk_optimal),
      pct_feas     = 100 * mean(h$feasible, na.rm = TRUE),
      total_evals  = max(h$true_evals, na.rm = TRUE),
      n_feas_found = sum(h$feasible, na.rm = TRUE)
    )
  }
  
  # Gradient frequencies across all pooled evaluations
  n_pool            <- nrow(X_pool)
  C_pool            <- t(apply(X_pool, 1,
                               function(x) as.integer(as.vector(A %*% x) > 0)))
  facility_sel_freq <- colMeans(X_pool)
  demand_cov_freq   <- colMeans(C_pool)
  
  map_data[[as.character(n_init)]] <- list(
    facility_sel_freq = facility_sel_freq,
    demand_cov_freq   = demand_cov_freq,
    n_pool            = n_pool
  )
  
  cat(sprintf("n_init=%2d: pooled %d solutions across %d instances\n",
              n_init, n_pool, n_instances))
}

history_df <- dplyr::bind_rows(all_history)
summary_df <- dplyr::bind_rows(all_summary)
glpk_ref   <- unique(summary_df$glpk_optimal)[1]

# =============================================================================
# 4.  OBJECTIVE TRACE PLOT
# =============================================================================
trace_agg <- history_df %>%
  dplyr::group_by(n_init, iteration) %>%
  dplyr::summarise(
    mean_fbest = mean(f_best_so_far, na.rm = TRUE),
    sd_fbest   = sd(f_best_so_far,   na.rm = TRUE),
    .groups    = "drop"
  ) %>%
  dplyr::filter(!is.na(mean_fbest)) %>%
  dplyr::mutate(n_init = factor(n_init, levels = ini_values,
                                labels = paste0("Initial samples = ", ini_values)))

p_trace <- ggplot(trace_agg,
                  aes(x = iteration, colour = n_init, fill = n_init)) +
  geom_ribbon(aes(ymin = mean_fbest - sd_fbest,
                  ymax = mean_fbest + sd_fbest),
              alpha = 0.15, colour = NA) +
  geom_line(aes(y = mean_fbest), linewidth = 0.9) +
  geom_hline(yintercept = glpk_ref, linetype = "dashed",
             colour = "black", linewidth = 0.6) +
  annotate("text", x = 5, y = glpk_ref - 0.15,
           label = sprintf("GLPK Optimal = %d", glpk_ref),
           hjust = 0, size = 3.4) +
  scale_colour_manual(values = trace_pal, name = "Initial Sample Size: ") +
  scale_fill_manual(values   = trace_pal, name = "Initial Sample Size: ") +
  labs(
    title    = "LSCP BART-CBO: Best Objective Convergence",
    subtitle = "Mean \u00b1 1 SD across 10 instances per initial sample size",
    x        = "Iteration",
    y        = "Best Feasible Objective (Number of Facilities)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(hjust = 0.5, colour = "grey40")
  )

ggsave(file.path(out_dir, "lscp_objective_trace.pdf"), p_trace,
       width = 8, height = 5, device = cairo_pdf, bg = "white")
cat("\nSaved: lscp_objective_trace.pdf\n")

# =============================================================================
# 5.  GRADIENT MAP FUNCTION
# =============================================================================
make_gradient_map <- function(n_init, md) {
  
  facility_sf <- st_as_sf(facility_loc,
                          coords = c("long", "lat"), crs = 4326) %>%
    dplyr::mutate(sel_freq = md$facility_sel_freq) %>%
    dplyr::arrange(sel_freq)
  
  demand_sf <- st_as_sf(demand_data,
                        coords = c("long", "lat"), crs = 4326) %>%
    dplyr::mutate(cov_freq = md$demand_cov_freq) %>%
    dplyr::arrange(cov_freq)
  
  mean_cov_pct <- round(100 * mean(md$demand_cov_freq), 1)
  
  p <- ggplot()
  if (!is.null(streets)) {
    p <- p + geom_sf(data = streets, colour = "grey70", linewidth = 0.25, alpha = 0.35)
  }
  p +
    # Demand nodes: coverage frequency gradient
    geom_sf(data  = demand_sf,
            aes(colour = cov_freq),
            shape = 16, size = 2.2, alpha = 0.9) +
    
    # Facility nodes: selection frequency gradient (diamond)
    geom_sf(data   = facility_sf,
            aes(fill = sel_freq),
            shape  = 23, size = 5,
            colour = "grey20", stroke = 0.9) +
    
    scale_colour_gradientn(
      name    = "Demand Coverage\nFrequency",
      colours = demand_pal,
      limits  = c(0, 1),
      breaks  = c(0, 0.25, 0.5, 0.75, 1),
      labels  = c("0%", "25%", "50%", "75%", "100%"),
      guide   = guide_colourbar(order       = 1,
                                barwidth    = unit(0.5, "cm"),
                                barheight   = unit(4,   "cm"),
                                title.hjust = 0)
    ) +
    
    scale_fill_gradientn(
      name    = "Facility Selection\nFrequency",
      colours = facility_pal,
      limits  = c(0, 1),
      breaks  = c(0, 0.25, 0.5, 0.75, 1),
      labels  = c("0%", "25%", "50%", "75%", "100%"),
      guide   = guide_colourbar(order       = 2,
                                barwidth    = unit(0.5, "cm"),
                                barheight   = unit(4,   "cm"),
                                title.hjust = 0)
    ) +
    
    coord_sf(xlim   = c(bbox_exp["xmin"], bbox_exp["xmax"]),
             ylim   = c(bbox_exp["ymin"], bbox_exp["ymax"]),
             expand = FALSE) +
    
    labs(
      title   = sprintf("BART-CBO  |  Initial Sample = %d  (%d instances)",
                        n_init, n_instances),
      caption = sprintf(
        "Gradients computed over best feasible solution from each of %d instances  |  Mean demand coverage: %.1f%%",
        md$n_pool, mean_cov_pct
      )
    ) +
    
    theme_minimal(base_size = 11) +
    theme(
      legend.position  = "right",
      legend.box       = "vertical",
      legend.spacing.y = unit(0.6, "cm"),
      legend.title     = element_text(size = 8,   face = "bold"),
      legend.text      = element_text(size = 7.5),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey92", linewidth = 0.2),
      plot.title       = element_text(hjust = 0.5, size = 11),
      plot.caption     = element_text(hjust = 0.5, size = 7.5, colour = "grey40"),
      axis.text        = element_text(size = 6.5)
    )
}

# =============================================================================
# 6.  Render and save maps
# =============================================================================
map_plots <- list()

for (n_init in ini_values) {
  p_map <- make_gradient_map(n_init, map_data[[as.character(n_init)]])
  map_plots[[as.character(n_init)]] <- p_map
  
  fname <- file.path(out_dir, sprintf("lscp_map_ini%d.pdf", n_init))
  ggsave(fname, p_map, width = 11, height = 10,
         device = cairo_pdf, bg = "white")
  cat(sprintf("Saved: %s\n", fname))
}

# Combined — 3 panels side by side
combined_maps <- (map_plots[["5"]] + map_plots[["10"]] + map_plots[["20"]]) +
  plot_layout(ncol = 3, guides = "collect") +
  plot_annotation(
    title    = "LSCP BART-CBO: Solution Gradient Maps by Initial Sample Size",
    subtitle = paste0(
      "Demand gradient = coverage frequency  |  ",
      "Facility gradient = selection frequency  |  ",
      "Computed over best feasible solution per instance"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", hjust = 0.5, size = 13),
      plot.subtitle = element_text(hjust = 0.5, size = 9, colour = "grey40")
    )
  ) &
  theme(legend.position = "right")

ggsave(file.path(out_dir, "lscp_maps_combined.pdf"), combined_maps,
       width = 30, height = 10, device = cairo_pdf, bg = "white")
cat("Saved: lscp_maps_combined.pdf\n")

# =============================================================================
# 7.  XTABLE — full summary + obj evaluations column
# =============================================================================
latex_tbl <- summary_df %>%
  dplyr::group_by(n_init) %>%
  dplyr::summarise(
    `GLPK Opt.`         = unique(glpk_optimal),
    `Mean Best Obj.`    = round(mean(f_best,       na.rm = TRUE), 2),
    `SD Best Obj.`      = round(sd(f_best,         na.rm = TRUE), 2),
    `Mean Gap`          = round(mean(gap,           na.rm = TRUE), 2),
    `SD Gap`            = round(sd(gap,             na.rm = TRUE), 2),
    `Mean Feas. (\\%)`  = round(mean(pct_feas,      na.rm = TRUE), 1),
    `SD Feas. (\\%)`    = round(sd(pct_feas,        na.rm = TRUE), 1),
    `Mean Obj. Evals`   = round(mean(total_evals,   na.rm = TRUE), 1),
    `SD Obj. Evals`     = round(sd(total_evals,     na.rm = TRUE), 1),
    `Mean Feas. Found`  = round(mean(n_feas_found,  na.rm = TRUE), 1),
    .groups = "drop"
  ) %>%
  dplyr::rename(`$n_0$` = n_init)

xt <- xtable(
  latex_tbl,
  caption = paste0(
    "Summary of BART-CBO performance for the LSCP across ", n_instances,
    " instances per initial sample size $n_0 \\in \\{5, 10, 20\\}$. ",
    "GLPK optimal $= ", glpk_ref, "$ facilities. ",
    "Gap $=$ Best Obj.\\ $-$ GLPK Opt. ",
    "Feas.\\ (\\%) is the percentage of all evaluated solutions satisfying ",
    "all coverage constraints. ",
    "Obj.\\ Evals is the total number of true objective evaluations per run."
  ),
  label  = "tab:lscp_ini_summary",
  digits = c(0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1)
)

tex_out <- capture.output(
  print(xt,
        include.rownames       = FALSE,
        booktabs               = TRUE,
        caption.placement      = "top",
        sanitize.text.function = identity,
        comment                = FALSE)
)

tex_path <- file.path(out_dir, "lscp_summary_table.tex")
writeLines(tex_out, tex_path)
cat(sprintf("Saved: %s\n\n", tex_path))

cat("--- LaTeX table preview ---\n")
cat(tex_out, sep = "\n")

cat("\n\n--- R summary table ---\n")
print(as.data.frame(latex_tbl), row.names = FALSE)