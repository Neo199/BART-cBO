# =============================================================================
# MCLP Map — two-panel figure 
#
#  Panel A (left) : DART-CBO gradient across ALL 10 runs
#                   Demand  → coverage frequency  (colorblind-safe gradient)
#                   Facility→ selection frequency  (colorblind-safe gradient)
#
#  Panel B (right): DART-CBO BEST single run (highest coverage)
#                   Styled identically to the existing OSM plot code
#
# Colorblind palette used: Okabe-Ito throughout
#   Demand gradient  : #D55E00 (vermillion) → #F0E442 (yellow) → #009E73 (bluish-green)
#   Facility gradient: #FFFFFF (white)      → #56B4E9 (sky blue)→ #0072B2 (blue)
#   These are all Okabe-Ito colours — safe for all common CVD types.
#
# Requires (in output_dir):
#   dart_best_x_run_01.rds ... dart_best_x_run_10.rds
#   glpk_best_x.rds   (optional — used for caption only)
#   summary_all_runs.csv
# =============================================================================

library(sf)
library(ggplot2)
library(osmdata)
library(dplyr)
library(tidyr)
library(patchwork)

output_dir <- "BART-CBO/Constrained/dartMachine/Facility Location/results_MCLP_comparison"
data_dir   <- "PhD-Data/SF_data"
save_path  <- file.path(output_dir, "map_dartcbo_gradient_and_best.pdf")
N_RUNS     <- 10

# ---------------------------------------------------------------------------
# Okabe-Ito palette reference
#   [1] #000000 black      [2] #E69F00 orange     [3] #56B4E9 sky blue
#   [4] #009E73 bl-green   [5] #F0E442 yellow      [6] #0072B2 blue
#   [7] #D55E00 vermillion [8] #CC79A7 red-purple
# ---------------------------------------------------------------------------
okabe_ito <- c("#000000","#E69F00","#56B4E9","#009E73",
               "#F0E442","#0072B2","#D55E00","#CC79A7")

# Gradient palettes (all Okabe-Ito colours)
demand_pal   <- c(okabe_ito[7], okabe_ito[5], okabe_ito[4])  # vermillion→yellow→bl-green
facility_pal <- c("#FFFFFF",    okabe_ito[3], okabe_ito[6])  # white→sky blue→blue

# Binary colours for best-run panel (matching existing plotting code)
col_covered    <- okabe_ito[4]   # bluish green
col_uncovered  <- okabe_ito[7]   # vermillion
col_selected   <- okabe_ito[6]   # blue
col_unselected <- okabe_ito[8]   # reddish purple

# ---------------------------------------------------------------------------
# 1. Load spatial / problem data
# ---------------------------------------------------------------------------
demand_data     <- read.csv(file.path(data_dir, "SF_demand_205_centroid_uniform_weight.csv"))
facility_loc    <- read.csv(file.path(data_dir, "SF_store_site_16_longlat.csv"))
distance_matrix <- read.csv(file.path(data_dir, "SF_network_distance_candidateStore_16_censusTract_205_new.csv"))

distance_matrix$covered <- as.integer(distance_matrix$distance <= 5000)

A_df <- distance_matrix %>%
  dplyr::select(DestinationName, name, covered) %>%
  tidyr::pivot_wider(names_from  = name,
                     values_from = covered,
                     values_fill = list(covered = 0))

A          <- as.matrix(A_df[, -1])
population <- demand_data$POP2000
total_pop  <- sum(population)
n_vars     <- ncol(A)
n_demand   <- nrow(A)

cat(sprintf("Demand nodes: %d  |  Candidate facilities: %d\n", n_demand, n_vars))

# ---------------------------------------------------------------------------
# 2. Load all DART-CBO x vectors
# ---------------------------------------------------------------------------
x_list <- lapply(seq_len(N_RUNS), function(i) {
  p <- file.path(output_dir, sprintf("dart_best_x_run_%02d.rds", i))
  if (!file.exists(p)) stop(sprintf("Missing: %s", p))
  as.integer(readRDS(p))
})

X_runs <- do.call(rbind, x_list)                         # N_RUNS x n_vars
C_runs <- t(apply(X_runs, 1, function(x) as.integer(as.vector(A %*% x) > 0)))

facility_sel_freq <- colMeans(X_runs)
demand_cov_freq   <- colMeans(C_runs)

# Coverage (population) per run — find best run
run_coverage <- sapply(x_list, function(x) sum(population[as.vector(A %*% x) > 0]))
best_run     <- which.max(run_coverage)
best_x       <- x_list[[best_run]]
best_covered <- as.vector(A %*% best_x) > 0
best_cov_pct <- round(100 * sum(population[best_covered]) / total_pop, 2)

cat(sprintf("Best run: %d  |  Coverage: %.0f  (%.2f%%)\n",
            best_run, run_coverage[best_run], best_cov_pct))

# GLPK comparison (optional)
glpk_path <- file.path(output_dir, "glpk_best_x.rds")
runs_at_glpk <- NA_integer_
if (file.exists(glpk_path)) {
  glpk_x    <- readRDS(glpk_path)
  glpk_cov  <- sum(population[as.vector(A %*% glpk_x) > 0])
  runs_at_glpk <- sum(abs(run_coverage - glpk_cov) < 1)
  cat(sprintf("GLPK coverage: %.0f  |  Runs matching GLPK: %d/%d\n",
              glpk_cov, runs_at_glpk, N_RUNS))
}

mean_cov_pct <- round(100 * mean(demand_cov_freq), 1)

# ---------------------------------------------------------------------------
# 3. Spatial objects
# ---------------------------------------------------------------------------

# --- Gradient panel ---
facility_sf_grad <- st_as_sf(facility_loc, coords = c("long", "lat"), crs = 4326) %>%
  dplyr::mutate(sel_freq = facility_sel_freq) %>%
  dplyr::arrange(sel_freq)

demand_sf_grad <- st_as_sf(demand_data, coords = c("long", "lat"), crs = 4326) %>%
  dplyr::mutate(cov_freq = demand_cov_freq) %>%
  dplyr::arrange(cov_freq)

# --- Best-run panel ---
facility_sf_best <- st_as_sf(facility_loc, coords = c("long", "lat"), crs = 4326) %>%
  dplyr::mutate(
    selected = best_x,
    selection_status = factor(ifelse(selected == 1, "Selected", "Not selected"),
                              levels = c("Selected", "Not selected"))
  )

demand_sf_best <- st_as_sf(demand_data, coords = c("long", "lat"), crs = 4326) %>%
  dplyr::mutate(
    covered = best_covered,
    coverage_status = factor(ifelse(covered, "Covered", "Uncovered"),
                             levels = c("Covered", "Uncovered"))
  )

# ---------------------------------------------------------------------------
# 4. Shared bounding box + OSM streets (fetched once)
# ---------------------------------------------------------------------------
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

cat("Fetching OSM street data...\n")
streets <- opq(bbox = bbox_exp) %>%
  add_osm_feature(key = "highway") %>%
  osmdata_sf() %>%
  `[[`("osm_lines")
cat("OSM data loaded.\n\n")

# ---------------------------------------------------------------------------
# 5. Panel A — gradient map
# ---------------------------------------------------------------------------
panel_grad <- ggplot() +
  
  geom_sf(data = streets, colour = "grey70", linewidth = 0.25, alpha = 0.35) +
  
  # Demand: coverage frequency gradient
  geom_sf(data  = demand_sf_grad,
          aes(colour = cov_freq),
          shape = 16, size = 2.4, alpha = 0.9) +
  
  # Facilities: selection frequency gradient
  geom_sf(data   = facility_sf_grad,
          aes(fill = sel_freq),
          shape  = 23, size = 5,
          colour = "grey20", stroke = 0.9) +
  
  scale_colour_gradientn(
    name    = "Demand coverage\nfrequency",
    colours = demand_pal,
    limits  = c(0, 1),
    breaks  = c(0, 0.25, 0.5, 0.75, 1),
    labels  = c("0%", "25%", "50%", "75%", "100%"),
    guide   = guide_colourbar(order     = 1,
                              barwidth  = unit(0.5, "cm"),
                              barheight = unit(4.5, "cm"),
                              title.hjust = 0)
  ) +
  
  scale_fill_gradientn(
    name    = "Facility selection\nfrequency",
    colours = facility_pal,
    limits  = c(0, 1),
    breaks  = c(0, 0.25, 0.5, 0.75, 1),
    labels  = c("0%", "25%", "50%", "75%", "100%"),
    guide   = guide_colourbar(order     = 2,
                              barwidth  = unit(0.5, "cm"),
                              barheight = unit(4.5, "cm"),
                              title.hjust = 0)
  ) +
  
  coord_sf(xlim = c(bbox_exp["xmin"], bbox_exp["xmax"]),
           ylim = c(bbox_exp["ymin"], bbox_exp["ymax"]),
           expand = FALSE) +
  
  labs(
    title   = sprintf("BART-CBO: Averaged across %d instances", N_RUNS),
    caption = paste0("Mean demand coverage: ", mean_cov_pct, "%",
                     if (!is.na(runs_at_glpk))
                       paste0("  |  Runs matching GLPK: ", runs_at_glpk, "/", N_RUNS)
                     else "")
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    legend.position   = "right",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.8, "cm"),
    legend.title      = element_text(size = 9,  face = "bold"),
    legend.text       = element_text(size = 8),
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(colour = "grey92", linewidth = 0.2),
    plot.title        = element_text(face = "bold", hjust = 0.5, size = 12),
    plot.caption      = element_text(hjust = 0.5, size = 8, colour = "grey40"),
    axis.text         = element_text(size = 7)
  )

# ---------------------------------------------------------------------------
# 6. Panel B — best single run (binary, same style as existing code)
# ---------------------------------------------------------------------------
panel_best <- ggplot() +
  
  geom_sf(data = streets, colour = "grey70", linewidth = 0.25, alpha = 0.35) +
  
  geom_sf(data = demand_sf_best,
          aes(colour = coverage_status, shape = coverage_status),
          size = 2.4, alpha = 0.85) +
  
  geom_sf(data   = facility_sf_best,
          aes(fill = selection_status),
          shape  = 23, size = 5,
          colour = "grey20", stroke = 0.9) +
  
  scale_colour_manual(
    name   = "Demand Status",
    values = c("Covered" = col_covered, "Uncovered" = col_uncovered),
    guide  = guide_legend(order = 1,
                          override.aes = list(size = 3))
  ) +
  
  scale_shape_manual(
    name   = "Demand Status",
    values = c("Covered" = 16, "Uncovered" = 1),
    guide  = "none"
  ) +
  
  scale_fill_manual(
    name   = "Facility Status",
    values = c("Selected" = col_selected, "Not selected" = col_unselected),
    guide  = guide_legend(order = 2,
                          override.aes = list(shape  = 23, size = 5,
                                              colour = okabe_ito[1], stroke = 0.9))
  ) +
  
  coord_sf(xlim = c(bbox_exp["xmin"], bbox_exp["xmax"]),
           ylim = c(bbox_exp["ymin"], bbox_exp["ymax"]),
           expand = FALSE) +
  
  labs(
    title   = sprintf("BART-CBO: Best result and true solution", best_run),
    caption = paste0(
      "Facilities selected: ", sum(best_x),
      "  |  Demand covered: ", sum(best_covered), "/", n_demand,
      "  (", best_cov_pct, "% of population)"
    )
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    legend.position   = "right",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.5, "cm"),
    legend.title      = element_text(size = 9,  face = "bold"),
    legend.text       = element_text(size = 8),
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(colour = "grey92", linewidth = 0.2),
    plot.title        = element_text(face = "bold", hjust = 0.5, size = 12),
    plot.caption      = element_text(hjust = 0.5, size = 8, colour = "grey40"),
    axis.text         = element_text(size = 7)
  )

# ---------------------------------------------------------------------------
# 7. Combine with patchwork
# ---------------------------------------------------------------------------
combined <- panel_grad + panel_best +
  plot_annotation(
    title    = "BART-CBO MCLP Solutions: Run-averaged vs Best Single Run",
    subtitle = sprintf(
      "Service radius: 5,000 m  |  Max facilities: 4  |  Total population: %s  |  %d runs",
      formatC(total_pop, format = "d", big.mark = ","), N_RUNS
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5,   size = 10,  colour = "grey40")
    )
  )

# ---------------------------------------------------------------------------
# 8. Save — individually + combined
# ---------------------------------------------------------------------------

# Panel A — gradient
ggsave(filename = file.path(output_dir, "map_dartcbo_gradient_only.pdf"),
       plot     = panel_grad,
       device   = cairo_pdf,
       width    = 11, height = 10, units = "in", dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", file.path(output_dir, "map_dartcbo_gradient_only.pdf")))

# Panel B — best run
ggsave(filename = file.path(output_dir, "map_dartcbo_best_run_only.pdf"),
       plot     = panel_best,
       device   = cairo_pdf,
       width    = 11, height = 10, units = "in", dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", file.path(output_dir, "map_dartcbo_best_run_only.pdf")))

# Combined
ggsave(filename = save_path, plot = combined, device = cairo_pdf,
       width = 20, height = 10, units = "in", dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", save_path))
