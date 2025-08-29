#!/usr/bin/env Rscript
# =========================================================
# NETWORK → 5×5 km aggregated cells + species presence/absence + LLM candidates (LOOSE ONLY)
# Input : /Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL/NETWORK.xlsx
# Output: all CSVs saved to the same New_FXL folder
# Predictors kept: RWQ, ALT, FFP, BIO1
# =========================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(sf)
  library(readr)
})

# ---- Paths ----
in_dir    <- "/Users/kristianmiok/Desktop/Lucian/LLM/Data/New_FXL"
xlsx_path <- file.path(in_dir, "NETWORK.xlsx")
out_dir   <- in_dir  # write results here

if (!file.exists(xlsx_path)) stop("File not found: ", xlsx_path)

# Pick sheet (first by default; change if needed)
sheets <- readxl::excel_sheets(xlsx_path)
sheet_name <- sheets[[1]]  # or set explicitly: "network"
message("Reading sheet: ", sheet_name)

# ---- 1) Read Excel ----
net_raw <- read_excel(xlsx_path, sheet = sheet_name)

# ---- 2) Detect/normalize coordinate columns ----
pick_col <- function(df, candidates) {
  for (cand in candidates) {
    if (cand %in% names(df)) return(cand)
  }
  return(NA_character_)
}

x_col <- pick_col(net_raw, c("X_WGS84_DD","Longitude","LONGITUDE","lon","x"))
y_col <- pick_col(net_raw, c("Y_WGS84_DD","Latitude","LATITUDE","lat","y"))
if (is.na(x_col) || is.na(y_col)) {
  stop("Could not find WGS84 coordinate columns. Expected e.g. X_WGS84_DD / Y_WGS84_DD.")
}

# ---- 3) Predictor set we’ll aggregate by median (ONLY those you have) ----
pred_cols <- c("RWQ","ALT","FFP","BIO1")
pred_cols <- pred_cols[pred_cols %in% names(net_raw)]

# ---- 4) Presence/true-absence flags — accept either has_* or bare names ----
presence_sets <- list(
  c("has_AUB_PREZ","has_AUT_PREZ","has_FXL_PREZ"),
  c("AUB_PREZ","AUT_PREZ","FXL_PREZ")
)
trueabs_sets <- list(
  c("has_AUB_TRUEABS","has_AUT_TRUEABS","has_FXL_TRUEABS"),
  c("AUB_TRUEABS","AUT_TRUEABS","FXL_TRUEABS")
)

choose_flag_set <- function(df, sets) {
  for (s in sets) if (all(s %in% names(df))) return(s)
  return(sets[[1]])  # default canonical names if nothing matches
}

presence_cols <- choose_flag_set(net_raw, presence_sets)
trueabs_cols  <- choose_flag_set(net_raw, trueabs_sets)

# ---- 5) Coerce numbers (handle comma decimals safely) ----
to_num_comma <- function(x) {
  if (inherits(x, "numeric")) return(x)
  x <- as.character(x)
  x <- str_replace_all(x, "\\s+", "")
  x <- str_replace_all(x, ",", ".")
  suppressWarnings(as.numeric(x))
}

num_cols <- unique(c(pred_cols, x_col, y_col, presence_cols, trueabs_cols))
num_cols <- num_cols[num_cols %in% names(net_raw)]

net <- net_raw %>%
  mutate(across(all_of(num_cols), to_num_comma)) %>%
  mutate(
    THS = if ("THS" %in% names(.)) as.character(THS) else NA_character_,
    TYS = if ("TYS" %in% names(.)) as.character(TYS) else NA_character_
  ) %>%
  drop_na(all_of(c(x_col, y_col)))  # drop rows without coords

# Ensure flags exist (if missing in input, create zeros)
for (c in c(presence_cols, trueabs_cols)) {
  if (!c %in% names(net)) net[[c]] <- 0
}

# ---- 6) Build 5×5 km CellID using EPSG:3035 ----
sf_pts  <- st_as_sf(net, coords = c(x_col, y_col), crs = 4326, remove = FALSE)
sf_laea <- st_transform(sf_pts, 3035)  # meters
coords  <- st_coordinates(sf_laea)
tile_x  <- floor(coords[,1] / 5000)    # 5 km bins
tile_y  <- floor(coords[,2] / 5000)
net$CellID <- paste0("C_", tile_x, "_", tile_y)

# ---- 7) Helpers ----
mode_chr <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

# ---- 8) Aggregate entire network to one row per CellID ----
net_aggr_all <- net %>%
  group_by(CellID) %>%
  summarise(
    across(all_of(pred_cols), ~median(.x, na.rm = TRUE)),
    !!y_col := median(.data[[y_col]], na.rm = TRUE),
    !!x_col := median(.data[[x_col]], na.rm = TRUE),
    THS = if ("THS" %in% names(net)) mode_chr(THS) else NA_character_,
    TYS = if ("TYS" %in% names(net)) mode_chr(TYS) else NA_character_,
    # normalize to canonical has_* columns
    has_AUB_PREZ    = any((.data[[presence_cols[1]]] %in% c(1, TRUE)), na.rm = TRUE),
    has_AUT_PREZ    = any((.data[[presence_cols[2]]] %in% c(1, TRUE)), na.rm = TRUE),
    has_FXL_PREZ    = any((.data[[presence_cols[3]]] %in% c(1, TRUE)), na.rm = TRUE),
    has_AUB_TRUEABS = any((.data[[trueabs_cols[1]]]  %in% c(1, TRUE)), na.rm = TRUE),
    has_AUT_TRUEABS = any((.data[[trueabs_cols[2]]]  %in% c(1, TRUE)), na.rm = TRUE),
    has_FXL_TRUEABS = any((.data[[trueabs_cols[3]]]  %in% c(1, TRUE)), na.rm = TRUE),
    n_rows_in_cell  = dplyr::n(),
    .groups = "drop"
  ) %>%
  # make standard coordinate names for downstream Python (X = lon, Y = lat)
  rename(
    Y_WGS84_DD = !!y_col,
    X_WGS84_DD = !!x_col
  ) %>%
  # drop rows with all predictors missing (if any)
  {
    if (length(pred_cols)) drop_na(., any_of(pred_cols)) else .
  }

# Quick sanity
message("\n--- net_aggr_all dim ---")
print(dim(net_aggr_all))
message("\n--- net_aggr_all head ---")
print(utils::head(net_aggr_all, 10))

# ---- 9) Per-species presence / true-absence tables ----
AUB_pres_cells <- net_aggr_all %>% filter(has_AUB_PREZ)    %>% mutate(presence = 1L)
AUB_abs_cells  <- net_aggr_all %>% filter(has_AUB_TRUEABS) %>% mutate(presence = 0L)

AUT_pres_cells <- net_aggr_all %>% filter(has_AUT_PREZ)    %>% mutate(presence = 1L)
AUT_abs_cells  <- net_aggr_all %>% filter(has_AUT_TRUEABS) %>% mutate(presence = 0L)

FXL_pres_cells <- net_aggr_all %>% filter(has_FXL_PREZ)    %>% mutate(presence = 1L)
FXL_abs_cells  <- net_aggr_all %>% filter(has_FXL_TRUEABS) %>% mutate(presence = 0L)

message("\n--- Counts per species (cells) ---")
message("AUB: pres = ", nrow(AUB_pres_cells), "  trueabs = ", nrow(AUB_abs_cells))
message("AUT: pres = ", nrow(AUT_pres_cells), "  trueabs = ", nrow(AUT_abs_cells))
message("FXL: pres = ", nrow(FXL_pres_cells), "  trueabs = ", nrow(FXL_abs_cells))

# ---- 10) LLM candidate pools (LOOSE ONLY) ----
# “Loose” = exclude only presences of the focal species;
# other species’ presences are allowed to remain in the pool.
AUB_llm_candidates_loose <- net_aggr_all %>% filter(!has_AUB_PREZ)
AUT_llm_candidates_loose <- net_aggr_all %>% filter(!has_AUT_PREZ)
FXL_llm_candidates_loose <- net_aggr_all %>% filter(!has_FXL_PREZ)

message("\n--- Candidate pool sizes (LOOSE) ---")
message("AUB: ", nrow(AUB_llm_candidates_loose),
        "  AUT: ", nrow(AUT_llm_candidates_loose),
        "  FXL: ", nrow(FXL_llm_candidates_loose))

# ---- 11) Save outputs ----
write_csv(net_aggr_all, file.path(out_dir, "net_aggr_all.csv"))

write_csv(AUB_pres_cells, file.path(out_dir, "AUB_pres_cells.csv"))
write_csv(AUB_abs_cells,  file.path(out_dir, "AUB_trueabs_cells.csv"))
write_csv(AUT_pres_cells, file.path(out_dir, "AUT_pres_cells.csv"))
write_csv(AUT_abs_cells,  file.path(out_dir, "AUT_trueabs_cells.csv"))
write_csv(FXL_pres_cells, file.path(out_dir, "FXL_pres_cells.csv"))
write_csv(FXL_abs_cells,  file.path(out_dir, "FXL_trueabs_cells.csv"))

write_csv(AUB_llm_candidates_loose, file.path(out_dir, "AUB_llm_candidates_loose.csv"))
write_csv(AUT_llm_candidates_loose, file.path(out_dir, "AUT_llm_candidates_loose.csv"))
write_csv(FXL_llm_candidates_loose, file.path(out_dir, "FXL_llm_candidates_loose.csv"))

message("\nAll done. Files written to:\n", out_dir)