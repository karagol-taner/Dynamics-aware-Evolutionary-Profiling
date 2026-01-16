# ==============================================================================
# Dynamics-Aware Evolutionary Analysis: Statistical Correlation with AlphaMissense
# ==============================================================================

# --- 0. AUTO-INSTALL PACKAGES ---
# This block automatically checks for missing packages and installs them.
required_packages <- c("tidyverse", "ggpubr", "viridis", "hexbin", "data.table", "svglite")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggpubr)
library(viridis)

# --- 1. CONFIGURATION & DATA LOADING ---

input_file <- "/content/All_Proteins_AM_Combined_Master_2.tsv"
output_dir <- "stats_output_human"
trim_termini_count <- 5   # Number of residues to remove from N- and C-termini to reduce noise
cutoff_medium <- 2.5      # Cutoff for focused analysis (The dense region)
cutoff_extreme <- 0.5     # Cutoff for EXTREMELY focused analysis (Background noise check)

# Create output directory
dir.create(output_dir, showWarnings = FALSE)

# --- START TEXT LOGGING ---
# Capture all console output (stats, p-values) to a text file
log_file <- file.path(output_dir, "analysis_results.txt")
sink(log_file, split = TRUE)

message("=======================================================")
message(paste("Analysis Started at:", Sys.time()))
message("=======================================================")

# Load Data
message("Loading Master Dataset...")
df <- read_tsv(input_file, na = c("N/A", "NA", ""))

# --- 2. PREPROCESSING & CLEANING ---

# Ensure numeric columns are actually numeric
numeric_cols <- c("i", "Dynamic_Conserved_Score_0_10",
                  "Dynamics_Coupling_Score_0_10", "Rigid_Conserved_Score_0_10",
                  "Rigid_Coupling_Score_0_10", "mean_am_pathogenicity")

df <- df %>% mutate(across(all_of(numeric_cols), as.numeric))

# Filter: Remove rows with missing essential scores (Residue Index)
df_clean <- df %>% filter(!is.na(i))

# --- HANDLING TERMINI BIAS ---
# Trimming the first and last X residues to avoid artifacts from fraying ends
df_trimmed <- df_clean %>%
  group_by(Protein_ID) %>%
  arrange(i) %>%
  filter(row_number() > trim_termini_count & row_number() <= (n() - trim_termini_count)) %>%
  ungroup()

message(paste0("Original Rows: ", nrow(df_clean)))
message(paste0("Rows after trimming termini (", trim_termini_count, " res): ", nrow(df_trimmed)))

# --- 3. DISTRIBUTION CHECKS ---

# Function to plot distribution
plot_dist <- function(data, column, title) {
  ggplot(data, aes_string(x = column)) +
    geom_density(fill = "#69b3a2", alpha = 0.6) +
    theme_pubr() +
    labs(title = paste("Distribution of", title), x = title, y = "Density")
}

p1 <- plot_dist(df_trimmed, "Dynamic_Conserved_Score_0_10", "DCS (Dynamic Conserved)")
p2 <- plot_dist(df_trimmed, "mean_am_pathogenicity", "AlphaMissense Score")
p3 <- plot_dist(df_trimmed, "Rigid_Conserved_Score_0_10", "RCS (Rigid Conserved)")
p4 <- plot_dist(df_trimmed, "Rigid_Coupling_Score_0_10", "RCopS (Rigid Coupled)")

# Save Distribution Plots (PNG + SVG) - 2x2 Grid
p_dist <- ggarrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
ggsave(file.path(output_dir, "distribution_checks.png"), p_dist, width = 10, height = 8, dpi = 300)
ggsave(file.path(output_dir, "distribution_checks.svg"), p_dist, width = 10, height = 8)


# --- 4. CORRELATION FUNCTION ---

run_correlation <- function(data, x_col, y_col, x_label, y_label) {
  
  # Remove NAs for this specific pair
  sub_data <- data %>% filter(!is.na(!!sym(x_col)) & !is.na(!!sym(y_col)))
  
  # 1. Global Spearman Correlation
  # exact=FALSE silences the "cannot compute exact p-value with ties" warning
  res <- cor.test(sub_data[[x_col]], sub_data[[y_col]], method = "spearman", exact = FALSE)
  
  message(paste0("\n-------------------------------------------------------"))
  message(paste0("Correlation: ", x_label, " vs ", y_label))
  message(paste0("-------------------------------------------------------"))
  message(paste("Spearman rho:", round(res$estimate, 4)))
  message(paste("P-value:", signif(res$p.value, 4)))
  message(paste("N (Residues):", nrow(sub_data)))
  
  # 2. Hexbin Scatter Plot (Best for large N to visualize density)
  p <- ggplot(sub_data, aes_string(x = x_col, y = y_col)) +
    geom_hex(bins = 50) +
    scale_fill_viridis(option = "magma", trans = "log") +
    geom_smooth(method = "gam", color = "cyan", se = TRUE) + # Generalized Additive Model for trend
    stat_cor(method = "spearman", label.x.npc = "left", label.y.npc = "top", color = "white") +
    theme_dark() +
    labs(title = paste(x_label, "vs AlphaMissense"),
         x = x_label, y = y_label)
  
  return(p)
}

# --- 4.1 GLOBAL ANALYSIS ---
message("\n=== Running Global Analysis ===")
p_dcs_glob <- run_correlation(df_trimmed, "Dynamic_Conserved_Score_0_10", "mean_am_pathogenicity",
                              "DCS (Global)", "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCS_Global.png"), p_dcs_glob, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCS_Global.svg"), p_dcs_glob, width = 7, height = 6)

p_dcops_glob <- run_correlation(df_trimmed, "Dynamics_Coupling_Score_0_10", "mean_am_pathogenicity",
                                "DCopS (Global)", "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCopS_Global.png"), p_dcops_glob, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCopS_Global.svg"), p_dcops_glob, width = 7, height = 6)

p_rcs_glob <- run_correlation(df_trimmed, "Rigid_Conserved_Score_0_10", "mean_am_pathogenicity",
                              "RCS (Global)", "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCS_Global.png"), p_rcs_glob, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCS_Global.svg"), p_rcs_glob, width = 7, height = 6)

p_rcops_glob <- run_correlation(df_trimmed, "Rigid_Coupling_Score_0_10", "mean_am_pathogenicity",
                                "RCopS (Global)", "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCopS_Global.png"), p_rcops_glob, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCopS_Global.svg"), p_rcops_glob, width = 7, height = 6)


# --- 4.2 FOCUSED ANALYSIS (< 2.5) ---
message(paste0("\n=== Running Focused Analysis (Scores < ", cutoff_medium, ") ==="))

df_med_dcs <- df_trimmed %>% filter(Dynamic_Conserved_Score_0_10 < cutoff_medium)
p_dcs_med <- run_correlation(df_med_dcs, "Dynamic_Conserved_Score_0_10", "mean_am_pathogenicity",
                             paste0("DCS (<", cutoff_medium, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCS_MediumRange.png"), p_dcs_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCS_MediumRange.svg"), p_dcs_med, width = 7, height = 6)

df_med_dcops <- df_trimmed %>% filter(Dynamics_Coupling_Score_0_10 < cutoff_medium)
p_dcops_med <- run_correlation(df_med_dcops, "Dynamics_Coupling_Score_0_10", "mean_am_pathogenicity",
                               paste0("DCopS (<", cutoff_medium, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCopS_MediumRange.png"), p_dcops_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCopS_MediumRange.svg"), p_dcops_med, width = 7, height = 6)

df_med_rcs <- df_trimmed %>% filter(Rigid_Conserved_Score_0_10 < cutoff_medium)
p_rcs_med <- run_correlation(df_med_rcs, "Rigid_Conserved_Score_0_10", "mean_am_pathogenicity",
                             paste0("RCS (<", cutoff_medium, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCS_MediumRange.png"), p_rcs_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCS_MediumRange.svg"), p_rcs_med, width = 7, height = 6)

df_med_rcops <- df_trimmed %>% filter(Rigid_Coupling_Score_0_10 < cutoff_medium)
p_rcops_med <- run_correlation(df_med_rcops, "Rigid_Coupling_Score_0_10", "mean_am_pathogenicity",
                               paste0("RCopS (<", cutoff_medium, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCopS_MediumRange.png"), p_rcops_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCopS_MediumRange.svg"), p_rcops_med, width = 7, height = 6)


# --- 4.3 EXTREME ANALYSIS (< 0.5) ---
message(paste0("\n=== Running Extreme Analysis (Scores < ", cutoff_extreme, ") ==="))

df_low_dcs <- df_trimmed %>% filter(Dynamic_Conserved_Score_0_10 < cutoff_extreme)
p_dcs_low <- run_correlation(df_low_dcs, "Dynamic_Conserved_Score_0_10", "mean_am_pathogenicity",
                             paste0("DCS (<", cutoff_extreme, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCS_ExtremeLow.png"), p_dcs_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCS_ExtremeLow.svg"), p_dcs_low, width = 7, height = 6)

df_low_dcops <- df_trimmed %>% filter(Dynamics_Coupling_Score_0_10 < cutoff_extreme)
p_dcops_low <- run_correlation(df_low_dcops, "Dynamics_Coupling_Score_0_10", "mean_am_pathogenicity",
                               paste0("DCopS (<", cutoff_extreme, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_DCopS_ExtremeLow.png"), p_dcops_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_DCopS_ExtremeLow.svg"), p_dcops_low, width = 7, height = 6)

df_low_rcs <- df_trimmed %>% filter(Rigid_Conserved_Score_0_10 < cutoff_extreme)
p_rcs_low <- run_correlation(df_low_rcs, "Rigid_Conserved_Score_0_10", "mean_am_pathogenicity",
                             paste0("RCS (<", cutoff_extreme, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCS_ExtremeLow.png"), p_rcs_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCS_ExtremeLow.svg"), p_rcs_low, width = 7, height = 6)

df_low_rcops <- df_trimmed %>% filter(Rigid_Coupling_Score_0_10 < cutoff_extreme)
p_rcops_low <- run_correlation(df_low_rcops, "Rigid_Coupling_Score_0_10", "mean_am_pathogenicity",
                               paste0("RCopS (<", cutoff_extreme, ")"), "Mean AM Pathogenicity")
ggsave(file.path(output_dir, "Correlation_RCopS_ExtremeLow.png"), p_rcops_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Correlation_RCopS_ExtremeLow.svg"), p_rcops_low, width = 7, height = 6)


# --- 5. BINNED ANALYSIS (Strong Statistics for Non-Linearity) ---
# Question: Do residues with HIGH Dynamic Scores actually have HIGHER pathogenicity?

analyze_bins <- function(data, score_col, score_name) {
  
  sub_data <- data %>% filter(!is.na(!!sym(score_col)) & !is.na(mean_am_pathogenicity))
  
  # FIX: Use Quantiles (Q1, Q2, Q3, Q4) instead of fixed cuts.
  quants <- quantile(sub_data[[score_col]], probs = c(0, 0.25, 0.5, 0.75, 1.0), na.rm = TRUE)
  
  # Robustness check: Handle potential identical quantiles (e.g. if >50% of data is 0)
  if(length(unique(quants)) < 5) {
     # Fallback: simple 3-way cut if data is very skewed
     sub_data$Score_Bin <- cut(sub_data[[score_col]], 3, labels=c("Low", "Med", "High"))
     comp_list <- list(c("Low", "High"))
  } else {
     sub_data$Score_Bin <- cut(sub_data[[score_col]],
                               breaks = quants,
                               labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
                               include.lowest = TRUE)
     comp_list <- list(c("Q1 (Low)", "Q4 (High)"))
  }

  # Kruskal-Wallis Test (Non-parametric ANOVA)
  kw_test <- kruskal.test(mean_am_pathogenicity ~ Score_Bin, data = sub_data)
  
  message(paste0("\n-------------------------------------------------------"))
  message(paste0("Kruskal-Wallis Test for ", score_name, " Bins"))
  message(paste0("-------------------------------------------------------"))
  print(kw_test)
  
  # Boxplot with stats
  p <- ggplot(sub_data, aes(x = Score_Bin, y = mean_am_pathogenicity, fill = Score_Bin)) +
    geom_boxplot(outlier.alpha = 0.1) +
    # Compare Lowest Bin vs Highest Bin
    stat_compare_means(method = "wilcox.test", comparisons = comp_list, label = "p.signif") +
    theme_pubr() +
    labs(title = paste("AlphaMissense Pathogenicity by", score_name, "Quartile"),
         x = paste(score_name, "Bin"), y = "AlphaMissense Pathogenicity") +
    theme(legend.position = "none")
  
  return(p)
}

# --- 5.1 GLOBAL BINNING ---
message("\n=== Running Global Binned Analysis ===")

plot_box_dcs <- analyze_bins(df_trimmed, "Dynamic_Conserved_Score_0_10", "DCS (Global)")
ggsave(file.path(output_dir, "Boxplot_DCS_Global.png"), plot_box_dcs, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCS_Global.svg"), plot_box_dcs, width = 7, height = 6)

plot_box_dcops <- analyze_bins(df_trimmed, "Dynamics_Coupling_Score_0_10", "DCopS (Global)")
ggsave(file.path(output_dir, "Boxplot_DCopS_Global.png"), plot_box_dcops, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCopS_Global.svg"), plot_box_dcops, width = 7, height = 6)

plot_box_rcs <- analyze_bins(df_trimmed, "Rigid_Conserved_Score_0_10", "RCS (Global)")
ggsave(file.path(output_dir, "Boxplot_RCS_Global.png"), plot_box_rcs, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCS_Global.svg"), plot_box_rcs, width = 7, height = 6)

plot_box_rcops <- analyze_bins(df_trimmed, "Rigid_Coupling_Score_0_10", "RCopS (Global)")
ggsave(file.path(output_dir, "Boxplot_RCopS_Global.png"), plot_box_rcops, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCopS_Global.svg"), plot_box_rcops, width = 7, height = 6)

# --- 5.2 FOCUSED BINNING (< 2.5) ---
message(paste0("\n=== Running Focused Binned Analysis (< ", cutoff_medium, ") ==="))

plot_box_dcs_med <- analyze_bins(df_med_dcs, "Dynamic_Conserved_Score_0_10", paste0("DCS (<", cutoff_medium, ")"))
ggsave(file.path(output_dir, "Boxplot_DCS_MediumRange.png"), plot_box_dcs_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCS_MediumRange.svg"), plot_box_dcs_med, width = 7, height = 6)

plot_box_dcops_med <- analyze_bins(df_med_dcops, "Dynamics_Coupling_Score_0_10", paste0("DCopS (<", cutoff_medium, ")"))
ggsave(file.path(output_dir, "Boxplot_DCopS_MediumRange.png"), plot_box_dcops_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCopS_MediumRange.svg"), plot_box_dcops_med, width = 7, height = 6)

plot_box_rcs_med <- analyze_bins(df_med_rcs, "Rigid_Conserved_Score_0_10", paste0("RCS (<", cutoff_medium, ")"))
ggsave(file.path(output_dir, "Boxplot_RCS_MediumRange.png"), plot_box_rcs_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCS_MediumRange.svg"), plot_box_rcs_med, width = 7, height = 6)

plot_box_rcops_med <- analyze_bins(df_med_rcops, "Rigid_Coupling_Score_0_10", paste0("RCopS (<", cutoff_medium, ")"))
ggsave(file.path(output_dir, "Boxplot_RCopS_MediumRange.png"), plot_box_rcops_med, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCopS_MediumRange.svg"), plot_box_rcops_med, width = 7, height = 6)


# --- 5.3 EXTREME BINNING (< 0.5) ---
message(paste0("\n=== Running Extreme Binned Analysis (< ", cutoff_extreme, ") ==="))

plot_box_dcs_low <- analyze_bins(df_low_dcs, "Dynamic_Conserved_Score_0_10", paste0("DCS (<", cutoff_extreme, ")"))
ggsave(file.path(output_dir, "Boxplot_DCS_ExtremeLow.png"), plot_box_dcs_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCS_ExtremeLow.svg"), plot_box_dcs_low, width = 7, height = 6)

plot_box_dcops_low <- analyze_bins(df_low_dcops, "Dynamics_Coupling_Score_0_10", paste0("DCopS (<", cutoff_extreme, ")"))
ggsave(file.path(output_dir, "Boxplot_DCopS_ExtremeLow.png"), plot_box_dcops_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_DCopS_ExtremeLow.svg"), plot_box_dcops_low, width = 7, height = 6)

plot_box_rcs_low <- analyze_bins(df_low_rcs, "Rigid_Conserved_Score_0_10", paste0("RCS (<", cutoff_extreme, ")"))
ggsave(file.path(output_dir, "Boxplot_RCS_ExtremeLow.png"), plot_box_rcs_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCS_ExtremeLow.svg"), plot_box_rcs_low, width = 7, height = 6)

plot_box_rcops_low <- analyze_bins(df_low_rcops, "Rigid_Coupling_Score_0_10", paste0("RCopS (<", cutoff_extreme, ")"))
ggsave(file.path(output_dir, "Boxplot_RCopS_ExtremeLow.png"), plot_box_rcops_low, width = 7, height = 6, dpi = 300)
ggsave(file.path(output_dir, "Boxplot_RCopS_ExtremeLow.svg"), plot_box_rcops_low, width = 7, height = 6)


message("\nAnalysis Complete. Check 'stats_output' folder for plots and text logs.")

# Stop logging
sink()
