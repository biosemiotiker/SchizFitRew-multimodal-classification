# =====================================================
# EMERGENCY BASE R VERSION - KEIN dplyr nötig! (FIXED)
# =====================================================

library(ppcor)
library(readxl)
library(readr)

# 1. PCA laden (BASE R)
pca_base <- "F:/Microbiome_analysis/Combined_analysis/Output/PCA_results_all_pathways/"
pca_file <- paste0(pca_base, "df_pca_with_rois.csv")

if(file.exists(pca_file)) {
  df_pca_raw <- read_csv(pca_file, show_col_types = FALSE)
} else {
  df_pca_raw <- read_excel(paste0(pca_base, "df_pca_with_rois.xlsx"))
}

df_pca <- data.frame(
  id = as.character(df_pca_raw$id),
  PC1 = as.numeric(as.character(df_pca_raw$PC1)),
  PC2 = as.numeric(as.character(df_pca_raw$PC2)),
  PC3 = as.numeric(as.character(df_pca_raw$PC3)),
  PC4 = as.numeric(as.character(df_pca_raw$PC4)),
  PC5 = as.numeric(as.character(df_pca_raw$PC5))
)
cat("✅ PCA geladen: N =", nrow(df_pca), "\n")

# 2. Fitness laden (BASE R)
df_fitness_raw <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")

df_fitness <- data.frame(
  id = as.character(df_fitness_raw$id),
  age = as.numeric(as.character(df_fitness_raw$age)),
  gender = as.numeric(df_fitness_raw$gender),
  bmi = as.numeric(as.character(df_fitness_raw$bmi)),
  cpzeq = as.numeric(as.character(df_fitness_raw$cpzeq)),
  group = as.character(df_fitness_raw$group),
  VO_max = as.numeric(as.character(df_fitness_raw$VO_max)),
  Jump = as.numeric(as.character(df_fitness_raw$Jump)),
  Hand_mean = as.numeric(as.character(df_fitness_raw$Hand_mean)),
  Sum_Skinfolds = as.numeric(as.character(df_fitness_raw$Sum_Skinfolds)),
  Brozek_BF_age = as.numeric(as.character(df_fitness_raw$Brozek_BF_age)),
  WHR = as.numeric(as.character(df_fitness_raw$WHR)),
  WHtR = as.numeric(as.character(df_fitness_raw$WHtR)),
  PC1_fitness = as.numeric(as.character(df_fitness_raw$PC1_fitness)),
  PC2_agility = as.numeric(as.character(df_fitness_raw$PC2_agility))
)

# Bereinigung
df_fitness$cpzeq[is.na(df_fitness$cpzeq)] <- 0
df_fitness$gender_num <- df_fitness$gender - 1
df_fitness$group2 <- ifelse(df_fitness$group %in% c("SZ", "SZ out"), "Patient", "CTRL")

cat("✅ Fitness geladen: N =", nrow(df_fitness), "\n")

# 3. MERGE (BASE R)
df <- merge(df_pca, df_fitness, by = "id")

complete_cols <- c("PC1", "PC2", "PC3", "PC4", "PC5", "age", "bmi", "gender_num",
                   "VO_max", "Jump", "Hand_mean", "Sum_Skinfolds", 
                   "Brozek_BF_age", "WHR", "WHtR", "PC1_fitness", "PC2_agility")

df <- df[complete.cases(df[, complete_cols]), ]
cat("✅ Nach Merge/Filter: N =", nrow(df), "\n")

# Output Dir
output_dir <- "F:/Microbiome_analysis/Combined_analysis/Output/PCA_Pathways_Fitness_partial_corr"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# === 1. PARTIELLE KORRELATIONEN (ROBUST - FIXED) ===
pcs <- c("PC1", "PC2", "PC3", "PC4", "PC5")
fitness <- c("VO_max", "Jump", "Hand_mean", "Sum_Skinfolds", "Brozek_BF_age",
             "WHR", "WHtR", "PC1_fitness", "PC2_agility")

cat("🔄 PARTIELLE KORRELATIONEN (mit Fehlerbehandlung)\n")

# Initialisiere mit korrekten Spalten!
results_withBMI <- data.frame(PC = character(), Fitness = character(), 
                              r_partial = numeric(), p_value = numeric(), n = numeric(),
                              stringsAsFactors = FALSE)
results_noBMI <- data.frame(PC = character(), Fitness = character(), 
                            r_partial = numeric(), p_value = numeric(), n = numeric(),
                            stringsAsFactors = FALSE)

# MIT BMI
for(pc in pcs) {
  for(fit_var in fitness) {
    cat(sprintf("  %s vs %s (BMI)\n", pc, fit_var))
    covars <- as.matrix(df[, c("age", "gender_num", "bmi")])
    tryCatch({
      result <- pcor.test(df[[pc]], df[[fit_var]], covars)
      results_withBMI <- rbind(results_withBMI, data.frame(
        PC = pc, Fitness = fit_var, r_partial = result$estimate, 
        p_value = result$p.value, n = result$n, stringsAsFactors = FALSE
      ))
    }, error = function(e) {
      cat(sprintf("    ❌ %s vs %s: %s\n", pc, fit_var, e$message))
      results_withBMI <<- rbind(results_withBMI, data.frame(  # <<- nur hier lokal ok
        PC = pc, Fitness = fit_var, r_partial = NA, p_value = NA, n = nrow(df),
        stringsAsFactors = FALSE
      ))
    })
  }
}

# OHNE BMI
for(pc in pcs) {
  for(fit_var in fitness) {
    cat(sprintf("  %s vs %s (noBMI)\n", pc, fit_var))
    covars <- as.matrix(df[, c("age", "gender_num")])
    tryCatch({
      result <- pcor.test(df[[pc]], df[[fit_var]], covars)
      results_noBMI <- rbind(results_noBMI, data.frame(
        PC = pc, Fitness = fit_var, r_partial = result$estimate, 
        p_value = result$p.value, n = result$n, stringsAsFactors = FALSE
      ))
    }, error = function(e) {
      cat(sprintf("    ❌ %s vs %s: %s\n", pc, fit_var, e$message))
      results_noBMI <<- rbind(results_noBMI, data.frame(
        PC = pc, Fitness = fit_var, r_partial = NA, p_value = NA, n = nrow(df),
        stringsAsFactors = FALSE
      ))
    })
  }
}

# === 2. GRUPPENWEISE KORRELATIONEN ===
cat("\n🔄 GRUPPENWEISE (Patient vs CTRL)\n")
group_results <- data.frame(PC = character(), Fitness = character(),
                            r_patient = numeric(), p_patient = numeric(), n_patient = numeric(),
                            r_control = numeric(), p_control = numeric(), n_control = numeric(),
                            stringsAsFactors = FALSE)

df_patient <- df[df$group2 == "Patient", ]
df_ctrl <- df[df$group2 == "CTRL", ]

for(pc in pcs) {
  for(fit_var in fitness) {
    # Patienten
    r_pat <- cor(df_patient[[pc]], df_patient[[fit_var]], use = "complete.obs")
    p_pat <- cor.test(df_patient[[pc]], df_patient[[fit_var]])$p.value
    
    # Controls
    r_ctrl <- cor(df_ctrl[[pc]], df_ctrl[[fit_var]], use = "complete.obs")
    p_ctrl <- cor.test(df_ctrl[[pc]], df_ctrl[[fit_var]])$p.value
    
    group_results <- rbind(group_results, data.frame(
      PC = pc, Fitness = fit_var,
      r_patient = round(r_pat, 3), p_patient = p_pat, n_patient = nrow(df_patient),
      r_control = round(r_ctrl, 3), p_control = p_ctrl, n_control = nrow(df_ctrl),
      stringsAsFactors = FALSE
    ))
  }
}

# FDR Korrektur (nur gültige p-Werte)
# === FDR FIX (integriert) ===
if(!"p_adj" %in% names(results_withBMI)) results_withBMI$p_adj <- NA
if(!"p_adj" %in% names(results_noBMI)) results_noBMI$p_adj <- NA

valid_idx_bmi <- !is.na(results_withBMI$p_value)
if(length(valid_idx_bmi) > 0) {
  results_withBMI$p_adj[valid_idx_bmi] <- p.adjust(results_withBMI$p_value[valid_idx_bmi], "BH")
}

valid_idx_nobmi <- !is.na(results_noBMI$p_value)
if(length(valid_idx_nobmi) > 0) {
  results_noBMI$p_adj[valid_idx_nobmi] <- p.adjust(results_noBMI$p_value[valid_idx_nobmi], "BH")
}

# SAVE ALLES
write.csv(results_withBMI, file.path(output_dir, "1_PARTIAL_with_BMI.csv"), row.names = FALSE)
write.csv(results_noBMI, file.path(output_dir, "2_PARTIAL_no_BMI.csv"), row.names = FALSE)
write.csv(group_results, file.path(output_dir, "3_GROUPWISE_correlations.csv"), row.names = FALSE)

# SUMMARY
summary_all <- merge(results_withBMI[, c("PC", "Fitness", "r_partial", "p_value", "p_adj")], 
                     results_noBMI[, c("PC", "Fitness", "r_partial", "p_value", "p_adj")], 
                     by = c("PC", "Fitness"), suffixes = c("_BMI", "_noBMI"))
summary_all <- merge(summary_all, group_results[, c("PC", "Fitness", "r_patient", "r_control")], by = c("PC", "Fitness"))

write.csv(summary_all[order(summary_all$p_value_BMI), ], 
          file.path(output_dir, "SUMMARY_ALL.csv"), row.names = FALSE)

cat("\n🎉 FERTIG! 4 CSV Files erstellt:\n")
cat("✅ N =", nrow(df), "\n")
cat("📊 Signifikant mit BMI (raw p<0.05):", sum(results_withBMI$p_value < 0.05, na.rm = TRUE), "\n")
cat("📊 Signifikant ohne BMI (raw p<0.05):", sum(results_noBMI$p_value < 0.05, na.rm = TRUE), "\n")

# TOP RESULTS
cat("\n🔥 TOP 10 mit BMI:\n")
print(head(results_withBMI[order(results_withBMI$p_value), ], 10))

cat("\n🔥 GRUPPENUNTERSCHIEDE (Δr > 0.2):\n")
group_diff <- group_results[abs(group_results$r_patient - group_results$r_control) > 0.2, ]
print(group_diff[order(abs(group_diff$r_patient - group_diff$r_control), decreasing = TRUE), ])