# ==========================================================
# KORRELATIONEN: A PRIORI PATHWAYS vs BETA ROIs (N=67)
# UNADJUSTED + PARTIELL (nur age, gender, bmi) - FINALE VERSION
# ==========================================================
install.packages('tidyverse')
library(tidyverse)
library(readxl)
library(corrplot)
library(car)

outputdir <- "C:/Users/User/Documents/Analysis/Microbiome_analysis/R_ouput/a_priori_correlations"
if (!dir.exists(outputdir)) dir.create(outputdir, recursive = TRUE)

cat("Output-Ordner:", outputdir, "\n")

# -----------------------------------------------------
# 1. DATEN LADEN & VORBEREITUNG (Original-Logik, cpzeq weggelassen)
# -----------------------------------------------------

df <- read_excel("D:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx") %>%
  mutate(
    # ROIs
    `roi_mean ACC` = as.numeric(`roi_mean ACC`),
    `roi_mean antIns` = as.numeric(`roi_mean antIns`), 
    `roi_mean Putamen` = as.numeric(`roi_mean Putamen`),
    `roi_mean Nacc` = as.numeric(`roi_mean Nacc`),
    `roi_mean Ncau` = as.numeric(`roi_mean Ncau`),
    
    # Pathways (a priori)
    `tyrosine.degradation.I` = as.numeric(`tyrosine.degradation.I`),
    `phenylalanine.degradation` = as.numeric(`phenylalanine.degradation`),
    `tryptophan.degradation` = as.numeric(`tryptophan.degradation`), 
    `X4.aminobutyrate.degradation` = as.numeric(`X4.aminobutyrate.degradation`),
    `butyrate.production.I` = as.numeric(`butyrate.production.I`),
    
    # Covariates (KEIN cpzeq!)
    age = as.numeric(age),
    bmi = as.numeric(bmi),
    
    group2 = factor(case_when(
      group %in% c("SZ", "SZ out") ~ "Patient",
      TRUE ~ "CTRL"
    ), levels = c("CTRL", "Patient")),
    
    gender = factor(gender, levels = c(1, 2),
                    labels = c("female", "male")),
    gender = as.numeric(gender),
  ) %>%
  filter(complete.cases(
    age, bmi, group2, gender,
    `roi_mean ACC`, `roi_mean antIns`, 
    `roi_mean Putamen`, `roi_mean Nacc`, `roi_mean Ncau`,
    `tyrosine.degradation.I`,
    `phenylalanine.degradation`,
    `tryptophan.degradation`,
    `X4.aminobutyrate.degradation`,
    `butyrate.production.I`
  ))

cat(sprintf("✅ N = %d (CTRL:%d, Patient:%d)\n", 
            nrow(df), 
            sum(df$group2=="CTRL"), 
            sum(df$group2=="Patient")))

# Definitionen
rois <- c("roi_mean ACC", "roi_mean antIns", 
          "roi_mean Putamen", "roi_mean Nacc", "roi_mean Ncau")

priors <- c("tyrosine.degradation.I", 
            "phenylalanine.degradation", 
            "tryptophan.degradation", 
            "X4.aminobutyrate.degradation", 
            "butyrate.production.I")

# ==========================================================
# 2. NORMALITÄTSTESTEN
# ==========================================================

normtests <- data.frame()

for (var in c(rois, priors)) {
  shapiro_p <- tryCatch(shapiro.test(df[[var]])$p.value, error = function(e) NA)
  normtests <- rbind(normtests,
                     data.frame(Variable = var,
                                Shapiro_p = shapiro_p,
                                Normal = !is.na(shapiro_p) & shapiro_p > 0.05))
}

write.csv(normtests, file.path(outputdir, "01_normalitytests_pathways_rois.csv"), row.names = FALSE)

# ==========================================================
# 3. UNADJUSTED KORRELATIONEN (normal, auto Pearson/Spearman)
# ==========================================================

unadj_results <- data.frame()

for (path in priors) {
  for (roi in rois) {
    
    sh_path <- normtests$Shapiro_p[normtests$Variable == path]
    sh_roi <- normtests$Shapiro_p[normtests$Variable == roi]
    
    use_spearman <- is.na(sh_path) | sh_path < 0.05 | is.na(sh_roi) | sh_roi < 0.05
    method_used <- ifelse(use_spearman, "spearman", "pearson")
    
    corrres <- cor.test(df[[path]], df[[roi]], method = method_used)
    
    unadj_results <- rbind(unadj_results,
                           data.frame(Pathway = path,
                                      ROI = roi,
                                      Method = method_used,
                                      r = round(corrres$estimate, 3),
                                      pvalue = round(corrres$p.value, 4)))
  }
}

unadj_results <- unadj_results %>%
  mutate(p_fdr = p.adjust(pvalue, method = "fdr"))

write.csv(unadj_results, file.path(outputdir, "02_unadjusted_correlations_auto_FDR.csv"), row.names = FALSE)

# ==========================================================
# 4. PARTIELLE KORRELATIONEN (adjusted für age, gender, bmi)
# ==========================================================

cat("\n🔬 Partielle Korrelationen (age + gender + bmi)...\n")

covars <- c("age", "gender", "bmi")
partial_results <- data.frame()

for (path in priors) {
  for (roi in rois) {
    
    vars <- c(path, roi, covars)
    sub_df <- df[complete.cases(df[, vars]), vars]
    
    if (nrow(sub_df) >= 10) {
      
      # Residualisierung
      model_path <- lm(reformulate(c("age", "gender", "bmi"), response = path), data = sub_df)
      model_roi <- lm(reformulate(c("age", "gender", "bmi"), response = roi), data = sub_df)
      
      resid_path <- residuals(model_path)
      resid_roi <- residuals(model_roi)
      
      # Residuen-Normalität
      sh_path <- tryCatch(shapiro.test(resid_path)$p.value, error = function(e) 0.01)
      sh_roi <- tryCatch(shapiro.test(resid_roi)$p.value, error = function(e) 0.01)
      
      use_spearman <- sh_path < 0.05 | sh_roi < 0.05
      method_used <- ifelse(use_spearman, "spearman", "pearson")
      
      corrres <- cor.test(resid_path, resid_roi, method = method_used)
      
      partial_results <- rbind(partial_results,
                               data.frame(Pathway = path,
                                          ROI = roi,
                                          N = nrow(sub_df),
                                          Method = method_used,
                                          partial_r = round(corrres$estimate, 3),
                                          pvalue = round(corrres$p.value, 4)))
    }
  }
}

partial_results <- partial_results %>%
  mutate(p_fdr = p.adjust(pvalue, method = "fdr"))

write.csv(partial_results, 
          file.path(outputdir, "03_partial_correlations_age_gender_bmi_auto_FDR.csv"),
          row.names = FALSE)

# ==========================================================
# 5. HEATMAPS (getrennt für unadj & partial)
# ==========================================================

# Unadjusted Heatmap
unadj_cormat <- matrix(NA, nrow = length(priors), ncol = length(rois))
unadj_pmat <- matrix(NA, nrow = length(priors), ncol = length(rois))
rownames(unadj_cormat) <- priors
colnames(unadj_cormat) <- rois
rownames(unadj_pmat) <- priors
colnames(unadj_pmat) <- rois

for (i in 1:nrow(unadj_results)) {
  unadj_cormat[unadj_results$Pathway[i], unadj_results$ROI[i]] <- unadj_results$r[i]
  unadj_pmat[unadj_results$Pathway[i], unadj_results$ROI[i]] <- unadj_results$pvalue[i]
}

png(file.path(outputdir, "04_unadjusted_correlations_heatmap.png"), width = 900, height = 500, res = 100)
corrplot(unadj_cormat, p.mat = unadj_pmat, method = "color", type = "full",
         insig = "p-value", sig.level = 0.05, tl.cex = 0.9, number.cex = 0.7,
         title = "Unadjusted Korrelationen (auto Pearson/Spearman)", mar = c(0,0,3,0))
dev.off()

# Partial Heatmap
partial_cormat <- matrix(NA, nrow = length(priors), ncol = length(rois))
partial_pmat <- matrix(NA, nrow = length(priors), ncol = length(rois))
rownames(partial_cormat) <- priors
colnames(partial_cormat) <- rois
rownames(partial_pmat) <- priors
colnames(partial_pmat) <- rois

for (i in 1:nrow(partial_results)) {
  partial_cormat[partial_results$Pathway[i], partial_results$ROI[i]] <- partial_results$partial_r[i]
  partial_pmat[partial_results$Pathway[i], partial_results$ROI[i]] <- partial_results$pvalue[i]
}

png(file.path(outputdir, "05_partial_correlations_heatmap_age_gender_bmi.png"), width = 900, height = 500, res = 100)
corrplot(partial_cormat, p.mat = partial_pmat, method = "color", type = "full",
         insig = "p-value", sig.level = 0.05, tl.cex = 0.9, number.cex = 0.7,
         title = "Partielle Korrelationen\n(adjusted: age, gender, bmi)", mar = c(0,0,3,0))
dev.off()

# ==========================================================
# 6. SUMMARY
# ==========================================================

cat(sprintf("\n📊 UNADJUSTED (p<.05): %d/%d | FDR: %d/%d\n", 
            sum(unadj_results$pvalue < 0.05), nrow(unadj_results),
            sum(unadj_results$p_fdr < 0.05), nrow(unadj_results)))
cat(sprintf("📊 PARTIELL (p<.05): %d/%d | FDR: %d/%d\n", 
            sum(partial_results$pvalue < 0.05), nrow(partial_results),
            sum(partial_results$p_fdr < 0.05), nrow(partial_results)))

cat("\n✅ ALLE DATEIEN GESPEICHERT:\n")
cat("- 01_normalitytests_pathways_rois.csv\n")
cat("- 02_unadjusted_correlations_auto_FDR.csv\n")
cat("- 03_partial_correlations_age_gender_bmi_auto_FDR.csv\n")
cat("- 04_unadjusted_correlations_heatmap.png\n")
cat("- 05_partial_correlations_heatmap_age_gender_bmi.png\n")

cat("\n=== SCRIPT 100% FERTIG ===\n")
