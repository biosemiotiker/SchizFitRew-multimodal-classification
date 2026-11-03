# ==========================================================
# FINAL SCRIPT: Normalität, Korrelationen, Partielle (auto Pearson/Spearman)
# ==========================================================

# Libraries
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(tidyr)
library(tibble)

outputdir <- "C:/Users/User/Documents/Analysis/Microbiome_analysis/R_ouput/correlation_results"
if (!dir.exists(outputdir)) dir.create(outputdir, recursive = TRUE)

cat("Output-Ordner:", outputdir, "\n")

# ==========================================================
# 1. DATEN LADEN
# ==========================================================

dfpca <- read_csv("C:/Users/User/Documents/Analysis/Microbiome_analysis/R_ouput/PCA_results_all_pathways/df_pca_with_rois.csv",
                  show_col_types = FALSE)

roicols <- c("roi_mean ACC", "roi_mean antIns", "roi_mean Putamen", 
             "roi_mean Nacc", "roi_mean Ncau")
pcs <- c("PC1", "PC2", "PC3", "PC4", "PC5")

# Kovariaten vorbereiten
dfpca$Geschlecht <- trimws(dfpca$Geschlecht)
dfpca$Geschlecht <- ifelse(dfpca$Geschlecht == "M", 0,
                           ifelse(dfpca$Geschlecht == "F", 1, NA))
dfpca$Geschlecht <- as.numeric(dfpca$Geschlecht)
dfpca$Alter <- as.numeric(dfpca$Alter)
dfpca$BMI <- as.numeric(dfpca$BMI)

cat("Daten geladen: N =", nrow(dfpca), "\n")

# ==========================================================
# 2. NORMALITÄTSTESTEN
# ==========================================================

normtests <- data.frame()

for (var in c(pcs, roicols)) {
  if (var %in% colnames(dfpca)) {
    shapiro_p <- tryCatch(shapiro.test(dfpca[[var]])$p.value,
                          error = function(e) NA)
    normtests <- rbind(normtests,
                       data.frame(Variable = var,
                                  Shapiro_p = shapiro_p,
                                  Normal = !is.na(shapiro_p) & shapiro_p > 0.05))
  }
}

write.csv(normtests, file.path(outputdir, "normalitytests.csv"), row.names = FALSE)

# ==========================================================
# 3. UNADJUSTED KORRELATIONEN (AUTO)
# ==========================================================

results <- data.frame()

for (pc in pcs) {
  for (roi in roicols) {
    if (pc %in% colnames(dfpca) & roi %in% colnames(dfpca)) {
      
      sh_pc <- normtests$Shapiro_p[normtests$Variable == pc]
      sh_roi <- normtests$Shapiro_p[normtests$Variable == roi]
      
      use_spearman <- is.na(sh_pc) | sh_pc < 0.05 |
        is.na(sh_roi) | sh_roi < 0.05
      
      method_used <- ifelse(use_spearman, "spearman", "pearson")
      
      corrres <- cor.test(dfpca[[pc]], dfpca[[roi]], method = method_used)
      
      results <- rbind(results,
                       data.frame(PC = pc,
                                  ROI = roi,
                                  Method = method_used,
                                  r = round(corrres$estimate, 3),
                                  pvalue = round(corrres$p.value, 4)))
    }
  }
}

write.csv(results, file.path(outputdir, "pc_roi_correlations_auto.csv"), row.names = FALSE)

# ==========================================================
# 4. PARTIELLE KORRELATIONEN (AUTO METHOD)
# ==========================================================

cat("\n=== PARTIELLE KORRELATIONEN (AUTO Pearson/Spearman) ===\n")

covars <- c("Alter", "Geschlecht", "BMI")
partial_results <- data.frame()

for (pc in pcs) {
  for (roi in roicols) {
    
    vars <- c(pc, roi, covars)
    sub_df <- dfpca[complete.cases(dfpca[, vars]), vars]
    
    if (nrow(sub_df) >= 10) {
      
      # Residualisierung
      model_pc <- lm(as.formula(paste0("`", pc, "` ~ Alter + Geschlecht + BMI")),
                     data = sub_df)
      
      model_roi <- lm(as.formula(paste0("`", roi, "` ~ Alter + Geschlecht + BMI")),
                      data = sub_df)
      
      resid_pc <- residuals(model_pc)
      resid_roi <- residuals(model_roi)
      
      # Normalität der Residuen prüfen
      sh_pc <- shapiro.test(resid_pc)$p.value
      sh_roi <- shapiro.test(resid_roi)$p.value
      
      use_spearman <- sh_pc < 0.05 | sh_roi < 0.05
      method_used <- ifelse(use_spearman, "spearman", "pearson")
      
      corrres <- cor.test(resid_pc, resid_roi, method = method_used)
      
      partial_results <- rbind(partial_results,
                               data.frame(PC = pc,
                                          ROI = roi,
                                          N = nrow(sub_df),
                                          Method = method_used,
                                          partial_r = round(corrres$estimate, 3),
                                          pvalue = round(corrres$p.value, 4)))
    }
  }
}

write.csv(partial_results,
          file.path(outputdir,
                    "pc_roi_partial_correlations_AUTO_Alter_Geschlecht_BMI.csv"),
          row.names = FALSE)

# ==========================================================
# 5. FDR KORREKTUR
# ==========================================================

partial_results$p_fdr <- p.adjust(partial_results$pvalue, method = "fdr")

write.csv(partial_results,
          file.path(outputdir,
                    "pc_roi_partial_correlations_AUTO_FDR.csv"),
          row.names = FALSE)

# ==========================================================
# 6. HEATMAP PARTIELL
# ==========================================================

pc_order <- unique(partial_results$PC)
roi_order <- unique(partial_results$ROI)

partial_cormat <- matrix(NA,
                         nrow = length(pc_order),
                         ncol = length(roi_order))
partial_pmat <- matrix(NA,
                       nrow = length(pc_order),
                       ncol = length(roi_order))

rownames(partial_cormat) <- pc_order
colnames(partial_cormat) <- roi_order
rownames(partial_pmat) <- pc_order
colnames(partial_pmat) <- roi_order

for (i in 1:nrow(partial_results)) {
  partial_cormat[partial_results$PC[i],
                 partial_results$ROI[i]] <- partial_results$partial_r[i]
  partial_pmat[partial_results$PC[i],
               partial_results$ROI[i]] <- partial_results$pvalue[i]
}

png(file.path(outputdir,
              "pc_roi_partial_correlations_heatmap_AUTO.png"),
    width = 800, height = 600, res = 100)

corrplot(partial_cormat,
         p.mat = partial_pmat,
         method = "color",
         type = "upper",
         insig = "p-value",
         sig.level = 0.05,
         tl.cex = 0.8,
         tl.col = "black",
         title = "Partielle Korrelationen (Auto Pearson/Spearman)",
         mar = c(0,0,3,0))

dev.off()

cat("\nSignifikante partielle (p < .05):",
    sum(partial_results$pvalue < 0.05), "von",
    nrow(partial_results), "\n")

cat("=== SCRIPT FERTIG ===\n")
