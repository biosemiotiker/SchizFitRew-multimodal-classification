# ==========================================================
# Pathways vs Fitness: Korrelationen + Partielle (AUTO) - FINAL PERFECT
# ==========================================================
install.packages("pheatmap")
install.packages("RColorBrewer")  
# Pakete laden
library(readxl)
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(tidyr)
library(tibble)
library(pheatmap)      # 🔥 NEU: für perfekte 5x9 Heatmap
library(RColorBrewer)  # 🔥 NEU: bessere Farben

# Daten laden und vorbereiten
df <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx") %>%
  mutate(
    `tyrosine.degradation.I`     = as.numeric(`tyrosine.degradation.I`),
    `phenylalanine.degradation`  = as.numeric(`phenylalanine.degradation`),
    `tryptophan.degradation`     = as.numeric(`tryptophan.degradation`),
    `X4.aminobutyrate.degradation` = as.numeric(`X4.aminobutyrate.degradation`),
    `butyrate.production.I`      = as.numeric(`butyrate.production.I`),
    
    VO_max         = as.numeric(VO_max),
    Jump           = as.numeric(Jump),
    Hand_mean      = as.numeric(`Hand_mean`),
    Sum_Skinfolds  = as.numeric(`Sum_Skinfolds`),
    Brozek_BF_age  = as.numeric(`Brozek_BF_age`),
    WHR            = as.numeric(WHR),
    WHtR           = as.numeric(`WHtR`),
    PC1_fitness    = as.numeric(PC1_fitness),
    PC2_agility    = as.numeric(PC2_agility),
    
    age   = as.numeric(age),
    bmi   = as.numeric(bmi),
    cpzeq = suppressWarnings(as.numeric(cpzeq)),
    cpzeq = ifelse(is.na(cpzeq), 0, cpzeq),
    
    group2 = factor(case_when(
      group %in% c("SZ", "SZ out") ~ "Patient",
      TRUE ~ "CTRL"
    ), levels = c("CTRL", "Patient")),
    
    gender = factor(gender, levels = c(1, 2), labels = c("female", "male"))
  ) %>%
  filter(complete.cases(
    age, bmi, group2, gender,
    `tyrosine.degradation.I`, `phenylalanine.degradation`, `tryptophan.degradation`,
    `X4.aminobutyrate.degradation`, `butyrate.production.I`,
    VO_max, Jump, Hand_mean, Sum_Skinfolds, Brozek_BF_age, WHR, WHtR, PC1_fitness, PC2_agility
  ))

cat(sprintf("✅ N = %d (CTRL: %d, Patient: %d)\n", nrow(df), sum(df$group2 == "CTRL"), sum(df$group2 == "Patient")))

# Output-Verzeichnis
outputdir <- "F:/Microbiome_analysis/Combined_analysis/Output/a_priori_pathways_fitness_partial_corr/"
if (!dir.exists(outputdir)) dir.create(outputdir, recursive = TRUE)

# VARIABLEN
priors <- c("tyrosine.degradation.I", "phenylalanine.degradation", "tryptophan.degradation", 
            "X4.aminobutyrate.degradation", "butyrate.production.I")
fitness <- c("VO_max", "Jump", "Hand_mean", "Sum_Skinfolds", "Brozek_BF_age", "WHR", "WHtR", "PC1_fitness", "PC2_agility")
covars <- c("age", "gender", "bmi")

cat(sprintf("📊 Analysiere %d Pathways × %d Fitness-Variablen\n", length(priors), length(fitness)))

# 1. NORMALITÄTSTESTEN
cat("🔍 Normalitätstests...\n")
normtests <- data.frame()
for (var in c(priors, fitness)) {
  shapiro_p <- tryCatch(shapiro.test(df[[var]])$p.value, error = function(e) NA)
  normtests <- rbind(normtests, data.frame(Variable = var, Shapiro_p = shapiro_p))
}
normtests$Normal <- normtests$Shapiro_p >= 0.05
write_csv(normtests, file.path(outputdir, "01_normality_tests.csv"))
cat("✅ Normalitätstests gespeichert\n")

# 2. PARTIELLE KORRELATIONEN - BUGFIX VERSION
cat("🔗 Partielle Korrelationen (BUGFIX)...\n")
partial_results <- data.frame()

for (path in priors) {
  for (fit_var in fitness) {
    # BUGFIX: Explizite numerische Subset
    sub_df <- df[, c(path, fit_var, "age", "bmi", "gender")]
    sub_df[] <- lapply(sub_df, as.numeric)  # ALLE zu numeric!
    sub_df <- sub_df[complete.cases(sub_df), ]
    
    if (nrow(sub_df) >= 10) {
      resid_path <- residuals(lm(paste(path, "~ age + bmi + gender"), sub_df))
      resid_fit  <- residuals(lm(paste(fit_var, "~ age + bmi + gender"), sub_df))
      
      sh_p <- tryCatch(shapiro.test(resid_path)$p.value, error=function(e)0.01)
      sh_f <- tryCatch(shapiro.test(resid_fit)$p.value, error=function(e)0.01)
      method <- ifelse(sh_p<0.05 | sh_f<0.05, "spearman", "pearson")
      
      cor_res <- cor.test(resid_path, resid_fit, method=method)
      
      partial_results <- rbind(partial_results, data.frame(
        Pathway=path, Fitness=fit_var, N=nrow(sub_df),
        Method=method, partial_r=round(cor_res$estimate,3),
        pvalue=round(cor_res$p.value,4), stringsAsFactors=FALSE
      ))
    }
  }
}


partial_results$p_fdr <- p.adjust(partial_results$pvalue, "fdr")
partial_results$significant <- partial_results$pvalue < 0.05
write_csv(partial_results, file.path(outputdir, "02_partial_correlations_FDR.csv"))

# 3. RESULTS SUMMARY
cat("\n📈 ERGEBNISSE:\n")
cat(sprintf("✅ Signifikante (p<0.05): %d\n", sum(partial_results$pvalue < 0.05)))
cat(sprintf("✅ Signifikante (FDR): %d\n", sum(partial_results$p_fdr < 0.05)))
cat("\n🔝 Top-5:\n")
print(head(partial_results[order(partial_results$pvalue), c("Pathway", "Fitness", "partial_r", "pvalue", "p_fdr")], 5))

# 🔥 4. PERFEKTE 5x9 HEATMAP (ALLE Variablen!)
cat("\n📊 Erstelle VOLLE 5x9 Heatmap...\n")

# VOLLE Matrix (KEINE fehlenden Spalten!)
full_data <- expand_grid(Pathway = priors, Fitness = fitness) %>%
  left_join(partial_results, by = c("Pathway", "Fitness")) %>%
  mutate(
    partial_r = ifelse(is.na(partial_r), 0, partial_r),
    pvalue = ifelse(is.na(pvalue), 1, pvalue),
    N = ifelse(is.na(N), 0, N),
    sig_star = case_when(
      pvalue < 0.001 ~ "***", pvalue < 0.01 ~ "**", 
      pvalue < 0.05 ~ "*", TRUE ~ ""
    )
  )

full_cormat <- full_data %>%
  select(Pathway, Fitness, partial_r) %>%
  pivot_wider(names_from = Fitness, values_from = partial_r) %>%
  column_to_rownames("Pathway") %>% as.matrix()

# 🔥 FEHLER FIX: Annotation korrekt erstellen
cat("\n🔧 Annotation fixen (9 Spalten)...\n")

# Dataframe für ALLE 9 Fitness-Variablen (rownames = Fitness-Namen)
fitness_annot <- data.frame(
  Fitness = fitness,
  Sig = "ns"
)
rownames(fitness_annot) <- fitness  # ✅ Perfekt unique!

# Signifikante updaten
sig_rows <- partial_results[partial_results$significant, "Fitness"]
fitness_annot$Sig[rownames(fitness_annot) %in% sig_rows] <- "p<0.05"

ann_colors <- list(Sig = c("p<0.05" = "red", "ns" = "grey80"))

# 💎 pheatmap (jetzt funktioniert!)
png(file.path(outputdir, "04_FULL_5x9_heatmap_perfect.png"), width=1400, height=700, res=200)
pheatmap(full_cormat, scale="none", cluster_rows=TRUE, cluster_cols=TRUE,
         color=colorRampPalette(rev(brewer.pal(11,"RdBu")))(100),
         annotation_col=fitness_annot, annotation_colors=ann_colors,
         fontsize_row=11, fontsize_col=9,
         main="Partielle Korrelationen 5×9 KOMPLETT\nage/gender/bmi adj. | Rot=p<0.05",
         border_color=NA, cellwidth=25, cellheight=40)
dev.off()

# Signifikanz-Tabelle
sig_table <- full_data %>% filter(pvalue < 0.05) %>% 
  mutate(sig_star = case_when(pvalue < 0.001 ~ "***", pvalue < 0.01 ~ "**", pvalue < 0.05 ~ "*", TRUE ~ "")) %>%
  select(Pathway, Fitness, partial_r, pvalue, N, sig_star) %>% arrange(pvalue)
write_csv(sig_table, file.path(outputdir, "05_sig_stars.csv"))

cat("\n🎉 HEATMAP & SIG-TABELLE FERTIG!\n")
print("Top Signifikante:")
print(sig_table)
cat(sprintf("📁 Neue Dateien:\n- 04_FULL_5x9_heatmap_perfect.png ⭐\n- 05_sig_stars.csv\n"))


cat("\n🎉 FINAL PERFECT! Dateien:\n")
cat("- 01_normality_tests.csv\n- 02_partial_correlations_FDR.csv\n")
cat("- 03_old_heatmap.png (alt)\n- 04_FULL_5x9_heatmap_perfect.png ⭐\n")
if(nrow(sig_table) > 0) cat("- 05_sig_stars.csv\n")
