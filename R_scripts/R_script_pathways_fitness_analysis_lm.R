# =====================================================
# FINAL v2: Pathway-PCs vs Fitness 
# N=67
# =====================================================

library(tidyverse)
library(car)
library(readxl)
library(readr)



pca_base <- "F:/Microbiome_analysis/Combined_analysis/Output/PCA_results_all_pathways/"

# Versuche CSV → XLSX → Fehler
if(file.exists(paste0(pca_base, "df_pca_with_rois.csv"))) {
  df_pca <- read_csv(paste0(pca_base, "df_pca_with_rois.csv"), show_col_types = FALSE)
} else if(file.exists(paste0(pca_base, "df_pca_with_rois.xlsx"))) {
  df_pca <- read_excel(paste0(pca_base, "df_pca_with_rois.xlsx"))
} else {
  stop("❌ df_pca_with_rois.* nicht gefunden! Prüfe Ordner.")
}

df_pca <- df_pca %>% select(id, PC1, PC2, PC3, PC4, PC5) %>% mutate(across(PC1:PC5, as.numeric))
cat("✅ PCs:", nrow(df_pca), "Zeilen geladen\n")


  

df_fitness <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx") %>%
  select(id, age, gender, bmi, cpzeq, group,
         VO_max, Jump, `Hand_mean`, `Sum_Skinfolds`, `Brozek_BF_age`,
         WHR, `WHtR`, PC1_fitness, PC2_agility) %>%  # PC1_fitness OK [file:2]
  mutate(
    # Sichere Konvertierung (NA bei Fehlern)
    across(c(age, bmi, cpzeq, VO_max, Jump, `Hand_mean`, `Sum_Skinfolds`, 
             `Brozek_BF_age`, WHR, `WHtR`, PC1_fitness, PC2_agility), 
           ~ as.numeric(as.character(.x))),
    cpzeq = ifelse(is.na(cpzeq), 0, cpzeq),
    group2 = factor(case_when(group %in% c("SZ", "SZ out") ~ "Patient", TRUE ~ "CTRL"),
                    levels = c("CTRL", "Patient")),
    gender = factor(gender, levels = c(1, 2), labels = c("female", "male"))
  )

# Check NAs
print(colSums(is.na(df_fitness)))


# MERGE + FILTER (FIXED)
df <- df_pca %>% 
  left_join(df_fitness, by = "id") %>%
  filter(complete.cases(PC1, PC2, PC3, PC4, PC5,  # Explizit einzeln!
                        age, bmi, group2, gender,
                        VO_max, Jump, `Hand_mean`, `Sum_Skinfolds`, 
                        `Brozek_BF_age`, WHR, `WHtR`, PC1_fitness, PC2_agility))

cat(sprintf("✅ N = %d\n", nrow(df)))

# DEFINITIONEN (schon da)
pcs <- c("PC1", "PC2", "PC3", "PC4", "PC5")
fitness <- c("VO_max", "Jump", "Hand_mean", "Sum_Skinfolds", "Brozek_BF_age",
             "WHR", "WHtR", "PC1_fitness", "PC2_agility")

results <- data.frame()
sig_results <- data.frame()

# DOPPELTER LOOP: Alle PCs × Alle Fitness
for(pc in pcs) {
  for(fit_var in fitness) {
    cat(sprintf("🔄 %s vs %s\n", pc, fit_var))
    
    formula_str <- sprintf("%s ~ group2 + age + gender + bmi + %s", pc, fit_var)
    model <- lm(as.formula(formula_str), data = df)
    
    summary_mod <- summary(model)
    p_group <- summary_mod$coefficients["group2Patient", "Pr(>|t|)"]
    r2 <- summary_mod$r.squared
    
    results <- rbind(results, data.frame(
      PC = pc, Fitness = fit_var, 
      p_group = p_group, R2 = r2,
      beta_group = summary_mod$coefficients["group2Patient", "Estimate"]
    ))
    
    # Signifikante speichern (p<0.05)
    if(p_group < 0.05) {
      sig_results <- rbind(sig_results, data.frame(
        PC = pc, Fitness = fit_var, p = p_group, R2 = r2
      ))
    }
  }
}

# OUTPUT
write_csv(results, "F:/Microbiome_analysis/Combined_analysis/FINAL_pcs_vs_fitness_ROBUST.csv")
write_csv(sig_results, "F:/Microbiome_analysis/Combined_analysis/SIGNIFICANT_results.csv")

cat("✅ Vollständig! Results:", nrow(results), "Modelle\n")
cat("📊 Signifikant:", nrow(sig_results), "\n")
print(sig_results)




# === OHNE BMI: PCs vs Fitness ===
cat("🔄 OHNE BMI: Alle PCs × Fitness\n")

results_noBMI <- data.frame()
sig_results_noBMI <- data.frame()

for(pc in pcs) {
  for(fit_var in fitness) {
    cat(sprintf("  %s ~ group2 + age + gender + %s\n", pc, fit_var))
    
    formula_str <- sprintf("%s ~ group2 + age + gender + %s", pc, fit_var)
    model <- lm(as.formula(formula_str), data = df)
    
    summary_mod <- summary(model)
    p_group <- summary_mod$coefficients["group2Patient", "Pr(>|t|)"]
    r2 <- summary_mod$r.squared
    
    results_noBMI <- rbind(results_noBMI, data.frame(
      PC = pc, Fitness = fit_var, 
      p_group = p_group, R2 = r2,
      beta_group = summary_mod$coefficients["group2Patient", "Estimate"]
    ))
    
    if(p_group < 0.05) {
      sig_results_noBMI <- rbind(sig_results_noBMI, data.frame(
        PC = pc, Fitness = fit_var, p = p_group, R2 = r2
      ))
    }
  }
}

# Sichere Ausgabe
write_csv(results_noBMI, "F:/Microbiome_analysis/Combined_analysis/FINAL_pcs_vs_fitness_NO_BMI.csv")
write_csv(sig_results_noBMI, "F:/Microbiome_analysis/Combined_analysis/SIGNIFICANT_noBMI.csv")

cat("✅ NO BMI Results:", nrow(results_noBMI), "Modelle\n")
cat("📊 NO BMI Signifikant:", nrow(sig_results_noBMI), "\n")
print(head(results_noBMI))
print(sig_results_noBMI)

library(dplyr)

# ROBUST (mit BMI)
results <- results %>% 
  mutate(p_adj = p.adjust(p_group, method = "BH")) %>%
  arrange(p_group)

# NO BMI  
results_noBMI <- results_noBMI %>% 
  mutate(p_adj = p.adjust(p_group, method = "BH")) %>%
  arrange(p_group)

# Überschreiben
write_csv(results, "F:/Microbiome_analysis/Combined_analysis/FINAL_pcs_vs_fitness_ROBUST_FDR.csv")
write_csv(results_noBMI, "F:/Microbiome_analysis/Combined_analysis/FINAL_pcs_vs_fitness_NO_BMI_FDR.csv")

# Trends (p_adj < 0.1)
results %>% filter(p_adj < 0.1) %>% select(PC, Fitness, p_group, p_adj)

