# =====================================================
# A PRIORI TESTS: FINALE VERSION (robust, N=67) - ERWEITERT
# =====================================================

library(tidyverse)
library(car)
library(readxl)

# -----------------------------------------------------
# WICHTIG: Type III braucht Sum-to-zero Kontraste
# -----------------------------------------------------

options(contrasts = c("contr.sum", "contr.poly"))

# -----------------------------------------------------
# DATEN LADEN
# -----------------------------------------------------

df <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx") %>%
  mutate(
    # ROIs
    `roi_mean ACC` = as.numeric(`roi_mean ACC`),
    `roi_mean antIns` = as.numeric(`roi_mean antIns`), 
    `roi_mean Putamen` = as.numeric(`roi_mean Putamen`),
    `roi_mean Nacc` = as.numeric(`roi_mean Nacc`),
    `roi_mean Ncau` = as.numeric(`roi_mean Ncau`),
    
    # Pathways
    `tyrosine.degradation.I` = as.numeric(`tyrosine.degradation.I`),
    `phenylalanine.degradation` = as.numeric(`phenylalanine.degradation`),
    `tryptophan.degradation` = as.numeric(`tryptophan.degradation`), 
    `X4.aminobutyrate.degradation` = as.numeric(`X4.aminobutyrate.degradation`),
    `butyrate.production.I` = as.numeric(`butyrate.production.I`),
    
    # Covariates
    age = as.numeric(age),
    bmi = as.numeric(bmi),
    cpzeq = suppressWarnings(as.numeric(cpzeq)),
    cpzeq = ifelse(is.na(cpzeq), 0, cpzeq),
    
    group2 = factor(case_when(
      group %in% c("SZ", "SZ out") ~ "Patient",
      TRUE ~ "CTRL"
    ), levels = c("CTRL", "Patient")),
    
    gender = factor(gender, levels = c(1, 2),
                    labels = c("female", "male"))
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

# -----------------------------------------------------
# DEFINITIONEN
# -----------------------------------------------------

rois <- c("roi_mean ACC", "roi_mean antIns", 
          "roi_mean Putamen", "roi_mean Nacc", "roi_mean Ncau")

priors <- c("tyrosine.degradation.I", 
            "phenylalanine.degradation", 
            "tryptophan.degradation", 
            "X4.aminobutyrate.degradation", 
            "butyrate.production.I")

results <- data.frame()

cat("\n🔬 Starte 75 Modelle (3 Varianten)...\n")

# -----------------------------------------------------
# MODELL-SCHLEIFE (robust, 3 Varianten)
# -----------------------------------------------------

for(path in priors) {
  for(roi in rois) {
    
    # ===============================
    # 1. OHNE cpzeq UND OHNE bmi
    # ===============================
    
    fml_no <- as.formula(paste0("`", roi, "` ~ `", path, 
                                "` * group2 + age + gender"))
    
    mod_no <- try(lm(fml_no, data = df), silent = TRUE)
    if(inherits(mod_no, "try-error")) next
    
    # prüfen ob Faktoren noch 2 Stufen haben
    mf_no <- model.frame(mod_no)
    if(any(sapply(mf_no[, sapply(mf_no, is.factor)], 
                  function(x) length(unique(x)) < 2))) next
    
    aov_no <- try(Anova(mod_no, type = "III"), silent = TRUE)
    if(inherits(aov_no, "try-error")) next
    
    results <- rbind(results, data.frame(
      Pathway = path,
      ROI = roi,
      p_inter = aov_no[paste0(path, ":group2"), "Pr(>F)"],
      p_main  = aov_no[path, "Pr(>F)"],
      model = "no_cpzeq_no_bmi"
    ))
    
    
    # ===============================
    # 2. OHNE cpzeq (mit bmi)
    # ===============================
    
    fml_cp_no_bmi <- as.formula(paste0("`", roi, "` ~ `", path, 
                                       "` * group2 + age + gender + bmi"))
    
    mod_cp_no_bmi <- try(lm(fml_cp_no_bmi, data = df), silent = TRUE)
    if(inherits(mod_cp_no_bmi, "try-error")) next
    
    mf_cp_no_bmi <- model.frame(mod_cp_no_bmi)
    if(any(sapply(mf_cp_no_bmi[, sapply(mf_cp_no_bmi, is.factor)], 
                  function(x) length(unique(x)) < 2))) next
    
    aov_cp_no_bmi <- try(Anova(mod_cp_no_bmi, type = "III"), silent = TRUE)
    if(inherits(aov_cp_no_bmi, "try-error")) next
    
    results <- rbind(results, data.frame(
      Pathway = path,
      ROI = roi,
      p_inter = aov_cp_no_bmi[paste0(path, ":group2"), "Pr(>F)"],
      p_main  = aov_cp_no_bmi[path, "Pr(>F)"],
      model = "no_cpzeq"
    ))
    
    
    # ===============================
    # 3. MIT cpzeq (mit bmi)
    # ===============================
    
    fml_cp <- as.formula(paste0("`", roi, "` ~ `", path, 
                                "` * group2 + age + gender + bmi + cpzeq"))
    
    mod_cp <- try(lm(fml_cp, data = df), silent = TRUE)
    if(inherits(mod_cp, "try-error")) next
    
    mf_cp <- model.frame(mod_cp)
    if(any(sapply(mf_cp[, sapply(mf_cp, is.factor)], 
                  function(x) length(unique(x)) < 2))) next
    
    aov_cp <- try(Anova(mod_cp, type = "III"), silent = TRUE)
    if(inherits(aov_cp, "try-error")) next
    
    results <- rbind(results, data.frame(
      Pathway = path,
      ROI = roi,
      p_inter = aov_cp[paste0(path, ":group2"), "Pr(>F)"],
      p_main  = aov_cp[path, "Pr(>F)"],
      model = "with_cpzeq"
    ))
  }
}

# -----------------------------------------------------
# FDR (getrennt pro Modellvariante)
# -----------------------------------------------------

results <- results %>%
  group_by(model) %>%
  mutate(
    p_inter_fdr = p.adjust(p_inter, method = "fdr"),
    p_main_fdr  = p.adjust(p_main, method = "fdr")
  ) %>%
  ungroup()

# -----------------------------------------------------
# EXPORT
# -----------------------------------------------------

output_path <- "C:/Users/User/Documents/Lara_UKF/Microbiome analysis/Analysis/combined_analysis/a_priori_75_results_N67_final.csv"

write.csv(results, output_path, row.names = FALSE)

cat("\n✅ CSV gespeichert unter:\n", output_path, "\n")
cat("Datei existiert:", file.exists(output_path), "\n")

# -----------------------------------------------------
# ZUSAMMENFASSUNG
# -----------------------------------------------------
cat("\n📊 ZUSAMMENFASSUNG:\n")
cat(sprintf("Gesamt Modelle: %d\n", nrow(results)))
cat("Pro Modelltyp:\n")
table(results$model) %>% print()

cat("\n✅ Fertig! Jetzt 3 Modelle pro Kombination:\n")
cat("- no_cpzeq_no_bmi (ohne cpzeq & bmi)\n")
cat("- no_cpzeq (mit bmi)\n") 
cat("- with_cpzeq (mit allen Covariates)\n")

library(ggplot2)
ggplot(df, aes(`tryptophan.degradation`, `roi_mean Ncau`, color=group2)) + 
  geom_point() + geom_smooth(method="lm") +
  labs(title="Interaktion: Tryptophan × group2 → Ncau")

# 1. Maßstab + Labels anpassen
ggplot(df, aes(`tryptophan.degradation`, `roi_mean Ncau`, color=group2)) + 
  geom_point(alpha=0.7, size=2) + 
  geom_smooth(method="lm", se=TRUE) +
  scale_color_manual(values=c("CTRL"="darkblue", "Patient"="darkred"),
                     labels=c("CTRL"= "Kontrollen", "Patient"= "Patienten")) +
  labs(
    title="Tryptophan-Abbau × Gruppe → Ncau",
    subtitle="Disordinale Interaktion (p=0.047)",
    x="Tryptophan Degradation (CLR)",
    y="Ncau (ROI-Mean)",
    color="Gruppe"
  ) +
  theme_minimal() +
  theme(legend.position="bottom")

# 2. Effektgröße berechnen
library(effectsize)
eta_sq <- eta_squared(lm(`roi_mean Ncau` ~ `tryptophan.degradation`*group2, df))
print(eta_sq)  # η² zeigt praktische Relevanz!
