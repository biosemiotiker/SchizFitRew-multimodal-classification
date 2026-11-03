# =====================================================
# A PRIORI: Pathways vs Fitness (N ~ 67)
# 5 Pathways × Fitness-Variablen
# Zwei Modellvarianten: mit BMI, ohne BMI
# =====================================================

library(tidyverse)
library(car)
library(readxl)
library(readr)

options(contrasts = c("contr.sum", "contr.poly"))

# -----------------------------------------------------
# DATEN LADEN
# -----------------------------------------------------

df <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx") %>%
  mutate(
    # Pathways (a priori)
    `tyrosine.degradation.I`     = as.numeric(`tyrosine.degradation.I`),
    `phenylalanine.degradation`  = as.numeric(`phenylalanine.degradation`),
    `tryptophan.degradation`     = as.numeric(`tryptophan.degradation`),
    `X4.aminobutyrate.degradation` = as.numeric(`X4.aminobutyrate.degradation`),
    `butyrate.production.I`      = as.numeric(`butyrate.production.I`),
    
    # Fitness-Variablen
    VO_max         = as.numeric(VO_max),
    Jump           = as.numeric(Jump),
    Hand_mean      = as.numeric(`Hand_mean`),
    Sum_Skinfolds  = as.numeric(`Sum_Skinfolds`),
    Brozek_BF_age  = as.numeric(`Brozek_BF_age`),
    WHR            = as.numeric(WHR),
    WHtR           = as.numeric(`WHtR`),
    PC1_fitness    = as.numeric(PC1_fitness),
    PC2_agility    = as.numeric(PC2_agility),
    
    # Covariates
    age   = as.numeric(age),
    bmi   = as.numeric(bmi),
    cpzeq = suppressWarnings(as.numeric(cpzeq)),
    cpzeq = ifelse(is.na(cpzeq), 0, cpzeq),
    
    group2 = factor(case_when(
      group %in% c("SZ", "SZ out") ~ "Patient",
      TRUE ~ "CTRL"
    ), levels = c("CTRL", "Patient")),
    
    gender = factor(gender, levels = c(1, 2),
                    labels = c("female", "male"))
  ) %>%
  # vollständige Fälle für alle interessierenden Variablen
  filter(complete.cases(
    age, bmi, group2, gender,
    `tyrosine.degradation.I`,
    `phenylalanine.degradation`,
    `tryptophan.degradation`,
    `X4.aminobutyrate.degradation`,
    `butyrate.production.I`,
    VO_max, Jump, Hand_mean, Sum_Skinfolds,
    Brozek_BF_age, WHR, WHtR, PC1_fitness, PC2_agility
  ))

cat(sprintf("✅ N = %d (CTRL:%d, Patient:%d)\n",
            nrow(df),
            sum(df$group2 == \"CTRL\"),
            sum(df$group2 == \"Patient\")))

# -----------------------------------------------------
# DEFINITIONEN
# -----------------------------------------------------

priors <- c(
  "tyrosine.degradation.I",
  "phenylalanine.degradation",
  "tryptophan.degradation",
  "X4.aminobutyrate.degradation",
  "butyrate.production.I"
)

fitness <- c(
  "VO_max", "Jump", "Hand_mean", "Sum_Skinfolds",
  "Brozek_BF_age", "WHR", "WHtR", "PC1_fitness", "PC2_agility"
)

results <- data.frame()

cat("\n🔬 Starte Modelle: Pathways × Fitness (mit/ohne BMI)...\n")

# -----------------------------------------------------
# MODELL-SCHLEIFE
# -----------------------------------------------------
# Struktur analog zu deinem PCA-Fitness-Skript,
# aber hier: Outcome = Pathway, Prädiktor = Fitnessvar + group2 + Covariates

for(path in priors) {
  for(fit_var in fitness) {

    # ===============================
    # 1. MIT BMI
    # ===============================
    cat(sprintf("🔄 MIT BMI: %s ~ %s + group2 + age + gender + bmi\n",
                path, fit_var))

    fml_bmi <- as.formula(
      paste0("`", path, "` ~ ", fit_var,
             " + group2 + age + gender + bmi")
    )

    mod_bmi <- try(lm(fml_bmi, data = df), silent = TRUE)
    if (inherits(mod_bmi, "try-error")) next

    sm_bmi <- summary(mod_bmi)

    # Robust: nur Koeffizienten verwenden, die existieren
    coefs_bmi <- sm_bmi$coefficients
    p_group <- if ("group2Patient" %in% rownames(coefs_bmi)) {
      coefs_bmi["group2Patient", "Pr(>|t|)"]
    } else NA

    p_fit <- if (fit_var %in% rownames(coefs_bmi)) {
      coefs_bmi[fit_var, "Pr(>|t|)"]
    } else NA

    beta_group <- if ("group2Patient" %in% rownames(coefs_bmi)) {
      coefs_bmi["group2Patient", "Estimate"]
    } else NA

    beta_fit <- if (fit_var %in% rownames(coefs_bmi)) {
      coefs_bmi[fit_var, "Estimate"]
    } else NA

    r2 <- sm_bmi$r.squared

    results <- rbind(results, data.frame(
      Pathway   = path,
      Fitness   = fit_var,
      model     = "with_bmi",
      p_group   = p_group,
      p_fit     = p_fit,
      beta_group = beta_group,
      beta_fit   = beta_fit,
      R2        = r2
    ))


    # ===============================
    # 2. OHNE BMI
    # ===============================
    cat(sprintf("   OHNE BMI: %s ~ %s + group2 + age + gender\n",
                path, fit_var))

    fml_no_bmi <- as.formula(
      paste0("`", path, "` ~ ", fit_var,
             " + group2 + age + gender")
    )

    mod_no_bmi <- try(lm(fml_no_bmi, data = df), silent = TRUE)
    if (inherits(mod_no_bmi, "try-error")) next

    sm_no_bmi <- summary(mod_no_bmi)

    coefs_no <- sm_no_bmi$coefficients
    p_group_no <- if ("group2Patient" %in% rownames(coefs_no)) {
      coefs_no["group2Patient", "Pr(>|t|)"]
    } else NA

    p_fit_no <- if (fit_var %in% rownames(coefs_no)) {
      coefs_no[fit_var, "Pr(>|t|)"]
    } else NA

    beta_group_no <- if ("group2Patient" %in% rownames(coefs_no)) {
      coefs_no["group2Patient", "Estimate"]
    } else NA

    beta_fit_no <- if (fit_var %in% rownames(coefs_no)) {
      coefs_no[fit_var, "Estimate"]
    } else NA

    r2_no <- sm_no_bmi$r.squared

    results <- rbind(results, data.frame(
      Pathway   = path,
      Fitness   = fit_var,
      model     = "no_bmi",
      p_group   = p_group_no,
      p_fit     = p_fit_no,
      beta_group = beta_group_no,
      beta_fit   = beta_fit_no,
      R2        = r2_no
    ))
  }
}

# -----------------------------------------------------
# FDR-KORREKTUR (getrennt nach Modelltyp)
# -----------------------------------------------------

results <- results %>%
  group_by(model) %>%
  mutate(
    p_group_fdr = p.adjust(p_group, method = "BH"),
    p_fit_fdr   = p.adjust(p_fit,   method = "BH")
  ) %>%
  ungroup()

# Signifikante/Tendenzen extrahieren (Beispiel: p_fit_fdr < 0.05)
sig_results <- results %>%
  filter(p_fit_fdr < 0.05)

# -----------------------------------------------------
# EXPORT
# -----------------------------------------------------

base_out <- "F:/Microbiome_analysis/Combined_analysis/Output/a_priori_pathways_fitness_regressions/"

if (!dir.exists(base_out)) {
  dir.create(base_out, recursive = TRUE)
}

write_csv(results,     paste0(base_out, "A_PRIORI_pathways_vs_fitness_ALL.csv"))
write_csv(sig_results, paste0(base_out, "A_PRIORI_pathways_vs_fitness_SIGNIFICANT.csv"))

cat("\n✅ Fertig! Gesamtmodelle:", nrow(results), "\n")
cat("📊 Signifikante (p_fit_fdr < 0.05):", nrow(sig_results), "\n")
print(sig_results)

