##############################
# PCA-VORBEREITUNG: KMO Test für MSA-Werte
# Datei: gbm_abundance_global_clr.xlsx
##############################

# Pakete laden
if (!requireNamespace("psych", quietly = TRUE)) install.packages("psych")
if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")

library(readxl)
library(psych)

# 1) Daten laden
dat <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")

# 2) Variablen (MIT BMI!)
vars_all <- c("bmi", "VO_max", "Jump", "Hand_mean", 
              "Sum_Skinfolds", "Brozek_BF_age", "WHR", "WHtR")

# 3) Numerisch + komplette Fälle
X_all <- dat[, vars_all]
X_all_num <- as.data.frame(lapply(X_all, function(z) as.numeric(as.character(z))))
X_all_complete <- X_all_num[complete.cases(X_all_num), ]

cat("Anzahl Proband*innen mit kompletten Daten:", nrow(X_all_complete), "\n\n")

# 4) Korrelation + KMO
cor_mat <- cor(X_all_complete, use = "pairwise.complete.obs")
kmo_res <- KMO(cor_mat)

cat("=== OVERALL KMO ===\n")
print(kmo_res$overall)

cat("\n=== ITEM-WEISE KMO ===\n")
kmo_items <- kmo_res$MSAi
print(round(kmo_items, 3))

cat("\n=== EMPFEHLUNG: KMO >= 0.6 behalten ===\n")
keep_vars <- names(kmo_items)[kmo_items >= 0.6]
print(keep_vars)

cat("\n=== ZU SCHWACH (KMO < 0.6) ===\n")
weak_vars <- names(kmo_items)[kmo_items < 0.6]
print(weak_vars)

cat("\n✅ ENTSCHEIDUNG: Welche Variablen für PCA?\n")
cat("   -> Alle KMO >= 0.6?\n")
cat("   -> Oder trotzdem WHR mitnehmen?\n")
