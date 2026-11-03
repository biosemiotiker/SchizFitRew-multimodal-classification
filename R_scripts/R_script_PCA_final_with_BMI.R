##############################
# PCA FINAL: MIT BMI, KMO-optimiert (+ Screeplots & Tabellen)
##############################

# Pakete (falls noch nicht geladen)
library(FactoMineR)
library(factoextra)
library(ggplot2)

dat <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")

# 1) Finale Variablen (KMO ≥ 0.6: bmi + die 6 anderen, OHNE WHR)
vars_final <- c("bmi", "VO_max", "Jump", "Hand_mean", 
                "Sum_Skinfolds", "Brozek_BF_age", "WHtR")

# 2) Numerisch machen + vollständige Fälle ----------------------------
X_final <- dat[ , vars_final]
X_final_num <- as.data.frame(lapply(X_final, function(z) as.numeric(as.character(z))))
X_final_complete <- X_final_num[complete.cases(X_final_num), ]

cat("Proband*innen:", nrow(X_final_complete), "\n\n")

# 3) PCA FINAL (BMI + 6 Variablen) -----------------------------------
pca_final <- PCA(X_final_complete,
                 scale.unit = TRUE,   # Standardisierung
                 graph = FALSE)

cat("=== PCA FINAL (BMI + 6 Var, ohne WHR) ===\n")
summary(pca_final)$eig     # Eigenwerte + Varianzaufklärung

# Screeplot
fviz_eig(pca_final, main = "PCA Final (+BMI, KMO-optimiert)")

# 4) EIGENWERTE als Tabelle
eig_final <- pca_final$eig
print(eig_final)

# 5) LADUNGEN (Korrelationen Variable–PC) -----------------------------
loadings_final <- pca_final$var$coord[, 1:2]  # PC1 & PC2
colnames(loadings_final) <- c("PC1_Ladung", "PC2_Ladung")
print(round(loadings_final, 3))

# Contributions (in %)
contrib_final <- pca_final$var$contrib[, 1:2]
colnames(contrib_final) <- c("PC1_Contrib_%", "PC2_Contrib_%")

# LADUNGEN-TABELLE
loadings_table_final <- cbind(Variable = rownames(loadings_final), 
                              round(loadings_final, 3), 
                              round(contrib_final, 1))
print("Ladungen + Contributions:")
print(loadings_table_final)

# 6) LADUNGSPLOT
p_var_final <- fviz_pca_var(pca_final,
                            col.var = "contrib",
                            gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                            repel = TRUE)  # Labels nicht überlappen

p_var_final + ggtitle("Variablenladungen (Final +BMI)")

# 7) SCORES SPEICHERN --------------------------------------
scores_final <- as.data.frame(pca_final$ind$coord[, 1:2])
colnames(scores_final) <- c("PC1_final", "PC2_final")

write.csv(cbind(X_final_complete, scores_final),
          "PCA_FINAL_noWHR.csv", row.names = FALSE)

cat("\n✅ PCA_FINAL_BMI_noWHR.csv gespeichert!\n")
cat("   -> PC1_final = global fitness (bmi/WHtR/BF)\n")
cat("   -> PC2_final = Agility (Jump/VO_max)\n")
