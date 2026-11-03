##############################
# PCA-VORBEREITUNG MIT KMO
# Datei: gbm_abundance_global_clr.xlsx
##############################

# Pakete laden (ggf. vorher installieren mit install.packages())
library(readxl)
# 1) Paket installieren (nur einmal nötig)
install.packages("psych")   # falls noch nicht installiert

# 2) Paket laden (bei jeder neuen R-Session nötig)
library(psych)


# 1) Excel-Datei einlesen ---------------------------------------------

# Pfad ggf. anpassen, falls Datei woanders liegt
dat <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")


# 2) Relevante Variablen auswählen -----------------------------------

vars <- c("VO_max",
          "Jump",
          "Hand_mean",
          "Sum_Skinfolds",
          "Brozek_BF_age",
          "WHR",
          "WHtR")

X <- dat[ , vars]

# 3) Alles in numerische Variablen umwandeln -------------------------

# (falls etwas als Charakter/Faktor eingelesen wurde)
X_num <- as.data.frame(
  lapply(X, function(z) as.numeric(as.character(z)))
)

# Optional: prüfen
str(X_num)

# 4) Fehlende Werte behandeln ----------------------------------------

# Da du sagst, du hast Werte für alle Probanden, sollte hier idealerweise 0 rauskommen
colSums(is.na(X_num))

# Falls doch irgendwo NA auftaucht, werden nur vollständige Fälle für KMO genutzt:
X_complete <- X_num[complete.cases(X_num), ]

# 5) Korrelation & KMO berechnen -------------------------------------

cor_mat <- cor(X_complete, use = "pairwise.complete.obs")

# KMO (overall + item-spezifisch)
kmo_res <- KMO(cor_mat)
print(kmo_res)

# Item-spezifische KMO-Werte
kmo_items <- kmo_res$MSAi
print(kmo_items)

# Variablen mit KMO >= 0.6
keep_vars <- names(kmo_items)[kmo_items >= 0.6]

cat("\nVariablen mit KMO >= 0.6:\n")
print(keep_vars)


# Ab hier PCA einmal mit und einmal ohne WHR (da WHR eine KMO-Wert unter 0.6)
##############################
# PCA (mit standardisiert Skalen)
##############################

install.packages("FactoMineR")
install.packages("factoextra")   # falls noch nicht installiert
library(factoextra)
library(readxl)
library(FactoMineR)
library(factoextra)

# 2) Variablen definieren ---------------------------------------------

vars_all  <- c("VO_max","Jump","Hand_mean",
               "Sum_Skinfolds","Brozek_BF_age",
               "WHR","WHtR")

vars_noWHR <- c("VO_max","Jump","Hand_mean",
                "Sum_Skinfolds","Brozek_BF_age",
                "WHtR")

# 3) Numerisch machen + vollständige Fälle ----------------------------

X_all <- dat[ , vars_all]
X_all_num <- as.data.frame(lapply(X_all, function(z) as.numeric(as.character(z))))
X_all_complete <- X_all_num[complete.cases(X_all_num), ]

X_noWHR <- X_all_complete[ , vars_noWHR]   # gleicher Zeilensatz, nur ohne WHR

# 4) PCA MIT WHR ------------------------------------------------------

pca_with_whr <- PCA(X_all_complete,
                    scale.unit = TRUE,   # Standardisierung
                    graph = FALSE)

cat("\n=== PCA MIT WHR ===\n")
summary(pca_with_whr)$eig     # Eigenwerte + Varianzaufklärung

library(factoextra)

# Screeplot PCA mit WHR
fviz_eig(pca_with_whr, main = "PCA mit WHR")  # zeigt Eigenwerte + % Varianz je Komponente

# (Optional) ohne ggplot, nur Tabelle:
eig_with_whr <- pca_with_whr$eig
eig_with_whr

# Ladungen (Korrelationen Variable–Komponente)
loadings_with_whr <- pca_with_whr$var$coord
loadings_with_whr
# Contributions (in %)
contrib_with_whr <- pca_with_whr$var$contrib

# Alles in ein Dataframe
loadings_table <- cbind(loadings_with_whr, contrib_with_whr)
loadings_table

# Variablen-Ladungen
p_var_with_whr <- fviz_pca_var(
  pca_with_whr,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)

p_var_with_whr + ggplot2::ggtitle("Variablenladungen (mit WHR)")

# 5) PCA OHNE WHR -----------------------------------------------------
##############################
# PCA OHNE WHR – komplett
##############################

library(FactoMineR)
library(factoextra)
library(ggplot2)

# PCA ohne WHR
pca_without_whr <- PCA(X_noWHR,
                       scale.unit = TRUE,
                       graph = FALSE)

cat("\n=== PCA OHNE WHR ===\n")

# 1) Eigenwerte / Varianzaufklärung als Tabelle ausgeben
eig_no_whr <- pca_without_whr$eig
print(eig_no_whr)

# 2) Screeplot
p_eig_no_whr <- fviz_eig(pca_without_whr)
print(p_eig_no_whr)

# 3) Variablenladungen als Plot
p_var_no_whr <- fviz_pca_var(
  pca_without_whr,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)

p_var_no_whr + ggtitle("Variablenladungen (ohne WHR)")

# 4) Ladungen + Contributions als Tabellen (falls du sie brauchst)
loadings_no_whr <- pca_without_whr$var$coord      # Ladungen
contrib_no_whr  <- pca_without_whr$var$contrib    # Beiträge in %

loadings_table_no_whr <- cbind(loadings_no_whr, contrib_no_whr)
# View(loadings_table_no_whr)  # in RStudio anschauen
# write.csv(loadings_table_no_whr, "PCA_without_WHR_loadings.csv", row.names = TRUE)


# 6) (Optional) Scores speichern --------------------------------------

scores_with_whr    <- as.data.frame(pca_with_whr$ind$coord)
scores_without_whr <- as.data.frame(pca_without_whr$ind$coord)

# Falls du sie exportieren willst:
write.csv(cbind(X_all_complete, scores_with_whr),
           "PCA_with_WHR_scores.csv", row.names = FALSE)
write.csv(cbind(X_noWHR, scores_without_whr),
           "PCA_without_WHR_scores.csv", row.names = FALSE)

