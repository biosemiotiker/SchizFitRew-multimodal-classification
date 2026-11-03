# =====================================================
# Gut-Brain-Module PCA Analyse (für alle 5 PCs erweitert)
# Output: C:\\Users\\User\\Documents\\Analysis\\Microbiome_analysis\\R_ouput
# Excel: Muss im gleichen Ordner wie Script liegen!
# =====================================================

library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)

# OUTPUT-ORDNER FESTLEGEN (wird erstellt falls nicht vorhanden)
output_dir <- "C:/Users/User/Documents/Analysis/Microbiome_analysis/R_ouput/PCA_results_all_pathways"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
cat("Outputs werden gespeichert in:", output_dir, "\n")
cat("Working directory:", getwd(), "\n")

# 1. DATEN LADEN (Excel laden)
df <- read_excel("D:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")
str(df)

# IDs extrahieren
df$id <- df$Column1

# Pathway-Komponenten (Spalte 2 bis vor roi_mean ACC)
pathway_cols <- colnames(df)[2:which(colnames(df) == "roi_mean ACC")-1]
cat("Pathways:", length(pathway_cols), "\n")

# fMRT-ROIs
roi_cols <- c("roi_mean ACC", "roi_mean antIns", "roi_mean Putamen", "roi_mean Nacc", "roi_mean Ncau")
roi_cols <- roi_cols[roi_cols %in% colnames(df)]

# Kovariaten bereinigen
df$group <- ifelse(df$group %in% c("SZ", "SZ out"), "SCZ", "HC")
df$Diagnose <- as.factor(df$group)
df$Alter <- as.numeric(df$age)
df$Geschlecht <- as.factor(ifelse(df$gender == 1, "M", 
                                  ifelse(df$gender == 2, "F", NA)))
df$BMI <- as.numeric(df$bmi)

# Vollständige Cases
df_complete <- df %>% 
  select(id, Diagnose, Alter, Geschlecht, BMI, all_of(pathway_cols), all_of(roi_cols)) %>%
  filter(complete.cases(.))
cat("Vollständige Cases:", nrow(df_complete), "\n")

df_complete <- droplevels(df_complete)

# 2. PCA VORBEREITUNG: 
X_path_raw <- df_complete %>% select(all_of(pathway_cols))

# Nur numerische Spalten
X_path_num <- X_path_raw %>% select(where(is.numeric))
cat("Numerische Pathways:", ncol(X_path_num), "\n")

# 👉 KONSTANTE SPALTEN ENTFERNEN (Varianz > 0)
X_path_var <- X_path_num[, apply(X_path_num, 2, var, na.rm = TRUE) > 0]

# LOGGING: Was wurde entfernt?
dropped <- setdiff(colnames(X_path_num), colnames(X_path_var))
cat("🗑️ Konstante Pathways entfernt:", length(dropped), "\n")
if(length(dropped) > 0) {
  cat("   →", paste(head(dropped, 5), collapse = ", "), "\n")
}
cat("✅ Finale variable Pathways für PCA:", ncol(X_path_var), "\n")

# PCA 
pca_res <- prcomp(X_path_var, scale. = TRUE, center = TRUE)
print(summary(pca_res))

# Scree-Plot
png(file.path(output_dir, "pca_scree.png"), width = 800, height = 600)
plot(pca_res, type = "l", main = paste("PCA:", ncol(X_path_num), "numerische Pathways"))
abline(h = 1, col = "red", lty = 2)
dev.off()

# PCs zu df_pca (X_path_var Reihenfolge!)
df_pca <- df_complete[rownames(X_path_var), ] %>%
  mutate(PC1 = pca_res$x[,1], PC2 = pca_res$x[,2], PC3 = pca_res$x[,3],
         PC4 = pca_res$x[,4], PC5 = pca_res$x[,5])

# Top PC1-Ladungen (variable Pathways)
pathway_cols_var <- colnames(X_path_var)
loadings_pc1 <- data.frame(Pathway = pathway_cols_var, Loading = pca_res$rotation[,1]) %>%
  arrange(desc(abs(Loading))) %>% slice_head(n = 10)
print(loadings_pc1)
write.csv(loadings_pc1, file.path(output_dir, "pc1_top_loadings.csv"), row.names = FALSE)

# Vollständige Loadings aller PCs als CSV
loadings_all <- as.data.frame(pca_res$rotation[,1:5])
loadings_all$Pathway <- rownames(loadings_all)
loadings_all <- loadings_all[, c("Pathway", "PC1", "PC2", "PC3", "PC4", "PC5")]
write.csv(loadings_all, file.path(output_dir, "pca_loadings_allPCs.csv"), row.names = FALSE)
cat("Vollständige Loadings gespeichert: pca_loadings_allPCs.csv\n")[file:1]

# Top 10 Loadings pro PC (sortiert nach |Loading|)
for (pc in paste0("PC", 1:5)) {
  top_pc <- loadings_all[order(abs(loadings_all[[pc]]), decreasing = TRUE), ][1:10, ]
  write.csv(top_pc, file.path(output_dir, paste0(pc, "_top10_loadings.csv")), row.names = FALSE)
  cat("Top 10 für", pc, "gespeichert!\n")
}


# 3. LINEARE MODELLE FÜR ALLE PCs
pcs <- c("PC1", "PC2", "PC3", "PC4", "PC5")  # Alle 5 PCs
results <- list()

# KORRIGIERTE MODELLE 
for(pc in pcs){
  cat("\n=== PC:", pc, "=== \n")
  results_pc <- list()
  
  for(roi_name in roi_cols){
    cat("\n  ROI:", roi_name, "\n")
    
    # Harte Formel mit Backticks (Leerzeichen-Problem gelöst)
    form_str <- paste0("`", roi_name, "` ~ ", pc, " + Diagnose + ", pc, ":Diagnose + Alter + Geschlecht + BMI")
    model_main <- lm(as.formula(form_str), data = df_pca)
    print(summary(model_main))
    
    # p_inter extrahieren
    coef_names <- rownames(summary(model_main)$coefficients)
    p_inter_name <- paste0(pc, ":DiagnoseSCZ")
    p_inter <- if(p_inter_name %in% coef_names) {
      summary(model_main)$coefficients[p_inter_name, "Pr(>|t|)"]
    } else {
      1  # Keine Interaktion
    }
    cat("  p(", pc, ":Diagnose) =", round(p_inter, 4), "\n")
    
    # Profs Logik: Getrennte Modelle wenn p<0.10
    if(p_inter < 0.10){
      model_scz <- lm(as.formula(paste0("`", roi_name, "` ~ ", pc, " + Alter + Geschlecht + BMI")), 
                      data = filter(df_pca, Diagnose == "SCZ"))
      model_hc <- lm(as.formula(paste0("`", roi_name, "` ~ ", pc, " + Alter + Geschlecht + BMI")), 
                     data = filter(df_pca, Diagnose == "HC"))
      results_pc[[roi_name]] <- list(main = model_main, scz = model_scz, hc = model_hc, p_inter = p_inter)
      cat("  → Getrennte Modelle (SCZ/HC)\n")
    } else {
      results_pc[[roi_name]] <- list(main = model_main, p_inter = p_inter)
      cat("  → Gepooltes Modell\n")
    }
    
    print(Anova(model_main, type = "II"))
  }
  results[[pc]] <- results_pc
}

cat("\n✅ Alle Modelle fertig!\n")

# Korrelations-Matrix
cor_matrix <- cor(df_pca %>% select(PC1:PC5, all_of(roi_cols)), use = "complete.obs")
png(file.path(output_dir, "pc_roi_correlation.png"), width = 800, height = 600)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, 
         title = "PC-ROI Korrelationen")
dev.off()

# 4. ALLES SPEICHERN
write.csv(df_pca, file.path(output_dir, "df_pca_with_rois.csv"), row.names = FALSE)
save(results, file = file.path(output_dir, "pca_model_results.RData"))

# Erweiterte Summary Tables für alle PCs
summary_tables <- list()
for(pc in pcs){
  # Schrittweise erstellen (vermeidet Syntax-Fehler)
  summary_table <- data.frame(
    ROI = roi_cols,
    N = sapply(roi_cols, function(r) sum(complete.cases(df_pca[[r]])))
  )
  
  summary_table[[paste0(pc, "_beta")]] <- sapply(results[[pc]], function(x) coef(x$main)[pc])
  summary_table[[paste0(pc, "_p")]] <- sapply(results[[pc]], function(x) {
    summary(x$main)$coefficients[pc, "Pr(>|t|)"]
  })
  summary_table[[paste0("Inter_", pc, "_p")]] <- sapply(results[[pc]], function(x) x$p_inter)
  
  summary_tables[[pc]] <- summary_table
  write.csv(summary_table, file.path(output_dir, paste0("pca_summary_table_", pc, ".csv")), row.names = FALSE)
}

# Gesamte Summary Table (wide format, N nur einmal) - ROBUSTE VERSION
all_cols <- c("ROI", "N")
col_names <- c("ROI", "N")

for(pc in pcs){
  all_cols <- c(all_cols, paste0(pc, "_beta"), paste0(pc, "_p"), paste0("Inter_", pc, "_p"))
  col_names <- c(col_names, paste0(pc, "_beta"), paste0(pc, "_p"), paste0("Inter_", pc, "_p"))
}

# Sicher cbind mit exakter Spaltenanzahl
all_data <- matrix(nrow = length(roi_cols), ncol = length(all_cols))
colnames(all_data) <- col_names

# ROI und N füllen
all_data[, "ROI"] <- roi_cols
all_data[, "N"] <- sapply(roi_cols, function(r) sum(complete.cases(df_pca[[r]])))

# PCs füllen
pc_idx <- 3  # Start nach ROI/N
for(pc in pcs){
  all_data[, pc_idx] <- sapply(results[[pc]], function(x) coef(x$main)[pc])
  all_data[, pc_idx+1] <- sapply(results[[pc]], function(x) summary(x$main)$coefficients[pc, "Pr(>|t|)"])
  all_data[, pc_idx+2] <- sapply(results[[pc]], function(x) x$p_inter)
  pc_idx <- pc_idx + 3
}

all_summary <- as.data.frame(all_data)
write.csv(all_summary, file.path(output_dir, "pca_summary_table_allPCs.csv"), row.names = FALSE)


cat("\n✅ FERTIG! Alle Dateien in:", output_dir, "\n")
cat("📄 Dateien: df_pca_with_rois.csv | pc1_top_loadings.csv | pca_summary_table_PC*.csv | pca_summary_table_allPCs.csv\n")
cat("💾 Modelle: pca_model_results.RData | pca_scree.png | pc_roi_correlation.png\n")
