##############################
# Korrelationen BMI: ALLE Variablen (fix!)
##############################

library(readxl)
library(corrplot)
library(psych)
library(rstatix)

# 1) Sheet 1 laden (dein Pfad)
dat <- read_excel("D:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx")

##############################

# 1) EXAKTE Namen aus Output
vars_all <- c("N_score_PANSS", "P_score_PANSS", "G_score_PANSS", "PANSS_total", 
              "PAS", "SAS", "group", "bmi", "PC1_fitness", "PC2_agility", 
              "VO_max", "Jump", "Hand_mean", "Sum_Skinfolds", "Brozek_BF_age", 
              "WHtR", "WHR")

# 2) Nur vorhandene finden
available <- intersect(vars_all, colnames(dat))
cat("✅ Alle Variablen gefunden:\n"); print(available); cat("\n")

# 3) NUMERISCH machen!
dat_num <- as.data.frame(lapply(dat[available], function(x) as.numeric(as.character(x))))
dat_num$group <- as.factor(dat_num$group)  # group als Factor

vars_num <- setdiff(colnames(dat_num), "group")
cor_matrix <- cor(dat_num[, vars_num], use = "pairwise.complete.obs")

# 4) BMI-Korrelationen (sortiert!)
bmi_cor <- cor_matrix["bmi", ]
cat("🔥 BMI-Korrelationen (stärkste zuerst):\n")
print(sort(round(bmi_cor, 3), decreasing = TRUE)); cat("\n\n")

# 5) Corrplot
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust", 
         tl.cex = 0.8, addCoef.col = "black", number.cex = 0.7,
         title = "BMI vs PC1_fitness/PC2_agility/PANSS_total")

# group noch umwandeln, weil aktuell noch numerisch und des NaNs 
dat_num$group <- dat$group
dat_num %>% t_test(bmi ~ group)
dat_num %>% t_test(PC1_fitness ~ group)

# 6) t-Tests (SZ vs CTRL)
cat("📈 Group-Unterschiede:\n")
print(dat_num %>% t_test(bmi ~ group) %>% add_significance())
print(dat_num %>% t_test(PC1_fitness ~ group) %>% add_significance())
print(dat_num %>% t_test(PC2_agility ~ group) %>% add_significance()); cat("\n")

# 7) PANSS nur in SZ
sz_data <- dat_num[dat_num$group == "SZ", ]
panss_vars <- c("PANSS_total", "N_score_PANSS", "P_score_PANSS", "G_score_PANSS")
for(v in panss_vars[panss_vars %in% colnames(sz_data)]){
  test <- cor.test(sz_data$bmi, sz_data[[v]])
  cat(sprintf("SZ: r(BMI,%s)=%.3f (p=%.3f)\n", v, test$estimate, test$p.value))
}

cat("\n🎯 BMI als Nuisance: löscht PC1_fitness + Group-Effekt?\n")
