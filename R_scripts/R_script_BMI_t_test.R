## Pakete
library(readxl)
library(dplyr)

## 1) Daten einlesen
dat <- read_excel("F:/Microbiome_analysis/Combined_analysis/gbm_abundance_global_clr.xlsx",
                  sheet = "gbm_abundance_global_clr")

## 2) Relevante Variablen auswählen und aufräumen
dat2 <- dat %>%
  select(id, group, bmi) %>%
  mutate(
    # Group neu kodieren: SZ_out und SZ -> "SZ"; CTRL bleibt "CTRL"
    Group2 = case_when(
      group %in% c("SZ", "SZ_out") ~ "SZ",
      group == "CTRL" ~ "CTRL",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(bmi), !is.na(Group2))

## BMI als numerisch sicherstellen
dat2$bmi <- as.numeric(dat2$bmi)
dat2$Group2 <- factor(dat2$Group2, levels = c("CTRL", "SZ"))

## 3) Test: Ist der BMI zwischen SZ und CTRL signifikant unterschiedlich?

# a) Deskriptive Statistiken
dat2 %>%
  group_by(Group2) %>%
  summarise(
    n = n(),
    mean_bmi = mean(bmi, na.rm = TRUE),
    sd_bmi   = sd(bmi, na.rm = TRUE)
  )

# b) t-Test (ungepaarter, standardmäßig mit Welch-Korrektur)
t_bmi <- t.test(bmi ~ Group2, data = dat2)
t_bmi

## Optional: vorher Normalität/Varianzhomogenität prüfen
shapiro.test(dat2$bmi[dat2$Group2 == "CTRL"])
shapiro.test(dat2$bmi[dat2$Group2 == "SZ"])
bartlett.test(bmi ~ Group2, data = dat2)

## Falls nicht normalverteilt: nonparametrischer Test
wilcox.test(bmi ~ Group2, data = dat2)

## 4) „Korrelation“ zwischen BMI und Group (nominal vs metrisch)

# a) Lineares Modell: BMI ~ Group2
lm_bmi_group <- lm(bmi ~ Group2, data = dat2)
summary(lm_bmi_group)   # p-Wert für Group2 zeigt, ob BMI mit Group „assoziiert“ ist

# b) Effektstärke: Eta-Quadrat / R²
summary(lm_bmi_group)$r.squared

# c) Alternativ: logistische Regression Group2 ~ BMI (wenn du Group als Outcome sehen willst)
dat2$Group2_bin <- ifelse(dat2$Group2 == "SZ", 1, 0)
glm_group_bmi <- glm(Group2_bin ~ bmi, data = dat2, family = binomial)
summary(glm_group_bmi)
