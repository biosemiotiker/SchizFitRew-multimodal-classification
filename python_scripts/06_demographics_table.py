"""
06_demographics_table.py
========================
Creates a demographics table (HC vs. SZ vs. Total).

Outputs:
  results/demographics_table.csv
  results/demographics_table.txt
  results/demographics_statistics.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE        = Path(__file__).parent.parent
EXCEL_PATH  = Path("data_combined_analysis/gbm_abundance_global_clr.xlsx")
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PANSS_COLS = {
    "PANSS Negative":  "N_score_PANSS",
    "PANSS Positive":  "P_score_PANSS",
    "PANSS General":   "G_score_PANSS",
    "PANSS Total":     "PANSS_total",
}

CLINICAL_COLS = {
    "Age of onset":              "age_of_onset",
    "CPZ equivalent (mg/day)":   "cpzeq",
    "Cumul. lifetime dose (mg)": "cum_ltd",
    "Physical Anhedonia (PAS)":  "PAS",
    "Social Anhedonia (SAS)":    "SAS",
    "WHODAS":                    "WHODAS",
    "CDSS":                      "CDSS",
}

# Variablen für HC-vs-SZ t-Tests (beide Gruppen haben Werte)
TTEST_COLS = {
    "Physical Anhedonia (PAS)": "PAS",
    "Social Anhedonia (SAS)":   "SAS",
    "WHODAS":                   "WHODAS",
    "CDSS":                     "CDSS",
}


def mean_sd(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.1f} ± {s.std():.1f}"


def count_gender(series: pd.Series) -> str:
    m = (series == 1).sum()
    f = (series == 2).sum()
    return f"{m}m / {f}f"


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name="gbm_abundance_global_clr")
    df["group"] = df["group"].str.strip()

    if "ageofonset" in df.columns and "age_of_onset" not in df.columns:
        df = df.rename(columns={"ageofonset": "age_of_onset"})
    if "cumltd" in df.columns and "cum_ltd" not in df.columns:
        df = df.rename(columns={"cumltd": "cum_ltd"})
    if "cpzeq" not in df.columns:
        print("WARNING: 'cpzeq' not found – treated as missing")
        df["cpzeq"] = np.nan

    groups = {
        "HC":    df[df["group"] == "CTRL"],
        "SZ":    df[df["group"].isin(["SZ", "SZ out"])],
        "Total": df,
    }

    rows = []

    rows.append({"Variable": "N",
                 **{g: str(len(gdf)) for g, gdf in groups.items()}})
    rows.append({"Variable": "Age (years)",
                 **{g: mean_sd(gdf["age"]) for g, gdf in groups.items()}})
    rows.append({"Variable": "Sex (m/f)",
                 **{g: count_gender(gdf["gender"]) for g, gdf in groups.items()}})

    rows.append({"Variable": "── PANSS ──", **{g: "" for g in groups}})
    for label, col in PANSS_COLS.items():
        rows.append({"Variable": label,
                     **{g: mean_sd(gdf[col]) if col in gdf.columns else "—"
                        for g, gdf in groups.items()}})

    rows.append({"Variable": "── Clinical ──", **{g: "" for g in groups}})
    for label, col in CLINICAL_COLS.items():
        rows.append({"Variable": label,
                     **{g: mean_sd(gdf[col]) if col in gdf.columns else "—"
                        for g, gdf in groups.items()}})

    # ── Group comparisons ────────────────────────────────────────────────────
    hc = groups["HC"]
    sz = groups["SZ"]

    t_age, p_age = stats.ttest_ind(hc["age"].dropna(), sz["age"].dropna())

    ct = pd.crosstab(
        df["group"].map(lambda g: "SZ" if g in ["SZ", "SZ out"] else "HC"),
        df["gender"]
    )
    chi2_sex, p_sex, _, _ = stats.chi2_contingency(ct)

    stats_rows = [
        {
            "Variable":            "Age (years)",
            "Test":                "Independent t-test",
            "Statistic":           f"t = {t_age:.3f}",
            "df":                  len(hc["age"].dropna()) + len(sz["age"].dropna()) - 2,
            "p-value":             round(p_age, 3),
            "Significant (p<.05)": "Yes" if p_age < .05 else "No",
        },
        {
            "Variable":            "Sex (m/f)",
            "Test":                "Chi-square",
            "Statistic":           f"χ² = {chi2_sex:.3f}",
            "df":                  1,
            "p-value":             round(p_sex, 3),
            "Significant (p<.05)": "Yes" if p_sex < .05 else "No",
        },
    ]

    # t-Tests für TTEST_COLS (PAS, SAS, WHODAS, CDSS)
    ttest_results = {}
    for label, col in TTEST_COLS.items():
        if col in hc.columns and col in sz.columns:
            t, p = stats.ttest_ind(hc[col].dropna(), sz[col].dropna())
            ttest_results[label] = (t, p)
            stats_rows.append({
                "Variable":            label,
                "Test":                "Independent t-test",
                "Statistic":           f"t = {t:.3f}",
                "df":                  len(hc[col].dropna()) + len(sz[col].dropna()) - 2,
                "p-value":             round(p, 3),
                "Significant (p<.05)": "Yes" if p < .05 else "No",
            })
        else:
            print(f"WARNING: '{col}' not found in data – skipping t-test")

    stats_df = pd.DataFrame(stats_rows)

    # ── Format demographics table ────────────────────────────────────────────
    col_order = ["HC", "SZ", "Total"]
    table = pd.DataFrame(rows).set_index("Variable")[col_order]

    W = 60
    txt_lines = []
    txt_lines.append("=" * W)
    txt_lines.append("DEMOGRAPHICS TABLE — SchizFitRew")
    txt_lines.append("=" * W)
    txt_lines.append(f"{'Variable':<30}" + "".join(f"{c:>12}" for c in col_order))
    txt_lines.append("-" * W)
    for idx, row in table.iterrows():
        if idx.startswith("──"):
            txt_lines.append(f"  {idx}")
        else:
            txt_lines.append(f"  {idx:<28}" + "".join(f"{str(row[c]):>12}" for c in col_order))
    txt_lines.append("-" * W)
    txt_lines.append("Group comparisons (HC vs. SZ):")
    txt_lines.append(f"  Age:    t={t_age:.3f}, p={p_age:.3f}")
    txt_lines.append(f"  Sex:    χ²={chi2_sex:.3f}, p={p_sex:.3f}")
    for label, (t, p) in ttest_results.items():
        txt_lines.append(f"  {label:<8} t={t:.3f}, p={p:.3f}")
    txt_lines.append("=" * W)
    txt_lines.append("m=male, f=female | Mean ± SD")
    txt_lines.append("SZ = patients with schizophrenia (inpatient + outpatient)")
    txt_lines.append("HC = healthy controls")
    txt_lines.append("CPZ = chlorpromazine equivalent")
    txt_lines.append("PAS = Physical Anhedonia Scale | SAS = Social Anhedonia Scale")
    txt_lines.append("WHODAS = WHO Disability Assessment Schedule")
    txt_lines.append("CDSS = Calgary Depression Scale for Schizophrenia")

    txt = "\n".join(txt_lines)
    print(txt)

    # ── Save outputs ─────────────────────────────────────────────────────────
    (RESULTS_DIR / "demographics_table.txt").write_text(txt)
    table.to_csv(RESULTS_DIR / "demographics_table.csv")

    excel_path = RESULTS_DIR / "demographics_statistics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        table.reset_index().to_excel(writer, sheet_name="Demographics Table", index=False)
        stats_df.to_excel(writer, sheet_name="Group Comparisons", index=False)

    print(f"\nSaved:")
    print(f"  {RESULTS_DIR / 'demographics_table.csv'}")
    print(f"  {RESULTS_DIR / 'demographics_table.txt'}")
    print(f"  {excel_path}")


if __name__ == "__main__":
    main()
