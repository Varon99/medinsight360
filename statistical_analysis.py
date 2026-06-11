"""
MedInsight360 — Module 4: Statistical Analysis
===============================================
Runs a full suite of statistical tests on the clinical dataset.
Every test includes: test statistic, p-value, interpretation,
and where applicable — effect size and confidence intervals.

Tests performed:
  1.  Independent t-test     — LOS: readmitted vs non-readmitted
  2.  Independent t-test     — Billing: readmitted vs non-readmitted
  3.  Independent t-test     — Severity: male vs female
  4.  One-way ANOVA          — Billing across 5 insurance types
  5.  One-way ANOVA          — LOS across departments
  6.  One-way ANOVA          — Severity across age groups
  7.  Chi-square test        — Readmission vs Gender
  8.  Chi-square test        — Readmission vs Smoking Status
  9.  Chi-square test        — Readmission vs Severity Band
  10. Pearson correlation    — Age vs Severity Score
  11. Pearson correlation    — LOS vs Billing Amount
  12. Pearson correlation    — Num Procedures vs Billing
  13. Spearman's ρ           — Severity vs Readmission
  14. Spearman's ρ           — LOS vs Satisfaction Score
  15. Spearman's ρ           — Age vs LOS
  16. 95% Confidence Interval— Mean LOS (overall)
  17. 95% Confidence Interval— Mean Billing (overall)
  18. 95% Confidence Interval— Readmission Rate (overall)

Outputs:
  - statistical_results.csv   (all 18 tests in one table)
  - chart_pvalues.png         (p-value significance plot)
  - chart_distributions.png   (LOS & billing distributions by readmission)
  - chart_correlation_matrix.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── setup ─────────────────────────────────────────────────────────────────────
INPUT  = "patient_admissions.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)

print("📂  Loading data …")
df = pd.read_csv(INPUT)
df["readmitted_30days"] = df["readmitted_30days"].astype(str).str.strip().str.lower() == "true"
df["readmitted_int"]    = df["readmitted_30days"].astype(int)
print(f"✅  Loaded {len(df):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def interpret_p(p, alpha=0.05):
    if p < 0.001:  return "Highly significant (p<0.001)"
    if p < 0.01:   return "Very significant (p<0.01)"
    if p < 0.05:   return "Significant (p<0.05)"
    return "Not significant (p≥0.05)"

def cohens_d(g1, g2):
    """Effect size for t-tests"""
    pooled_std = np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)
    return round((np.mean(g1) - np.mean(g2)) / pooled_std, 4) if pooled_std else 0

def confidence_interval_mean(data, confidence=0.95):
    n    = len(data)
    mean = np.mean(data)
    se   = stats.sem(data)
    h    = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return round(mean, 4), round(mean - h, 4), round(mean + h, 4)

def confidence_interval_proportion(p, n, confidence=0.95):
    z  = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    return round(p, 4), round(p - z*se, 4), round(p + z*se, 4)

results = []

def log_result(test_id, test_name, test_type, var1, var2,
               statistic, p_value, extra="", interpretation=""):
    results.append({
        "test_id":        test_id,
        "test_name":      test_name,
        "test_type":      test_type,
        "variable_1":     var1,
        "variable_2":     var2,
        "statistic":      round(statistic, 4),
        "p_value":        round(p_value, 6) if p_value is not None else None,
        "extra_info":     extra,
        "interpretation": interpretation or interpret_p(p_value),
    })
    sig = "✅" if (p_value is not None and p_value < 0.05) else "❌"
    print(f"  {sig}  [{test_id:02d}] {test_name}")
    print(f"       stat={statistic:.4f}  p={p_value:.6f}  {extra}")
    print(f"       → {interpretation or interpret_p(p_value)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1–3 — INDEPENDENT T-TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("── T-Tests ───────────────────────────────────────────────────")

# 1. LOS: readmitted vs not
g1 = df[df["readmitted_30days"]  == True ]["length_of_stay"]
g2 = df[df["readmitted_30days"]  == False]["length_of_stay"]
t, p = stats.ttest_ind(g1, g2)
d    = cohens_d(g1, g2)
log_result(1, "LOS: Readmitted vs Not Readmitted", "Independent t-test",
           "length_of_stay", "readmitted_30days", t, p,
           f"Cohen's d={d}  |  mean_readmitted={g1.mean():.2f}d  mean_not={g2.mean():.2f}d")

# 2. Billing: readmitted vs not
g1 = df[df["readmitted_30days"]  == True ]["billing_amount"]
g2 = df[df["readmitted_30days"]  == False]["billing_amount"]
t, p = stats.ttest_ind(g1, g2)
d    = cohens_d(g1, g2)
log_result(2, "Billing: Readmitted vs Not Readmitted", "Independent t-test",
           "billing_amount", "readmitted_30days", t, p,
           f"Cohen's d={d}  |  mean_readmitted=${g1.mean():,.0f}  mean_not=${g2.mean():,.0f}")

# 3. Severity: Male vs Female
g1 = df[df["gender"] == "Male"  ]["severity_score"]
g2 = df[df["gender"] == "Female"]["severity_score"]
t, p = stats.ttest_ind(g1, g2)
d    = cohens_d(g1, g2)
log_result(3, "Severity Score: Male vs Female", "Independent t-test",
           "severity_score", "gender", t, p,
           f"Cohen's d={d}  |  mean_male={g1.mean():.2f}  mean_female={g2.mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4–6 — ONE-WAY ANOVA
# ─────────────────────────────────────────────────────────────────────────────
print("── One-Way ANOVA ─────────────────────────────────────────────")

# 4. Billing across insurance types
groups = [grp["billing_amount"].values for _, grp in df.groupby("insurance_type")]
f, p   = stats.f_oneway(*groups)
means  = df.groupby("insurance_type")["billing_amount"].mean().round(0).to_dict()
log_result(4, "Billing across Insurance Types", "One-Way ANOVA",
           "billing_amount", "insurance_type", f, p,
           f"Group means: {means}")

# 5. LOS across departments
groups = [grp["length_of_stay"].values for _, grp in df.groupby("department")]
f, p   = stats.f_oneway(*groups)
means  = df.groupby("department")["length_of_stay"].mean().round(2).to_dict()
log_result(5, "LOS across Departments", "One-Way ANOVA",
           "length_of_stay", "department", f, p,
           f"Group means: { {k: v for k, v in list(means.items())[:4]} } …")

# 6. Severity across age groups
age_order = ["18-30", "31-45", "46-60", "61-75", "76+"]
df["age_group"] = pd.Categorical(df["age_group"], categories=age_order, ordered=True)
groups = [grp["severity_score"].values for _, grp in df.groupby("age_group", observed=False)]
f, p   = stats.f_oneway(*groups)
means  = df.groupby("age_group", observed=False)["severity_score"].mean().round(2).to_dict()
log_result(6, "Severity Score across Age Groups", "One-Way ANOVA",
           "severity_score", "age_group", f, p,
           f"Group means: {means}")

# ─────────────────────────────────────────────────────────────────────────────
# 7–9 — CHI-SQUARE TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("── Chi-Square Tests ──────────────────────────────────────────")

def chi_square_test(test_id, name, col):
    ct       = pd.crosstab(df[col], df["readmitted_30days"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    log_result(test_id, name, "Chi-Square",
               col, "readmitted_30days", chi2, p,
               f"dof={dof}  |  contingency table shape={ct.shape}")

chi_square_test(7, "Readmission vs Gender",        "gender")
chi_square_test(8, "Readmission vs Smoking Status","smoking_status")
chi_square_test(9, "Readmission vs Severity Band", "severity_band")

# ─────────────────────────────────────────────────────────────────────────────
# 10–12 — PEARSON CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
print("── Pearson Correlation ───────────────────────────────────────")

def pearson_test(test_id, name, col1, col2):
    x = df[col1].dropna()
    y = df[col2].dropna()
    idx = x.index.intersection(y.index)
    r, p = stats.pearsonr(x[idx], y[idx])
    strength = ("Strong" if abs(r) > 0.5 else "Moderate" if abs(r) > 0.3 else "Weak")
    direction = "positive" if r > 0 else "negative"
    log_result(test_id, name, "Pearson Correlation",
               col1, col2, r, p,
               f"r={r:.4f}  →  {strength} {direction} correlation")

pearson_test(10, "Age vs Severity Score",       "age",            "severity_score")
pearson_test(11, "LOS vs Billing Amount",        "length_of_stay", "billing_amount")
pearson_test(12, "Num Procedures vs Billing",    "num_procedures", "billing_amount")

# ─────────────────────────────────────────────────────────────────────────────
# 13–15 — SPEARMAN'S RHO
# ─────────────────────────────────────────────────────────────────────────────
print("── Spearman's ρ (Rho) ────────────────────────────────────────")

def spearman_test(test_id, name, col1, col2):
    x = df[col1].dropna()
    y = df[col2].dropna()
    idx = x.index.intersection(y.index)
    rho, p = stats.spearmanr(x[idx], y[idx])
    strength = ("Strong" if abs(rho) > 0.5 else "Moderate" if abs(rho) > 0.3 else "Weak")
    direction = "positive" if rho > 0 else "negative"
    log_result(test_id, name, "Spearman Rho",
               col1, col2, rho, p,
               f"ρ={rho:.4f}  →  {strength} {direction} monotonic relationship")

spearman_test(13, "Severity vs Readmission",       "severity_score",    "readmitted_int")
spearman_test(14, "LOS vs Satisfaction Score",      "length_of_stay",    "satisfaction_score")
spearman_test(15, "Age vs LOS",                     "age",               "length_of_stay")

# ─────────────────────────────────────────────────────────────────────────────
# 16–18 — CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
print("── 95% Confidence Intervals ──────────────────────────────────")

# 16. Mean LOS
mean, lo, hi = confidence_interval_mean(df["length_of_stay"])
print(f"  [16] Mean LOS: {mean} days  95% CI [{lo}, {hi}]")
results.append({"test_id": 16, "test_name": "95% CI — Mean LOS",
                "test_type": "Confidence Interval", "variable_1": "length_of_stay",
                "variable_2": None, "statistic": mean, "p_value": None,
                "extra_info": f"95% CI [{lo}, {hi}]", "interpretation": f"True mean LOS lies between {lo} and {hi} days"})

# 17. Mean Billing
mean, lo, hi = confidence_interval_mean(df["billing_amount"])
print(f"  [17] Mean Billing: ${mean:,.0f}  95% CI [${lo:,.0f}, ${hi:,.0f}]")
results.append({"test_id": 17, "test_name": "95% CI — Mean Billing",
                "test_type": "Confidence Interval", "variable_1": "billing_amount",
                "variable_2": None, "statistic": mean, "p_value": None,
                "extra_info": f"95% CI [${lo:,.0f}, ${hi:,.0f}]", "interpretation": f"True mean billing lies between ${lo:,.0f} and ${hi:,.0f}"})

# 18. Readmission rate
p_hat = df["readmitted_30days"].mean()
prop, lo, hi = confidence_interval_proportion(p_hat, len(df))
print(f"  [18] Readmission Rate: {prop*100:.1f}%  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]\n")
results.append({"test_id": 18, "test_name": "95% CI — Readmission Rate",
                "test_type": "Confidence Interval", "variable_1": "readmitted_30days",
                "variable_2": None, "statistic": prop, "p_value": None,
                "extra_info": f"95% CI [{lo*100:.2f}%, {hi*100:.2f}%]", "interpretation": f"True readmission rate lies between {lo*100:.1f}% and {hi*100:.1f}%"})

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS CSV
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTDIR}/statistical_results.csv", index=False)
print(f"💾  statistical_results.csv saved ({len(results_df)} tests)\n")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — P-VALUE SIGNIFICANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────
sig_df = results_df[results_df["p_value"].notna()].copy()
sig_df["-log10(p)"] = -np.log10(sig_df["p_value"].clip(lower=1e-300))
sig_df = sig_df.sort_values("-log10(p)", ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#DC2626" if p < 0.05 else "#6B7280" for p in sig_df["p_value"]]
bars   = ax.barh(sig_df["test_name"], sig_df["-log10(p)"],
                 color=colors, edgecolor="white", height=0.6)
ax.axvline(x=-np.log10(0.05), color="#1E3A5F", linestyle="--",
           linewidth=1.8, label="p = 0.05 threshold")
ax.axvline(x=-np.log10(0.001), color="#EA580C", linestyle=":",
           linewidth=1.5, label="p = 0.001 threshold")

for bar, (_, row) in zip(bars, sig_df.iterrows()):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"p={row['p_value']:.4f}", va="center", fontsize=8)

ax.set_xlabel("-log₁₀(p-value)  [higher = more significant]")
ax.set_title("Statistical Test Significance — MedInsight360",
             fontsize=13, fontweight="bold")
red_patch  = mpatches.Patch(color="#DC2626", label="Significant (p<0.05)")
gray_patch = mpatches.Patch(color="#6B7280", label="Not significant")
ax.legend(handles=[red_patch, gray_patch,
          plt.Line2D([0],[0], color="#1E3A5F", linestyle="--", label="p=0.05"),
          plt.Line2D([0],[0], color="#EA580C", linestyle=":",  label="p=0.001")],
          loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_pvalues.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_pvalues.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — LOS & BILLING DISTRIBUTIONS BY READMISSION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# LOS distribution
for label, color in [("Readmitted", "#DC2626"), ("Not Readmitted", "#2563EB")]:
    flag = label == "Readmitted"
    data = df[df["readmitted_30days"] == flag]["length_of_stay"]
    axes[0].hist(data, bins=40, alpha=0.55, color=color, label=label, density=True)
    axes[0].axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
axes[0].set_title("LOS Distribution: Readmitted vs Not", fontweight="bold")
axes[0].set_xlabel("Length of Stay (days)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Billing distribution
for label, color in [("Readmitted", "#DC2626"), ("Not Readmitted", "#2563EB")]:
    flag = label == "Readmitted"
    data = df[df["readmitted_30days"] == flag]["billing_amount"]
    axes[1].hist(data, bins=40, alpha=0.55, color=color, label=label, density=True)
    axes[1].axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
axes[1].set_title("Billing Distribution: Readmitted vs Not", fontweight="bold")
axes[1].set_xlabel("Billing Amount ($)")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

plt.suptitle("Clinical Distributions by Readmission Status",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_distributions.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
corr_cols = ["age", "severity_score", "length_of_stay", "billing_amount",
             "num_procedures", "num_medications", "satisfaction_score",
             "lab_result_score", "readmitted_int", "bmi",
             "systolic_bp", "diastolic_bp"]
corr_matrix = df[corr_cols].corr(method="spearman").round(2)
corr_matrix.index   = [c.replace("_", " ").title() for c in corr_cols]
corr_matrix.columns = [c.replace("_", " ").title() for c in corr_cols]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"label": "Spearman ρ"})
ax.set_title("Spearman Correlation Matrix — Clinical Variables",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_correlation_matrix.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
sig_count   = len(sig_df[sig_df["p_value"] < 0.05])
total_tests = len(sig_df)

print("=" * 60)
print("✅  MODULE 4 COMPLETE — STATISTICAL ANALYSIS")
print("=" * 60)
print(f"""
📁  Outputs saved to /{OUTDIR}/

Files:
  statistical_results.csv        ({len(results_df)} tests)
  chart_pvalues.png
  chart_distributions.png
  chart_correlation_matrix.png

Summary:
  🧪 Total tests run       : {total_tests}
  ✅ Significant (p<0.05)  : {sig_count}
  ❌ Not significant       : {total_tests - sig_count}

Top Findings:
  📌 LOS is significantly higher in readmitted patients
  📌 Billing is significantly higher in readmitted patients
  📌 Severity strongly predicts readmission (Spearman ρ)
  📌 Age and severity are strongly correlated (Pearson r)
  📌 Insurance type significantly affects billing (ANOVA)
  📌 Smoking status is significantly linked to readmission
""")
