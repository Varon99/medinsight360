"""
MedInsight360 — Module 2: Cohort Analysis
==========================================
Segments 50,000 patients into cohorts and tracks key clinical metrics
across each cohort. Four cohort types:

  1. Monthly Admission Cohorts   — how each month's patients perform
  2. Age Group Cohorts           — behaviour across life stages
  3. Diagnosis Cohorts           — condition-level benchmarking
  4. Severity Band Cohorts       — Low / Medium / High severity comparison

Metrics per cohort:
  - Patient volume
  - Avg length of stay (LOS)
  - Avg billing amount
  - 30-day readmission rate (%)
  - Avg satisfaction score
  - Avg severity score

Outputs (saved to outputs/):
  - cohort_monthly.csv
  - cohort_age_group.csv
  - cohort_diagnosis.csv
  - cohort_severity.csv
  - chart_monthly_readmission.png
  - chart_age_group_metrics.png
  - chart_diagnosis_heatmap.png
  - chart_severity_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── setup ─────────────────────────────────────────────────────────────────────
INPUT  = "patient_admissions.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
}
sns.set_theme(style="whitegrid", font_scale=1.05)

print("📂  Loading data …")
df = pd.read_csv(INPUT, parse_dates=["admission_date", "discharge_date"])
df["readmitted_30days"] = df["readmitted_30days"].astype(str).str.strip().str.lower() == "true"
df["cohort_month"] = df["admission_date"].dt.to_period("M").astype(str)

print(f"✅  Loaded {len(df):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def cohort_metrics(group_col):
    return (
        df.groupby(group_col)
        .agg(
            patient_count    = ("patient_id",        "count"),
            avg_los          = ("length_of_stay",    "mean"),
            avg_billing      = ("billing_amount",    "mean"),
            readmission_rate = ("readmitted_30days", "mean"),
            avg_satisfaction = ("satisfaction_score","mean"),
            avg_severity     = ("severity_score",    "mean"),
        )
        .round(2)
        .reset_index()
    )

# ─────────────────────────────────────────────────────────────────────────────
# COHORT 1 — MONTHLY
# ─────────────────────────────────────────────────────────────────────────────
print("── Cohort 1: Monthly Admissions ──────────────────────────────")
monthly = cohort_metrics("cohort_month").sort_values("cohort_month")
monthly["readmission_pct"] = (monthly["readmission_rate"] * 100).round(1)
monthly.to_csv(f"{OUTDIR}/cohort_monthly.csv", index=False)
print(monthly[["cohort_month","patient_count","readmission_pct","avg_los","avg_billing"]].to_string(index=False))

# Chart 1 — Monthly readmission rate trend
fig, ax1 = plt.subplots(figsize=(14, 5))
x = range(len(monthly))
ax1.bar(x, monthly["patient_count"], color=PALETTE["blue"], alpha=0.35, label="Patient Volume")
ax1.set_ylabel("Patient Volume", color=PALETTE["blue"])
ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])

ax2 = ax1.twinx()
ax2.plot(x, monthly["readmission_pct"], color=PALETTE["red"],
         linewidth=2.5, marker="o", markersize=4, label="Readmission %")
ax2.set_ylabel("30-Day Readmission Rate (%)", color=PALETTE["red"])
ax2.tick_params(axis="y", labelcolor=PALETTE["red"])
ax2.set_ylim(0, 40)

ax1.set_xticks(list(x))
ax1.set_xticklabels(monthly["cohort_month"], rotation=45, ha="right", fontsize=8)
ax1.set_xlabel("Admission Month")
plt.title("Monthly Patient Volume vs 30-Day Readmission Rate (2022–2024)",
          fontsize=13, fontweight="bold", pad=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_monthly_readmission.png", dpi=150)
plt.close()
print("   ✅  chart_monthly_readmission.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# COHORT 2 — AGE GROUP
# ─────────────────────────────────────────────────────────────────────────────
print("── Cohort 2: Age Group ───────────────────────────────────────")
age_order = ["18-30", "31-45", "46-60", "61-75", "76+"]
df["age_group"] = pd.Categorical(df["age_group"], categories=age_order, ordered=True)
age_cohort = cohort_metrics("age_group").sort_values("age_group")
age_cohort["readmission_pct"] = (age_cohort["readmission_rate"] * 100).round(1)
age_cohort.to_csv(f"{OUTDIR}/cohort_age_group.csv", index=False)
print(age_cohort[["age_group","patient_count","readmission_pct","avg_los","avg_billing","avg_severity"]].to_string(index=False))

# Chart 2 — Age group multi-metric bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [
    ("avg_los",          "Avg Length of Stay (days)", PALETTE["blue"]),
    ("readmission_pct",  "30-Day Readmission Rate (%)", PALETTE["red"]),
    ("avg_billing",      "Avg Billing Amount ($)", PALETTE["green"]),
]
for ax, (col, title, color) in zip(axes, metrics):
    bars = ax.bar(age_cohort["age_group"].astype(str), age_cohort[col],
                  color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Age Group")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + bar.get_height()*0.01,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if "Billing" in title else f"{v:.1f}")
    )
plt.suptitle("Clinical Metrics by Patient Age Group", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_age_group_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_age_group_metrics.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# COHORT 3 — DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
print("── Cohort 3: Diagnosis ───────────────────────────────────────")
diag_cohort = cohort_metrics("diagnosis").sort_values("readmission_rate", ascending=False)
diag_cohort["readmission_pct"] = (diag_cohort["readmission_rate"] * 100).round(1)
diag_cohort.to_csv(f"{OUTDIR}/cohort_diagnosis.csv", index=False)
print(diag_cohort[["diagnosis","patient_count","readmission_pct","avg_los","avg_billing"]].to_string(index=False))

# Chart 3 — Diagnosis heatmap
heatmap_data = diag_cohort.set_index("diagnosis")[
    ["avg_los", "readmission_pct", "avg_severity", "avg_satisfaction"]
].rename(columns={
    "avg_los":          "Avg LOS (days)",
    "readmission_pct":  "Readmission %",
    "avg_severity":     "Avg Severity",
    "avg_satisfaction": "Avg Satisfaction",
})
# normalise each column 0-1 for heatmap colouring
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    heatmap_norm, annot=heatmap_data.round(1), fmt="g",
    cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
    ax=ax, cbar_kws={"label": "Normalised Score (0=best, 1=worst)"}
)
ax.set_title("Diagnosis Cohort Heatmap — Key Clinical Metrics",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_diagnosis_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_diagnosis_heatmap.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# COHORT 4 — SEVERITY BAND
# ─────────────────────────────────────────────────────────────────────────────
print("── Cohort 4: Severity Band ───────────────────────────────────")
sev_order = ["Low", "Medium", "High"]
df["severity_band"] = pd.Categorical(df["severity_band"], categories=sev_order, ordered=True)
sev_cohort = cohort_metrics("severity_band").sort_values("severity_band")
sev_cohort["readmission_pct"] = (sev_cohort["readmission_rate"] * 100).round(1)
sev_cohort.to_csv(f"{OUTDIR}/cohort_severity.csv", index=False)
print(sev_cohort.to_string(index=False))

# Chart 4 — Severity comparison (grouped bars)
metrics_sev = ["avg_los", "readmission_pct", "avg_satisfaction"]
labels_sev  = ["Avg LOS (days)", "Readmission %", "Avg Satisfaction"]
colors_sev  = [PALETTE["blue"], PALETTE["red"], PALETTE["green"]]

x     = np.arange(len(sev_cohort))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
for i, (col, label, color) in enumerate(zip(metrics_sev, labels_sev, colors_sev)):
    vals = sev_cohort[col].values
    # scale satisfaction to be comparable visually
    if col == "avg_satisfaction":
        vals_plot = vals * 4   # scale 1-5 → 4-20 for visibility
        bars = ax.bar(x + i*width, vals_plot, width, label=f"{label} (×4)", color=color, alpha=0.85)
    else:
        bars = ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, sev_cohort[col].values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(sev_cohort["severity_band"].astype(str), fontsize=12)
ax.set_xlabel("Severity Band")
ax.set_ylabel("Score / Rate")
ax.set_title("Clinical Outcomes by Severity Band", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_severity_comparison.png", dpi=150)
plt.close()
print("   ✅  chart_severity_comparison.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("✅  MODULE 2 COMPLETE — COHORT ANALYSIS")
print("=" * 60)
print(f"\n📁  Outputs saved to /{OUTDIR}/")
print("""
CSVs:
  cohort_monthly.csv
  cohort_age_group.csv
  cohort_diagnosis.csv
  cohort_severity.csv

Charts:
  chart_monthly_readmission.png
  chart_age_group_metrics.png
  chart_diagnosis_heatmap.png
  chart_severity_comparison.png

Key Insights:
""")
top_diag   = diag_cohort.iloc[0]
worst_age  = age_cohort.loc[age_cohort["readmission_pct"].idxmax()]
best_age   = age_cohort.loc[age_cohort["readmission_pct"].idxmin()]
high_sev   = sev_cohort[sev_cohort["severity_band"] == "High"].iloc[0]
low_sev    = sev_cohort[sev_cohort["severity_band"] == "Low"].iloc[0]

print(f"  🔴 Highest readmission diagnosis : {top_diag['diagnosis']} ({top_diag['readmission_pct']}%)")
print(f"  👴 Highest readmission age group : {worst_age['age_group']} ({worst_age['readmission_pct']}%)")
print(f"  👶 Lowest  readmission age group : {best_age['age_group']} ({best_age['readmission_pct']}%)")
print(f"  ⚠️  High severity avg LOS        : {high_sev['avg_los']} days vs {low_sev['avg_los']} days (Low)")
print(f"  💰 High severity avg billing     : ${high_sev['avg_billing']:,.0f} vs ${low_sev['avg_billing']:,.0f} (Low)")
