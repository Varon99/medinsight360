"""
MedInsight360 — Module 5: Correlation & Rho Analysis
=====================================================
Deep-dive into relationships between clinical variables using:

  1. Pearson r    — linear relationships (continuous variables)
  2. Spearman ρ   — monotonic relationships (ordinal/non-normal)
  3. Point-Biserial — continuous vs binary (e.g. vs readmission)
  4. Partial Correlation — controlling for confounders
  5. Ranked Correlation Report — top strongest relationships

Variable pairs analysed:
  - Age           ↔ Severity, LOS, Billing, Readmission
  - Severity      ↔ LOS, Billing, Readmission, Satisfaction
  - LOS           ↔ Billing, Satisfaction, Readmission
  - BMI           ↔ Severity, LOS, Billing
  - Num Procedures↔ Billing, LOS, Medications
  - Systolic BP   ↔ Age, Severity, LOS
  - Lab Score     ↔ Severity, Readmission, Satisfaction

Outputs:
  - correlation_pearson.csv
  - correlation_spearman.csv
  - correlation_pointbiserial.csv
  - correlation_ranked.csv
  - chart_scatter_matrix.png
  - chart_rho_ranked.png
  - chart_partial_correlation.png
  - chart_pointbiserial.png
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

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

print("📂  Loading data …")
df = pd.read_csv(INPUT)
df["readmitted_30days"] = df["readmitted_30days"].astype(str).str.strip().str.lower() == "true"
df["readmitted_int"]    = df["readmitted_30days"].astype(int)
print(f"✅  Loaded {len(df):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS VARIABLES FOR CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
CONT_VARS = [
    "age", "severity_score", "length_of_stay", "billing_amount",
    "num_procedures", "num_medications", "bmi",
    "systolic_bp", "diastolic_bp", "lab_result_score",
    "satisfaction_score", "readmitted_int"
]

CONT_LABELS = {
    "age":               "Age",
    "severity_score":    "Severity Score",
    "length_of_stay":    "Length of Stay",
    "billing_amount":    "Billing Amount",
    "num_procedures":    "Num Procedures",
    "num_medications":   "Num Medications",
    "bmi":               "BMI",
    "systolic_bp":       "Systolic BP",
    "diastolic_bp":      "Diastolic BP",
    "lab_result_score":  "Lab Score",
    "satisfaction_score":"Satisfaction",
    "readmitted_int":    "Readmitted (0/1)",
}

df_corr = df[CONT_VARS].dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 1. PEARSON CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("── 1. Pearson Correlation Matrix ─────────────────────────────")
pearson_matrix = df_corr.corr(method="pearson").round(4)
pearson_matrix.index   = list(CONT_LABELS.values())
pearson_matrix.columns = list(CONT_LABELS.values())
pearson_matrix.to_csv(f"{OUTDIR}/correlation_pearson.csv")
print(pearson_matrix.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPEARMAN RHO MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("── 2. Spearman ρ Matrix ──────────────────────────────────────")
spearman_matrix = df_corr.corr(method="spearman").round(4)
spearman_matrix.index   = list(CONT_LABELS.values())
spearman_matrix.columns = list(CONT_LABELS.values())
spearman_matrix.to_csv(f"{OUTDIR}/correlation_spearman.csv")
print(spearman_matrix.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. POINT-BISERIAL CORRELATIONS (continuous vs readmission)
# ─────────────────────────────────────────────────────────────────────────────
print("── 3. Point-Biserial Correlations (vs Readmission) ──────────")
binary_var = "readmitted_int"
pb_results = []

for col in CONT_VARS:
    if col == binary_var:
        continue
    r, p = stats.pointbiserialr(df_corr[binary_var], df_corr[col])
    strength = "Strong" if abs(r) > 0.3 else "Moderate" if abs(r) > 0.1 else "Weak"
    direction = "positive" if r > 0 else "negative"
    pb_results.append({
        "variable":      CONT_LABELS[col],
        "r_pb":          round(r, 4),
        "p_value":       round(p, 6),
        "strength":      strength,
        "direction":     direction,
        "significant":   p < 0.05,
    })
    sig = "✅" if p < 0.05 else "❌"
    print(f"  {sig}  {CONT_LABELS[col]:20s}  r_pb={r:+.4f}  p={p:.6f}  → {strength} {direction}")

pb_df = pd.DataFrame(pb_results).sort_values("r_pb", ascending=False)
pb_df.to_csv(f"{OUTDIR}/correlation_pointbiserial.csv", index=False)
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. RANKED CORRELATION REPORT (top pairs by |rho|)
# ─────────────────────────────────────────────────────────────────────────────
print("── 4. Top Correlated Pairs (Spearman ρ) ─────────────────────")
ranked_rows = []
cols = CONT_VARS
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c1, c2 = cols[i], cols[j]
        rho, p = stats.spearmanr(df_corr[c1], df_corr[c2])
        ranked_rows.append({
            "variable_1":  CONT_LABELS[c1],
            "variable_2":  CONT_LABELS[c2],
            "spearman_rho": round(rho, 4),
            "abs_rho":      round(abs(rho), 4),
            "p_value":      round(p, 6),
            "significant":  p < 0.05,
        })

ranked_df = (pd.DataFrame(ranked_rows)
             .sort_values("abs_rho", ascending=False)
             .drop(columns="abs_rho")
             .reset_index(drop=True))
ranked_df.to_csv(f"{OUTDIR}/correlation_ranked.csv", index=False)
print(ranked_df.head(20).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PARTIAL CORRELATION (controlling for age)
# ─────────────────────────────────────────────────────────────────────────────
print("── 5. Partial Correlations (controlling for Age) ─────────────")

def partial_correlation(df, x, y, z):
    """Pearson partial correlation of x,y controlling for z"""
    r_xy  = stats.pearsonr(df[x], df[y])[0]
    r_xz  = stats.pearsonr(df[x], df[z])[0]
    r_yz  = stats.pearsonr(df[y], df[z])[0]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom == 0:
        return 0, 1
    r_partial = (r_xy - r_xz * r_yz) / denom
    # approximate p-value using t-distribution
    n = len(df)
    t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
    p = 2 * stats.t.sf(abs(t_stat), df=n - 3)
    return round(r_partial, 4), round(p, 6)

partial_pairs = [
    ("severity_score",    "length_of_stay",    "age"),
    ("severity_score",    "billing_amount",     "age"),
    ("severity_score",    "readmitted_int",     "age"),
    ("length_of_stay",    "billing_amount",     "age"),
    ("length_of_stay",    "satisfaction_score", "age"),
    ("num_procedures",    "billing_amount",     "age"),
    ("bmi",               "severity_score",     "age"),
    ("systolic_bp",       "severity_score",     "age"),
]

partial_rows = []
for x, y, z in partial_pairs:
    r, p = partial_correlation(df_corr, x, y, z)
    r_raw = stats.pearsonr(df_corr[x], df_corr[y])[0]
    sig = "✅" if p < 0.05 else "❌"
    change = round(r - r_raw, 4)
    print(f"  {sig}  {CONT_LABELS[x]:20s} ↔ {CONT_LABELS[y]:20s}  "
          f"r_partial={r:+.4f}  (raw r={r_raw:+.4f}, Δ={change:+.4f})  p={p:.6f}  controlling for {CONT_LABELS[z]}")
    partial_rows.append({
        "variable_1":       CONT_LABELS[x],
        "variable_2":       CONT_LABELS[y],
        "controlled_for":   CONT_LABELS[z],
        "r_partial":        r,
        "r_raw":            round(r_raw, 4),
        "delta":            change,
        "p_value":          p,
        "significant":      p < 0.05,
    })

partial_df = pd.DataFrame(partial_rows)
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — SCATTER MATRIX (key pairs)
# ─────────────────────────────────────────────────────────────────────────────
print("── Generating Charts ─────────────────────────────────────────")

scatter_cols = ["age", "severity_score", "length_of_stay",
                "billing_amount", "satisfaction_score"]
scatter_labels = {c: CONT_LABELS[c] for c in scatter_cols}

sample = df[scatter_cols + ["readmitted_int"]].dropna().sample(3000, random_state=42)
sample["Readmitted"] = sample["readmitted_int"].map({1: "Yes", 0: "No"})

fig, axes = plt.subplots(5, 5, figsize=(18, 16))
palette = {"Yes": RED, "No": BLUE}

for i, c1 in enumerate(scatter_cols):
    for j, c2 in enumerate(scatter_cols):
        ax = axes[i][j]
        if i == j:
            for val, color in palette.items():
                subset = sample[sample["Readmitted"] == val][c1]
                ax.hist(subset, bins=25, alpha=0.55, color=color, density=True)
            ax.set_ylabel("")
        else:
            for val, color in palette.items():
                s = sample[sample["Readmitted"] == val]
                ax.scatter(s[c2], s[c1], alpha=0.15, s=4, color=color)
            rho, _ = stats.spearmanr(sample[c1], sample[c2])
            ax.text(0.05, 0.92, f"ρ={rho:.2f}", transform=ax.transAxes,
                    fontsize=8, color=PURPLE, fontweight="bold")

        if i == 4:
            ax.set_xlabel(scatter_labels[c2], fontsize=8, rotation=15)
        if j == 0:
            ax.set_ylabel(scatter_labels[c1], fontsize=8)
        ax.tick_params(labelsize=6)

yes_patch = mpatches.Patch(color=RED,  label="Readmitted")
no_patch  = mpatches.Patch(color=BLUE, label="Not Readmitted")
fig.legend(handles=[yes_patch, no_patch], loc="upper right",
           fontsize=10, framealpha=0.9)
plt.suptitle("Scatter Matrix — Key Clinical Variables (Spearman ρ shown)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_scatter_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅  chart_scatter_matrix.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — RANKED RHO BAR CHART (top 20 pairs)
# ─────────────────────────────────────────────────────────────────────────────
top20 = ranked_df.head(20).copy()
top20["pair"] = top20["variable_1"] + "  ↔  " + top20["variable_2"]
top20["color"] = top20["spearman_rho"].apply(lambda r: RED if r > 0 else BLUE)

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(top20["pair"][::-1], top20["spearman_rho"][::-1],
               color=top20["color"][::-1].values,
               edgecolor="white", height=0.65)
ax.axvline(0, color=GRAY, linewidth=0.8)
ax.axvline( 0.3, color=GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
ax.axvline(-0.3, color=GRAY, linewidth=0.6, linestyle="--", alpha=0.5)

for bar, (_, row) in zip(bars, top20[::-1].iterrows()):
    offset = 0.005 if row["spearman_rho"] >= 0 else -0.005
    ha     = "left"  if row["spearman_rho"] >= 0 else "right"
    ax.text(bar.get_width() + offset,
            bar.get_y() + bar.get_height()/2,
            f"ρ={row['spearman_rho']:+.3f}",
            va="center", ha=ha, fontsize=8.5)

ax.set_xlabel("Spearman ρ")
ax.set_title("Top 20 Variable Pair Correlations — Spearman ρ",
             fontsize=13, fontweight="bold")
pos_patch = mpatches.Patch(color=RED,  label="Positive correlation")
neg_patch = mpatches.Patch(color=BLUE, label="Negative correlation")
ax.legend(handles=[pos_patch, neg_patch])
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_rho_ranked.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_rho_ranked.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — PARTIAL CORRELATION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
x     = np.arange(len(partial_df))
width = 0.35
pair_labels = [f"{r['variable_1']}\n↔ {r['variable_2']}" for _, r in partial_df.iterrows()]

b1 = ax.bar(x - width/2, partial_df["r_raw"],     width, label="Raw r",       color=BLUE,   alpha=0.8)
b2 = ax.bar(x + width/2, partial_df["r_partial"], width, label="Partial r\n(controlling for Age)",
            color=ORANGE, alpha=0.8)

ax.axhline(0, color=GRAY, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pair_labels, fontsize=8)
ax.set_ylabel("Correlation Coefficient (r)")
ax.set_title("Raw vs Partial Correlation (controlling for Age)",
             fontsize=13, fontweight="bold")
ax.legend()

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.005 if h >= 0 else -0.015),
            f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_partial_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_partial_correlation.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — POINT-BISERIAL (vs readmission)
# ─────────────────────────────────────────────────────────────────────────────
pb_plot = pb_df.sort_values("r_pb")
colors  = [RED if r > 0 else BLUE for r in pb_plot["r_pb"]]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(pb_plot["variable"], pb_plot["r_pb"],
               color=colors, edgecolor="white", height=0.6)
ax.axvline(0, color=GRAY, linewidth=0.8)

for bar, (_, row) in zip(bars, pb_plot.iterrows()):
    offset = 0.002 if row["r_pb"] >= 0 else -0.002
    ha     = "left" if row["r_pb"] >= 0 else "right"
    sig    = "*" if row["significant"] else ""
    ax.text(bar.get_width() + offset,
            bar.get_y() + bar.get_height()/2,
            f"r={row['r_pb']:+.3f}{sig}", va="center", ha=ha, fontsize=9)

ax.set_xlabel("Point-Biserial Correlation (r_pb)\n* = significant at p<0.05")
ax.set_title("Point-Biserial Correlations: Clinical Variables vs 30-Day Readmission",
             fontsize=13, fontweight="bold")
pos_patch = mpatches.Patch(color=RED,  label="Positively associated with readmission")
neg_patch = mpatches.Patch(color=BLUE, label="Negatively associated with readmission")
ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_pointbiserial.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_pointbiserial.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
top3   = ranked_df.head(3)
top_pb = pb_df.iloc[0]
bot_pb = pb_df.iloc[-1]

print("=" * 60)
print("✅  MODULE 5 COMPLETE — CORRELATION & RHO ANALYSIS")
print("=" * 60)
print(f"""
📁  Outputs saved to /{OUTDIR}/

CSVs:
  correlation_pearson.csv
  correlation_spearman.csv
  correlation_pointbiserial.csv
  correlation_ranked.csv

Charts:
  chart_scatter_matrix.png
  chart_rho_ranked.png
  chart_partial_correlation.png
  chart_pointbiserial.png

Top 3 Strongest Correlations (Spearman ρ):
  1. {top3.iloc[0]['variable_1']} ↔ {top3.iloc[0]['variable_2']:25s} ρ={top3.iloc[0]['spearman_rho']:+.4f}
  2. {top3.iloc[1]['variable_1']} ↔ {top3.iloc[1]['variable_2']:25s} ρ={top3.iloc[1]['spearman_rho']:+.4f}
  3. {top3.iloc[2]['variable_1']} ↔ {top3.iloc[2]['variable_2']:25s} ρ={top3.iloc[2]['spearman_rho']:+.4f}

Point-Biserial vs Readmission:
  Most positively linked : {top_pb['variable']} (r={top_pb['r_pb']:+.4f})
  Most negatively linked : {bot_pb['variable']} (r={bot_pb['r_pb']:+.4f})
""")
