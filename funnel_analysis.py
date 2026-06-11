"""
MedInsight360 — Module 3: Funnel Analysis
==========================================
Tracks the full patient journey through the clinical pipeline and
measures drop-off (or progression) at each stage.

Patient Journey Funnel:
  Stage 1 — Admitted          : all patients who entered the hospital
  Stage 2 — Assessed          : patients who received a severity assessment
  Stage 3 — Diagnosed         : patients assigned a primary diagnosis
  Stage 4 — Treated           : patients who received active treatment
                                (not just Observation)
  Stage 5 — Discharged        : patients who completed their stay (LOS >= 1)
  Stage 6 — Readmitted (30d)  : patients who came back within 30 days

Secondary Funnels (breakdowns):
  - Funnel by Department
  - Funnel by Insurance Type
  - Funnel by Severity Band

Outputs:
  - funnel_overall.csv
  - funnel_by_department.csv
  - funnel_by_insurance.csv
  - chart_funnel_overall.png
  - chart_funnel_department.png
  - chart_funnel_insurance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── setup ─────────────────────────────────────────────────────────────────────
INPUT  = "patient_admissions.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

PALETTE = {
    "stage1": "#1E3A5F",
    "stage2": "#2563EB",
    "stage3": "#3B82F6",
    "stage4": "#60A5FA",
    "stage5": "#16A34A",
    "stage6": "#DC2626",
    "gray":   "#6B7280",
}

print("📂  Loading data …")
df = pd.read_csv(INPUT, parse_dates=["admission_date", "discharge_date"])
df["readmitted_30days"] = df["readmitted_30days"].astype(str).str.strip().str.lower() == "true"
print(f"✅  Loaded {len(df):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# FUNNEL STAGE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each stage is a boolean mask — patient either qualifies or not

df["stage_admitted"]   = True                                          # everyone
df["stage_assessed"]   = df["severity_score"].notna()                  # has severity score
df["stage_diagnosed"]  = df["diagnosis"].notna()                       # has diagnosis
df["stage_treated"]    = df["treatment_type"] != "Observation"         # active treatment
df["stage_discharged"] = df["length_of_stay"] >= 1                     # completed stay
df["stage_readmitted"] = df["readmitted_30days"] == True               # came back

STAGES = [
    ("Admitted",    "stage_admitted"),
    ("Assessed",    "stage_assessed"),
    ("Diagnosed",   "stage_diagnosed"),
    ("Treated",     "stage_treated"),
    ("Discharged",  "stage_discharged"),
    ("Readmitted",  "stage_readmitted"),
]

STAGE_COLORS = list(PALETTE.values())[:6]

# ─────────────────────────────────────────────────────────────────────────────
# FUNNEL 1 — OVERALL
# ─────────────────────────────────────────────────────────────────────────────
print("── Funnel 1: Overall Patient Journey ────────────────────────")

funnel_rows = []
prev_count  = None
for label, col in STAGES:
    count      = int(df[col].sum())
    pct_total  = round(count / len(df) * 100, 1)
    pct_prev   = round(count / prev_count * 100, 1) if prev_count else 100.0
    drop_off   = prev_count - count if prev_count else 0
    funnel_rows.append({
        "stage":          label,
        "patient_count":  count,
        "pct_of_total":   pct_total,
        "pct_of_prev":    pct_prev,
        "drop_off":       drop_off,
    })
    prev_count = count

funnel_df = pd.DataFrame(funnel_rows)
funnel_df.to_csv(f"{OUTDIR}/funnel_overall.csv", index=False)
print(funnel_df.to_string(index=False))

# Chart 1 — Classic funnel (horizontal bars, centred)
fig, ax = plt.subplots(figsize=(12, 7))
max_count = funnel_df["patient_count"].max()
bar_height = 0.55

for i, row in funnel_df.iterrows():
    bar_width  = row["patient_count"] / max_count
    bar_left   = (1 - bar_width) / 2
    color      = STAGE_COLORS[i]

    ax.barh(i, bar_width, left=bar_left, height=bar_height,
            color=color, edgecolor="white", linewidth=1.2)

    # stage label (left)
    ax.text(bar_left - 0.01, i, row["stage"],
            ha="right", va="center", fontsize=11, fontweight="bold")

    # count + % inside bar
    ax.text(0.5, i,
            f"{row['patient_count']:,}  ({row['pct_of_total']}%)",
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")

    # drop-off annotation (right)
    if row["drop_off"] > 0:
        ax.text(bar_left + bar_width + 0.01, i,
                f"▼ {row['drop_off']:,} ({100 - row['pct_of_prev']:.1f}%)",
                ha="left", va="center", fontsize=9, color=PALETTE["gray"])

ax.set_xlim(0, 1)
ax.set_ylim(-0.6, len(funnel_df) - 0.4)
ax.invert_yaxis()
ax.axis("off")
plt.title("Patient Journey Funnel — MedInsight360",
          fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_funnel_overall.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_funnel_overall.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# FUNNEL 2 — BY DEPARTMENT
# ─────────────────────────────────────────────────────────────────────────────
print("── Funnel 2: Funnel by Department ───────────────────────────")

dept_funnel_rows = []
for dept, grp in df.groupby("department"):
    total     = len(grp)
    treated   = int((grp["stage_treated"]).sum())
    discharged= int((grp["stage_discharged"]).sum())
    readmitted= int((grp["stage_readmitted"]).sum())
    dept_funnel_rows.append({
        "department":       dept,
        "admitted":         total,
        "treated":          treated,
        "discharged":       discharged,
        "readmitted":       readmitted,
        "treatment_rate":   round(treated   / total * 100, 1),
        "discharge_rate":   round(discharged/ total * 100, 1),
        "readmission_rate": round(readmitted/ total * 100, 1),
    })

dept_funnel = pd.DataFrame(dept_funnel_rows).sort_values("readmission_rate", ascending=False)
dept_funnel.to_csv(f"{OUTDIR}/funnel_by_department.csv", index=False)
print(dept_funnel[["department","admitted","treatment_rate","discharge_rate","readmission_rate"]].to_string(index=False))

# Chart 2 — Department grouped bar (3 key rates)
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(dept_funnel))
w = 0.25

b1 = ax.bar(x - w,   dept_funnel["treatment_rate"],   w, label="Treatment Rate %",   color="#2563EB", alpha=0.85)
b2 = ax.bar(x,       dept_funnel["discharge_rate"],   w, label="Discharge Rate %",   color="#16A34A", alpha=0.85)
b3 = ax.bar(x + w,   dept_funnel["readmission_rate"], w, label="Readmission Rate %", color="#DC2626", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(dept_funnel["department"], rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Rate (%)")
ax.set_title("Patient Journey Rates by Department", fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(0, 110)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.0f}%",
                ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_funnel_department.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_funnel_department.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# FUNNEL 3 — BY INSURANCE TYPE
# ─────────────────────────────────────────────────────────────────────────────
print("── Funnel 3: Funnel by Insurance Type ───────────────────────")

ins_funnel_rows = []
for ins, grp in df.groupby("insurance_type"):
    total      = len(grp)
    treated    = int(grp["stage_treated"].sum())
    discharged = int(grp["stage_discharged"].sum())
    readmitted = int(grp["stage_readmitted"].sum())
    avg_bill   = round(grp["billing_amount"].mean(), 0)
    ins_funnel_rows.append({
        "insurance_type":   ins,
        "admitted":         total,
        "treated":          treated,
        "discharged":       discharged,
        "readmitted":       readmitted,
        "treatment_rate":   round(treated   / total * 100, 1),
        "discharge_rate":   round(discharged/ total * 100, 1),
        "readmission_rate": round(readmitted/ total * 100, 1),
        "avg_billing":      avg_bill,
    })

ins_funnel = pd.DataFrame(ins_funnel_rows).sort_values("readmission_rate", ascending=False)
ins_funnel.to_csv(f"{OUTDIR}/funnel_by_insurance.csv", index=False)
print(ins_funnel.to_string(index=False))

# Chart 3 — Insurance funnel heatmap
heat_data = ins_funnel.set_index("insurance_type")[
    ["treatment_rate", "discharge_rate", "readmission_rate", "avg_billing"]
].rename(columns={
    "treatment_rate":   "Treatment Rate %",
    "discharge_rate":   "Discharge Rate %",
    "readmission_rate": "Readmission Rate %",
    "avg_billing":      "Avg Billing ($)",
})
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    heat_norm, annot=heat_data.round(1), fmt="g",
    cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
    ax=ax, cbar_kws={"label": "Normalised Score"}
)
ax.set_title("Patient Journey Metrics by Insurance Type",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/chart_funnel_insurance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅  chart_funnel_insurance.png saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("✅  MODULE 3 COMPLETE — FUNNEL ANALYSIS")
print("=" * 60)

obs_pct   = round(df[df["treatment_type"] == "Observation"].shape[0] / len(df) * 100, 1)
top_dept  = dept_funnel.iloc[0]
top_ins   = ins_funnel.iloc[0]
final_row = funnel_df[funnel_df["stage"] == "Readmitted"].iloc[0]

print(f"""
📁  Outputs saved to /{OUTDIR}/

CSVs:
  funnel_overall.csv
  funnel_by_department.csv
  funnel_by_insurance.csv

Charts:
  chart_funnel_overall.png
  chart_funnel_department.png
  chart_funnel_insurance.png

Key Insights:
  🏥 Total patients admitted         : {len(df):,}
  💊 Observation-only (not treated)  : {obs_pct}% of admissions
  🔴 Overall 30-day readmission rate : {final_row['pct_of_total']}%
  🏨 Highest readmission dept        : {top_dept['department']} ({top_dept['readmission_rate']}%)
  💳 Highest readmission insurance   : {top_ins['insurance_type']} ({top_ins['readmission_rate']}%)
""")
