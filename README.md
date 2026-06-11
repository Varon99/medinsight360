# MedInsight360 — Clinical Analytics Portfolio Project

## Overview
MedInsight360 is an end-to-end clinical analytics project built to demonstrate a full data analyst stack — from raw data generation and storage to statistical analysis, predictive modelling, and interactive dashboarding.

The project simulates a real-world hospital analytics use case, analysing 50,000 synthetic patient admissions across 31 dimensions to uncover insights around readmissions, patient demographics, clinical outcomes, and financial performance.

---

## Dataset
- **Size:** 50,000 rows × 31 columns
- **Type:** Synthetic patient admissions data (generated using Python, Faker, NumPy, Pandas)
- **Storage:** AWS S3 (raw CSV) + AWS Athena (external table for SQL querying)
- **File:** `patient_admissions.csv`

### Key Columns
| Column | Description |
|---|---|
| patient_id | Unique patient identifier |
| age, gender, bmi | Patient demographics |
| diagnosis, department | Clinical classification |
| admission_date, discharge_date | Admission timeline |
| length_of_stay | Days admitted |
| severity_score, severity_band | Clinical severity |
| billing_amount, insurance_type | Financial data |
| readmitted_30days | 30-day readmission flag |
| satisfaction_score | Patient satisfaction (1-5) |
| lab_result_score | Lab result indicator |
| high_cost_flag | High billing amount flag |

---

## Tech Stack
| Tool | Purpose |
|---|---|
| Python (Faker, NumPy, Pandas) | Data generation & analysis |
| AWS S3 | Cloud data storage |
| AWS Athena | Serverless SQL querying |
| SQL | Data exploration & cohort analysis |
| Power BI + DAX | Interactive dashboarding |
| Scikit-learn | Predictive modelling |
| SciPy, StatsModels | Statistical analysis |
| Jupyter Notebook | Analysis environment |

---

## Modules Built

### 1. Data Generation
- Generated 50,000 synthetic patient records using Python (Faker, NumPy, Pandas)
- Designed 31 clinically relevant columns covering demographics, clinical, financial, and outcome dimensions
- Uploaded to AWS S3 and queried via AWS Athena external table

### 2. Cohort Analysis
- Segmented patients by age group, gender, diagnosis, and department
- Identified high-risk cohorts based on severity band and readmission patterns

### 3. Funnel Analysis
- Built admission → treatment → discharge → readmission funnel
- Quantified drop-off rates across the patient journey

### 4. Statistical Analysis
- **T-tests** — compared means across two groups (e.g. readmitted vs non-readmitted billing)
- **ANOVA** — compared means across multiple groups (e.g. length of stay by department)
- **Chi-square tests** — tested associations between categorical variables
- **Pearson & Spearman correlation** — analysed relationships between continuous variables
- **Confidence intervals** — quantified uncertainty around key metrics

### 5. Correlation Analysis
- Explored relationships between length of stay, billing amount, severity score, and lab results
- Identified positive correlation between length of stay and billing amount

### 6. Predictive Modelling
- Built binary classification model to predict **30-day readmission**
- Models: Logistic Regression + Random Forest
- **AUC = 0.59** (baseline model on synthetic data)
- Features: age, severity score, length of stay, num procedures, num medications, lab result score

### 7. Power BI Dashboard (5 pages)
- **Overview** — KPI cards, admissions trend, department breakdown, insurance split
- **Patient Profile** — Age group, gender, BMI band, smoking status distributions
- **Clinical Analysis** — Diagnosis breakdown, severity band, lab scores, length of stay
- **Financial Analysis** — Billing by department, insurance type, diagnosis, high cost analysis
- **Outcome Analysis** — Readmission rates by diagnosis and age group, satisfaction scores, length of stay vs billing scatter

---

## Key Insights

1. **Older patients readmit more** — 76+ age group has 32% readmission rate vs 15% for 18-30, suggesting age-targeted intervention strategies
2. **Acute MI has highest readmission rate** at 24% among all diagnoses
3. **Length of stay positively correlates with billing amount** — longer admissions drive higher costs
4. **25% of patients are high cost** — a small proportion driving disproportionate billing
5. **Corporate insurance patients have highest avg billing** at $25K vs $10K for uninsured patients
6. **Gender split is near equal** — 50.2% Female, 48.9% Male, 0.9% Other

---

## How to Run

### Prerequisites
```
pip install pandas numpy faker scikit-learn scipy statsmodels matplotlib seaborn
```

### Steps
1. Clone the repository
```
git clone https://github.com/YOUR_USERNAME/medinsight360.git
cd medinsight360
```

2. Generate the dataset
```
python src/generate_data.py
```

3. Run analysis notebooks in order
```
jupyter notebook
```
- `01_cohort_analysis.ipynb`
- `02_funnel_analysis.ipynb`
- `03_statistical_analysis.ipynb`
- `04_correlation_analysis.ipynb`
- `05_predictive_model.ipynb`

4. Open Power BI dashboard
- Open `MedInsight360.pbix` in Power BI Desktop
- Data source is `patient_admissions.csv`

---

## Project Structure
```
medinsight360/
│
├── data/
│   └── patient_admissions.csv
│
├── src/
│   └── generate_data.py
│
├── notebooks/
│   ├── 01_cohort_analysis.ipynb
│   ├── 02_funnel_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_correlation_analysis.ipynb
│   └── 05_predictive_model.ipynb
│
├── outputs/
│   └── (analysis outputs, charts)
│
├── dashboard/
│   └── MedInsight360.pbix
│
└── README.md
```

---

## Author
**Chilu**
Data Analyst | Python | SQL | Power BI | AWS
[LinkedIn](#) | [GitHub](#)

---

*This project uses entirely synthetic data generated for portfolio demonstration purposes. No real patient data was used.*
