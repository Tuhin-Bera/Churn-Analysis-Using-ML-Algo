# Customer Lifetime Value & Churn Intelligence System
## 10-Minute Final Presentation

**Presenter:** Tuhin Bera | Data Science Intern  
**Company:** Nexvia Technologies  
**Intern ID:** NEX-INT-2026-3974  
**Project:** DS-01 | Deadline: 20 November 2026

---

## Slide 1: Problem Statement

### Why does this matter?

> Acquiring a new telecom customer costs **5-25×** more than retaining an existing one.
> Yet most companies still treat retention as an afterthought.

**The company's four questions:**

| Question | Our Answer |
|----------|-----------|
| Who will leave? | LightGBM Churn Classifier (ROC-AUC = 0.8456) |
| How long will they stay? | Cox PH Survival Model (median = 66 months) |
| How much are they worth? | Discounted CLV (avg Rs.835) |
| How can we retain them? | Uplift Modeling — target Persuadables only |

---

## Slide 2: Dataset & Data Pipeline

### IBM Telco Customer Churn Dataset

- **7,043 customers** with 20 original features
- **Churn rate: 26.5%** (imbalanced — handled via scale_pos_weight)

### Pipeline Architecture

```
CSV  -->  PostgreSQL/SQLite  -->  Feature Engineering  -->  5 ML Models
                                                                |
                                                         Streamlit App
```

### Key Data Issues Found
- TotalCharges stored as string with blank values for new customers
- 11 rows needed imputation (filled with MonthlyCharges for tenure=0 customers)

---

## Slide 3: Feature Engineering

### Why feature engineering matters more than model choice

I created 13 engineered features from the original 20 columns:

| Feature | Logic | Business Meaning |
|---------|-------|-----------------|
| `tenure_group` | 0-1yr, 1-2yr, ... | Loyalty cohort |
| `charge_ratio` | MonthlyCharges / avg_monthly | Price consistency |
| `service_count` | Count of "Yes" services | Customer stickiness |
| `contract_risk` | Month-to-month=3, One year=2, Two year=1 | Commitment level |
| `is_electronic_check` | Payment method flag | Payment risk indicator |
| `has_family` | Partner OR Dependents | Life stage indicator |

> **Key insight from EDA:** Contract type explains 40%+ of churn variance alone.

---

## Slide 4: LightGBM Churn Classifier

### Model Choice: Why LightGBM?

- Faster training than XGBoost on tabular data
- Better handling of class imbalance via `scale_pos_weight = 2.76`
- Native early stopping prevents overfitting

### Results

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.8456** |
| Accuracy | 76.4% |
| Precision | 53.9% |
| Recall | **76.2%** |
| F1 Score | 63.1% |

> I optimized for **recall** over precision — in churn, missing a churner is worse than a false alarm.

### SHAP: Top 5 Churn Drivers
1. `contract_risk` — month-to-month customers are 2.35× more likely to churn
2. `tenure` — every additional year reduces churn risk significantly
3. `MonthlyCharges` — higher bills = higher churn propensity
4. `is_electronic_check` — 1.7× hazard ratio vs other payment methods
5. `service_count` — each service reduces churn risk by ~19%

---

## Slide 5: Survival Analysis

### Kaplan-Meier Survival Probabilities

| Time | All | Month-to-month | One Year | Two Year |
|------|-----|---------------|----------|---------|
| 6m   | ~91% | ~82% | ~96% | ~99% |
| 12m  | ~80% | ~65% | ~89% | ~97% |
| 24m  | ~71% | ~53% | ~85% | ~95% |

### Cox PH Hazard Ratios (key findings)

| Feature | Hazard Ratio | p-value |
|---------|-------------|---------|
| contract_risk | **2.35** | < 0.001 |
| is_electronic_check | **1.71** | < 0.001 |
| has_internet | **1.65** | < 0.001 |
| monthly_charges | 1.005 | < 0.001 |
| service_count | **0.81** | < 0.001 |
| has_family | **0.63** | < 0.001 |

> **Key finding:** Converting a month-to-month customer to a one-year contract is the single highest-impact retention action.

---

## Slide 6: Uplift Modeling (CausalML)

### The Problem with Standard Churn Models

If you target the top 100 churners with discounts:
- 30 might stay due to the offer (**Persuadables** — good!)
- 20 would stay anyway (**Sure Things** — money wasted)
- 30 will churn regardless (**Lost Causes** — money wasted)
- 20 might be annoyed by unsolicited offers

### Our Approach: CausalML S-Learner

```
Uplift Score = P(stay | offered) - P(stay | not offered)
```

### Results

| Category | Count | Action |
|----------|-------|--------|
| **Persuadable** | **67** | **Priority — offer discount** |
| Lost Cause | 3,135 | Skip — won't respond |
| Sure Thing | 122 | Skip — will stay anyway |
| Do Not Disturb | 3,719 | Avoid — don't disturb |

> **Only 0.9% of customers** are true Persuadables — but they represent Rs.70,000+ in CLV at stake.

### Data Limitation (honesty matters)
The IBM Telco dataset has no randomized treatment/control data.
I simulated treatment assignment using a business-realistic rule and
used CausalML's S-Learner as a meta-learner for ITE estimation.
Production deployment would require an A/B experiment.

---

## Slide 7: Customer Lifetime Value

### Formula

$$CLV = MonthlyCharges \times 30\% \times \frac{1 - (1+r)^{-N}}{r}$$

Where N = predicted months from Cox model, r = 0.833%/month

### CLV by Segment

| Segment | Customers | Avg CLV | Strategy |
|---------|-----------|---------|----------|
| High-Value Loyal | 1,758 | Rs.1,440 | Loyalty rewards, upsell |
| At-Risk Churners | 1,719 | Rs.1,052 | Win-back campaigns |
| New Promising | 2,161 | Rs.420 | Strong onboarding |
| Low-Value Passive | 1,405 | Rs.449 | Upsell to increase ARPU |

> Top 10% customers contribute **~28%** of total portfolio CLV.

---

## Slide 8: ROI Simulator — Demo

### Example Campaign Calculation

**Scenario:** Target 67 Persuadable customers with Rs.200/month discount for 3 months

```
Total discount value per customer: Rs.200 × 3 = Rs.600
Campaign cost: 67 × Rs.600 = Rs.40,200

Estimated response rate: ~8% (based on uplift scores)
Expected customers retained: 67 × 8% ≈ 5 customers

Avg remaining months (Cox model): 28 months
Avg monthly margin: Rs.65 × 30% = Rs.19.5

Benefit: 5 × Rs.19.5 × 28 = Rs.2,730 ... wait, small?
```

> **Why small?** Because the telecom margin is low and our Persuadables are few.
> The ROI simulator helps find the optimal balance between discount size and response rate.

**In the app, users can:**
- Adjust churn threshold, discount amount, duration
- See sensitivity charts (discount vs ROI)
- Download the targeted customer list

---

## Slide 9: Architecture & Technology

### System Architecture

```
Data Layer          ML Layer              App Layer
-----------         ---------             -----------
IBM Telco CSV  -->  LightGBM (churn)  -->  Streamlit
     |          -->  Cox PH (survival)      - Overview
     v          -->  K-Means (segments)     - EDA
PostgreSQL/     -->  CausalML (uplift)      - Churn Predictor
SQLite          -->  CLV (DCF)              - Survival Analysis
                                           - Uplift Modeling
                                           - CLV & Segments
                                           - ROI Simulator
```

### Technology Stack

| Technology | Role |
|-----------|------|
| **PostgreSQL** (SQLite fallback) | Data storage |
| **Pandas** | Data cleaning, feature engineering |
| **LightGBM** | Churn classification |
| **SHAP** | Model explainability |
| **Lifelines** | Survival analysis |
| **CausalML** | Uplift modeling |
| **Streamlit** | Interactive dashboard |
| SQLAlchemy | Database ORM |

---

## Slide 10: Results & Lessons Learned

### What Worked Well
- LightGBM + early stopping achieved ROC-AUC 0.8456 — solid for this dataset size
- SHAP explanations made model results interpretable for non-technical stakeholders
- Survival analysis added real value over raw churn prediction alone
- The SQLite fallback made the system portable without a PostgreSQL installation

### Challenges
- CausalML requires experimental data; had to simulate treatment assignment
- SHAP computation was slow for 7,000 customers — sampled 300 for the summary
- Feature names with special characters needed encoding for database compatibility

### What I'd Do With More Time
1. Hyperparameter tuning with Optuna (full grid search)
2. Run an actual A/B test to get real uplift data
3. Add time-series churn data (usage trends over time)
4. Deploy to a cloud platform (GCP / Heroku)
5. Add email/CRM integration for automated campaign trigger

### Individual Reflection
This project taught me that **the model is only 20% of the work.**
Understanding the business problem (Step 1), cleaning data (Step 2),
and communicating results clearly (Streamlit app) are equally important.
The ROI simulator was the most valuable deliverable because it directly
translates ML output into business decisions.

---

## Deliverables Checklist

- [x] Working Streamlit demo (`streamlit run app.py`)
- [x] Complete ML workflow notebook (`notebooks/churn_intelligence_workflow.ipynb`)
- [x] Architecture documentation (this file + code comments)
- [x] Model metrics: ROC-AUC 0.8456, F1 0.6312, Accuracy 76.4%
- [x] 10-minute presentation (this document)
- [x] Individual reflection (Slide 10)
- [x] PostgreSQL integration (with SQLite fallback)
- [x] All 7 required technologies used

---

*Nexvia Technologies | Data Science Internship | October 2026*  
*Tuhin Bera | NEX-INT-2026-3974*

