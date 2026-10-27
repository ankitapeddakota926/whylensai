# 🔍 WhyLens AI: Business Root-Cause & Contribution Investigator

> **Dashboards show that a metric changed. WhyLens AI investigates WHERE the change came from.**

---

## 📌 Executive Summary

Modern enterprise BI dashboards (Tableau, PowerBI, Looker) excel at displaying that a metric shifted—e.g., *"Gross Sales fell -12.4% month-over-month."* However, business leaders and operators are inevitably left asking:
- *Which regions, product lines, or customer segments drove this drop?*
- *Did some segments grow while one critical segment cratered?*
- *How much of the total decline is attributable to each specific segment?*

**WhyLens AI** performs exact **mathematical contribution decomposition** between any two time periods (Baseline vs. Comparison). It breaks down aggregate variance into segment-level contributions that sum to the total variance, isolating primary root causes without statistical noise.

---

## 📐 Mathematical Formulation: Contribution vs. Causation

### 1. Mathematical Contribution
Let $A$ denote **Period A (Baseline)** and $B$ denote **Period B (Comparison)**.
For each business segment $k \in \{1, 2, \dots, K\}$:

$$\text{Total Baseline: } T_A = \sum_{k=1}^K V_{A,k}, \quad \text{Total Comparison: } T_B = \sum_{k=1}^K V_{B,k}$$

$$\text{Total Absolute Variance: } \Delta T = T_B - T_A$$

$$\text{Segment Absolute Variance: } \Delta V_k = V_{B,k} - V_{A,k}$$

$$\text{Segment Percentage Growth: } \%\Delta V_k = \begin{cases} \frac{V_{B,k} - V_{A,k}}{V_{A,k}} \times 100\% & \text{if } V_{A,k} \neq 0 \\ \text{NaN / Undefined} & \text{if } V_{A,k} = 0 \end{cases}$$

$$\text{Segment Contribution \%: } C_k = \begin{cases} \frac{\Delta V_k}{\Delta T} \times 100\% & \text{if } \Delta T \neq 0 \\ 0.0\% & \text{if } \Delta T = 0 \end{cases}$$

#### The Additive Identity
Because summation is linear:
$$\sum_{k=1}^K \Delta V_k = \sum_{k=1}^K (V_{B,k} - V_{A,k}) = T_B - T_A = \Delta T \implies \sum_{k=1}^K C_k = 100\%$$

*Note: Individual segment contributions can exceed 100% or be negative if positive and negative segment movements offset each other.*

### 2. Why Contribution $\neq$ Causation
- **Mathematical Contribution** proves *arithmetic responsibility*: e.g., *"West region accounted for -85.2% of the company's drop in April."*
- **Causation** explains *underlying real-world mechanisms*: e.g., *"Did West drop because of a port strike, a warehouse fire, competitor price cuts, or inventory outages?"*
- Domain knowledge and operational context remain essential to establish true causal links.

---

## 🛡️ Production Edge-Case Defenses

| Production Edge Case | Risk in Naive Implementations | WhyLens AI Defense in `analyzer.py` |
| :--- | :--- | :--- |
| **Zero Baseline Metric ($V_{A,k} = 0$)** | `ZeroDivisionError` crashes the application | Evaluated safely; outputs `NaN` / `0.0%` with indicator badge |
| **Flat Net Metric ($\Delta T = 0$)** | Division by zero when computing Contribution % | If $\Delta T == 0$, sets contribution to `0.0%` and flags offsetting shifts |
| **Missing Metric Values (NaN/Null)** | Silent data dropping or distorted averages | Configurable pipeline: fills with `0.0` or drops cleanly with audit count |
| **Malformed / Missing Dates** | Parser exception or invalid period slicing | `pd.to_datetime(..., errors='coerce')`, invalid dates pruned and logged |
| **Blank Categorical Dimensions** | Unallocated rows lost from aggregation | Filled with `"(Unassigned)"` to preserve total reconciliations |
| **Discontinued / Newly Launched Segments** | Inner-join drops segments absent in one period | Outer join with `fillna(0.0)` guarantees full segment inventory |

---

## 📂 Project Architecture

```
FSI-project/
├── analyzer.py               # Core mathematical engine & pandas processing
├── app.py                    # Streamlit interactive UI & Plotly visualizations
├── generate_test_data.py     # Synthetic retail data generator with injected anomaly
├── requirements.txt          # Production dependencies
└── README.md                 # Documentation & setup guide
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13, 3.14 supported)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `streamlit`, `pandas`, `numpy`, `plotly`)*

### 3. Generate the Benchmark Dataset
Run the data generator to create a 6-month retail dataset (16,530 rows):
```bash
python generate_test_data.py
```
This produces `synthetic_retail_data.csv`.

### 4. Launch the Streamlit App
```bash
py -3.12 -m streamlit run app.py
```
*(Or if Python 3.12 is your default: `streamlit run app.py`)*

Open your browser at `http://localhost:8501`.

---

## 🧪 Injected Root-Cause Scenario Walkthrough

The test dataset contains a simulated real-world supply chain failure:
- **Baseline Period**: Month 3 (March 2024) or Q1 (Jan–Mar 2024). Normal retail operations across all 4 regions and 5 product categories.
- **Comparison Period**: Month 4 (April 2024). An operational outage in the **West** regional warehouse severely disrupts **Electronics** inventory (sales drop ~85% for that segment).

### Step-by-Step Test:
1. Open the app and click **"⚡ Load Synthetic Retail Benchmark"**.
2. In the Sidebar:
   - Metric: `Sales`
   - Breakdown Dimension: `Region` (or `Category`)
   - Preset: **"Month 3 vs Month 4 (Root Cause Spotlight)"**
3. Click **"🚀 Run Root-Cause Investigation"**.
4. **Result**:
   - The **Automated Root-Cause Narrative** immediately identifies **West** as the primary drag, contributing the lion's share of the total decline.
   - Switch the Breakdown Dimension to `Category` to see **Electronics** surface as the dominant product category failure.
   - The **Waterfall Bridge** provides an executive-ready chart showing baseline sales bridging directly to comparison sales through each segment's contribution.

---

## 📊 Feature Matrix

- **Interactive File Ingestion**: Upload any business CSV or one-click load the benchmark dataset.
- **Data Health Profiler**: Real-time inspection of column types, missing value ratios, and cardinality.
- **Variance Bridge (Waterfall)**: Dynamic Plotly Waterfall chart grouping tail segments into an "Other" bucket for clean presentation.
- **Segment Contribution Ranking**: Horizontal bar chart color-coded by positive/negative variance.
- **Evidence Matrix**: High-density interactive data table with formatted deltas, progress bars, and one-click CSV export.
- **Executive Audit Trail**: Full accounting of cleaned rows, missing values handled, and date filters applied.