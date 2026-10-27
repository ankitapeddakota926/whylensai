"""
app.py
======
Streamlit UI and interactive workflow for WhyLens AI: Business Root-Cause
and Contribution Investigator.

Allows business analysts, data leaders, and operations executives to pinpoint
WHERE metric changes originated between any two business periods using exact
mathematical contribution decomposition.
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analyzer import ContributionAnalyzer, prepare_waterfall_data, AnalysisResult

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WhyLens AI - Root-Cause & Contribution Investigator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Sleek, Modern Executive Palette)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric & Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta-pos {
        color: #10b981;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .narrative-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 18px 24px;
        color: #f8fafc;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .disclaimer-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 36px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #d97706;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
    }
    .badge-primary {
        background-color: #3b82f6;
        color: #ffffff;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def auto_detect_columns(df: pd.DataFrame):
    """Identifies suitable candidate columns for Date, Metric, and Dimension."""
    date_candidates = []
    numeric_candidates = []
    categorical_candidates = []

    for col in df.columns:
        # Check if date-like
        col_lower = col.lower()
        if any(term in col_lower for term in ["date", "time", "dt", "day", "month", "timestamp"]):
            date_candidates.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_candidates.append(col)

        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)

        # Check if categorical/string
        if (
            pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(df[col])
        ):
            # Avoid columns with high cardinality (like transaction IDs or timestamps)
            if df[col].nunique() < 500:
                categorical_candidates.append(col)

    # Defaults if not found
    if not date_candidates:
        date_candidates = [col for col in df.columns if "date" in col.lower()] or list(df.columns)
    if not numeric_candidates:
        numeric_candidates = list(df.select_dtypes(include=[np.number]).columns) or list(df.columns)
    if not categorical_candidates:
        categorical_candidates = [c for c in df.columns if c not in numeric_candidates and c not in date_candidates] or list(df.columns)

    return date_candidates, numeric_candidates, categorical_candidates


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_header_title, col_header_badge = st.columns([0.8, 0.2])
with col_header_title:
    st.title("🔍 WhyLens AI")
    st.caption("Business Root-Cause & Mathematical Contribution Investigator")
with col_header_badge:
    st.markdown("<br><span class='badge badge-primary'>MVP v1.0 Production</span>", unsafe_allow_html=True)

st.markdown(
    "Dashboards show **that** a metric moved. **WhyLens AI decomposes WHERE the change came from**, "
    "calculating the exact additive contribution of each business segment between baseline and comparison periods."
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Data Ingestion & State Management
# -----------------------------------------------------------------------------
SAMPLE_CSV_PATH = "synthetic_retail_data.csv"

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# Ingestion Source Selector
with st.expander("📂 1. Data Ingestion & File Upload", expanded=(st.session_state.df_raw is None)):
    ingest_col1, ingest_col2 = st.columns([0.65, 0.35])

    with ingest_col1:
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"],
            help="Upload your own business transaction or metric dataset."
        )

    with ingest_col2:
        st.markdown("**Quick Test Benchmark:**")
        sample_exists = os.path.exists(SAMPLE_CSV_PATH)
        if st.button("⚡ Load Synthetic Retail Benchmark (16.5k rows)", type="primary" if st.session_state.df_raw is None else "secondary"):
            if not sample_exists:
                # Generate if not on disk
                from generate_test_data import generate_retail_dataset
                generate_retail_dataset(output_filename=SAMPLE_CSV_PATH)
            st.session_state.df_raw = pd.read_csv(SAMPLE_CSV_PATH)
            st.success("Loaded synthetic retail benchmark with injected West-Electronics anomaly.")

    if uploaded_file is not None:
        try:
            st.session_state.df_raw = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(st.session_state.df_raw):,} records).")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

df_raw = st.session_state.df_raw

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
if df_raw is None:
    st.info("👆 Please upload a CSV dataset or click **'Load Synthetic Retail Benchmark'** above to begin investigation.")
    st.stop()

# Data Quality & Profiling Summary
with st.expander("📊 Dataset Profiling & Column Health", expanded=False):
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    prof_col1.metric("Total Rows", f"{len(df_raw):,}")
    prof_col2.metric("Total Columns", f"{len(df_raw.columns)}")
    total_nulls = df_raw.isna().sum().sum()
    prof_col3.metric("Missing Cells (Total)", f"{total_nulls:,}")

    col_meta = []
    for col in df_raw.columns:
        null_count = df_raw[col].isna().sum()
        pct_null = (null_count / len(df_raw)) * 100
        col_meta.append({
            "Column": col,
            "Inferred Type": str(df_raw[col].dtype),
            "Non-Null Count": len(df_raw) - null_count,
            "Null Count": null_count,
            "Missing %": f"{pct_null:.1f}%",
            "Unique Values": df_raw[col].nunique(dropna=True),
            "Sample Value": str(df_raw[col].dropna().iloc[0]) if not df_raw[col].dropna().empty else ""
        })
    st.dataframe(pd.DataFrame(col_meta), use_container_width=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Investigation Setup")

date_candidates, numeric_candidates, categorical_candidates = auto_detect_columns(df_raw)

# Column mapping
date_col = st.sidebar.selectbox(
    "1. Date Column",
    options=list(df_raw.columns),
    index=list(df_raw.columns).index(date_candidates[0]) if date_candidates else 0,
    help="Chronological date or timestamp column."
)

metric_col = st.sidebar.selectbox(
    "2. Metric to Decompose",
    options=numeric_candidates if numeric_candidates else list(df_raw.columns),
    index=0,
    help="Numerical metric to analyze (e.g. Sales, Revenue, Quantity)."
)

dimension_col = st.sidebar.selectbox(
    "3. Breakdown Dimension",
    options=list(df_raw.columns),
    index=list(df_raw.columns).index(categorical_candidates[0]) if categorical_candidates else 0,
    help="Categorical segment attribute to investigate (e.g., Region, Category, Product, Channel)."
)

missing_handling = st.sidebar.radio(
    "Missing Metric Values Policy",
    options=["fill_zero", "drop"],
    format_func=lambda x: "Fill Missing with 0.0 (Recommended)" if x == "fill_zero" else "Drop Missing Rows",
    help="How to handle null/NaN values found in the metric column."
)

# Parse min/max date safely for date picker range
temp_dates = pd.to_datetime(df_raw[date_col], errors="coerce").dropna()
if temp_dates.empty:
    st.sidebar.error("Selected date column does not contain valid dates.")
    st.stop()

min_date = temp_dates.min().date()
max_date = temp_dates.max().date()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Period Comparison Windows")

# Presets for ease of use
preset_choice = st.sidebar.selectbox(
    "Date Range Presets",
    options=[
        "Custom Date Selection",
        "Month 3 vs Month 4 (Root Cause Spotlight)",
        "Q1 Baseline vs Month 4",
        "Split Dataset into Equal Halves",
    ]
)

# Default boundaries
if preset_choice == "Month 3 vs Month 4 (Root Cause Spotlight)":
    # 2024-03-01 to 2024-03-31 vs 2024-04-01 to 2024-04-30
    default_a_start = max(min_date, pd.to_datetime("2024-03-01").date())
    default_a_end = min(max_date, pd.to_datetime("2024-03-31").date())
    default_b_start = max(min_date, pd.to_datetime("2024-04-01").date())
    default_b_end = min(max_date, pd.to_datetime("2024-04-30").date())
elif preset_choice == "Q1 Baseline vs Month 4":
    default_a_start = max(min_date, pd.to_datetime("2024-01-01").date())
    default_a_end = min(max_date, pd.to_datetime("2024-03-31").date())
    default_b_start = max(min_date, pd.to_datetime("2024-04-01").date())
    default_b_end = min(max_date, pd.to_datetime("2024-04-30").date())
elif preset_choice == "Split Dataset into Equal Halves":
    mid_point = min_date + (max_date - min_date) / 2
    default_a_start = min_date
    default_a_end = mid_point
    default_b_start = mid_point + pd.Timedelta(days=1)
    default_b_end = max_date
else:
    # First half vs Second half default
    days_span = (max_date - min_date).days
    quarter_span = max(1, days_span // 4)
    default_a_start = min_date
    default_a_end = min_date + pd.Timedelta(days=quarter_span * 2)
    default_b_start = default_a_end + pd.Timedelta(days=1)
    default_b_end = max_date

col_sb_a1, col_sb_a2 = st.sidebar.columns(2)
with col_sb_a1:
    period_a_start = st.date_input("Period A Start", value=default_a_start, min_value=min_date, max_value=max_date)
with col_sb_a2:
    period_a_end = st.date_input("Period A End", value=default_a_end, min_value=min_date, max_value=max_date)

col_sb_b1, col_sb_b2 = st.sidebar.columns(2)
with col_sb_b1:
    period_b_start = st.date_input("Period B Start", value=default_b_start, min_value=min_date, max_value=max_date)
with col_sb_b2:
    period_b_end = st.date_input("Period B End", value=default_b_end, min_value=min_date, max_value=max_date)

run_button = st.sidebar.button("🚀 Run Root-Cause Investigation", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execute Contribution Analysis
# -----------------------------------------------------------------------------
analyzer = ContributionAnalyzer(handle_missing_metric=missing_handling)

try:
    result = analyzer.analyze(
        df=df_raw,
        date_col=date_col,
        metric_col=metric_col,
        dimension_col=dimension_col,
        period_a_start=period_a_start,
        period_a_end=period_a_end,
        period_b_start=period_b_start,
        period_b_end=period_b_end,
    )
except Exception as e:
    st.error(f"⚠️ Analysis could not complete: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Section 1: Top-Line Executive KPIs
# -----------------------------------------------------------------------------
st.subheader("1. Top-Line Variance Summary")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Period A (Baseline)</div>
            <div class="metric-value">{result.total_period_a:,.2f}</div>
            <div style="font-size:0.8rem; color:#64748b;">
                {result.period_a_range[0].strftime('%b %d, %Y')} - {result.period_a_range[1].strftime('%b %d, %Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Period B (Comparison)</div>
            <div class="metric-value">{result.total_period_b:,.2f}</div>
            <div style="font-size:0.8rem; color:#64748b;">
                {result.period_b_range[0].strftime('%b %d, %Y')} - {result.period_b_range[1].strftime('%b %d, %Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

delta_color_class = "metric-delta-pos" if result.total_abs_change >= 0 else "metric-delta-neg"
with kpi_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Absolute Change (ΔT)</div>
            <div class="metric-value {delta_color_class}">{result.total_abs_change:+,.2f}</div>
            <div style="font-size:0.8rem; color:#64748b;">Additive Total Net Variance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pct_display = f"{result.total_pct_change:+.2f}%" if pd.notna(result.total_pct_change) else "N/A (Baseline 0)"
with kpi_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total % Variance</div>
            <div class="metric-value {delta_color_class}">{pct_display}</div>
            <div style="font-size:0.8rem; color:#64748b;">Relative Growth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Section 2: Automated Executive Narrative
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="narrative-card">
        <strong>💡 Automated Root-Cause Narrative:</strong><br>
        {result.narrative_summary}
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Section 3: Visual Root-Cause Decomposition
# -----------------------------------------------------------------------------
st.subheader("2. Visual Root-Cause Decomposition")

chart_tab1, chart_tab2, chart_tab3 = st.tabs([
    "📊 Waterfall Variance Bridge",
    "🏆 Segment Contribution Ranking",
    "⚖️ Baseline vs. Comparison Comparison",
])

# TAB 1: Waterfall Chart
with chart_tab1:
    waterfall_payload = prepare_waterfall_data(result, max_segments=8)

    fig_waterfall = go.Figure(
        go.Waterfall(
            name="Decomposition",
            orientation="v",
            measure=waterfall_payload["measure"],
            x=waterfall_payload["x"],
            textposition="outside",
            text=waterfall_payload["text"],
            y=waterfall_payload["y"],
            connector={"line": {"color": "rgba(148, 163, 184, 0.4)", "width": 1.5, "dash": "solid"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#3b82f6"}},
        )
    )

    fig_waterfall.update_layout(
        title={
            "text": f"Bridge from Baseline to Comparison ({metric_col} by {dimension_col})",
            "x": 0.02,
            "font": {"size": 16, "family": "Inter, Roboto, sans-serif"}
        },
        waterfallgap=0.25,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(showgrid=False, tickangle=-20),
        yaxis=dict(title=f"Total {metric_col}", gridcolor="rgba(148, 163, 184, 0.15)"),
        showlegend=False,
        height=480,
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)
    st.caption("The Waterfall Bridge shows how each segment pushes the baseline up or down to arrive at the comparison total.")

# TAB 2: Ranked Contribution Bar Chart
with chart_tab2:
    df_sorted = result.breakdown_df.sort_values(by="abs_change", ascending=True).copy()

    # Dynamic colors: red for negative changes, green for positive
    colors = ["#ef4444" if val < 0 else "#10b981" for val in df_sorted["abs_change"]]

    fig_bar = go.Figure(
        go.Bar(
            y=df_sorted[dimension_col],
            x=df_sorted["abs_change"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{val:+,.2f} ({contrib:.1f}%)" for val, contrib in zip(df_sorted["abs_change"], df_sorted["contribution_pct"])],
            textposition="auto",
        )
    )

    fig_bar.update_layout(
        title={
            "text": f"Segment Ranking by Absolute Change & Contribution %",
            "x": 0.02,
            "font": {"size": 16, "family": "Inter, Roboto, sans-serif"}
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(title=f"Absolute Change ({metric_col})", gridcolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(title=dimension_col, categoryorder="total ascending"),
        height=max(400, len(df_sorted) * 35),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3: Period A vs Period B Comparison
with chart_tab3:
    df_compare = result.breakdown_df.sort_values(by="period_a_value", ascending=False).copy()

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Bar(
            name="Period A (Baseline)",
            x=df_compare[dimension_col],
            y=df_compare["period_a_value"],
            marker_color="#64748b",
        )
    )
    fig_comp.add_trace(
        go.Bar(
            name="Period B (Comparison)",
            x=df_compare[dimension_col],
            y=df_compare["period_b_value"],
            marker_color="#3b82f6",
        )
    )

    fig_comp.update_layout(
        barmode="group",
        title={
            "text": f"Volume Comparison: Baseline vs. Comparison",
            "x": 0.02,
            "font": {"size": 16, "family": "Inter, Roboto, sans-serif"}
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=40, r=40),
        xaxis=dict(title=dimension_col, showgrid=False),
        yaxis=dict(title=metric_col, gridcolor="rgba(148, 163, 184, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )

    st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------------------------------------------------------
# Section 4: Detailed Evidence Table
# -----------------------------------------------------------------------------
st.subheader("3. Segment Evidence & Contribution Matrix")

table_display = result.breakdown_df[[
    dimension_col,
    "period_a_value",
    "period_b_value",
    "abs_change",
    "pct_change",
    "contribution_pct",
    "direction"
]].copy()

# Sort options
sort_col1, sort_col2 = st.columns([0.4, 0.6])
with sort_col1:
    sort_by = st.selectbox(
        "Sort Evidence Matrix By",
        options=[
            "Absolute Impact Magnitude (|ΔV|)",
            "Absolute Change (Ascending - Biggest Drops First)",
            "Absolute Change (Descending - Biggest Gains First)",
            "Contribution %",
            f"Segment Name ({dimension_col})",
        ]
    )

if sort_by == "Absolute Impact Magnitude (|ΔV|)":
    table_display = table_display.sort_values(by="abs_change", key=abs, ascending=False)
elif sort_by == "Absolute Change (Ascending - Biggest Drops First)":
    table_display = table_display.sort_values(by="abs_change", ascending=True)
elif sort_by == "Absolute Change (Descending - Biggest Gains First)":
    table_display = table_display.sort_values(by="abs_change", ascending=False)
elif sort_by == "Contribution %":
    table_display = table_display.sort_values(by="contribution_pct", ascending=False)
else:
    table_display = table_display.sort_values(by=dimension_col, ascending=True)

# Render formatted dataframe with custom column configurations
st.dataframe(
    table_display,
    column_config={
        dimension_col: st.column_config.TextColumn("Segment", help=f"Unique {dimension_col} categories"),
        "period_a_value": st.column_config.NumberColumn("Period A (Baseline)", format="%.2f"),
        "period_b_value": st.column_config.NumberColumn("Period B (Comparison)", format="%.2f"),
        "abs_change": st.column_config.NumberColumn("Absolute Change (ΔV)", format="%+.2f"),
        "pct_change": st.column_config.NumberColumn("% Change", format="%+.2f%%"),
        "contribution_pct": st.column_config.ProgressColumn(
            "Contribution %",
            help="Mathematical share of the total net shift (can exceed 100% or be negative if offsetting)",
            format="%.2f%%",
            min_value=float(min(-100.0, table_display["contribution_pct"].min())),
            max_value=float(max(100.0, table_display["contribution_pct"].max())),
        ),
        "direction": st.column_config.TextColumn("Impact Direction"),
    },
    use_container_width=True,
    hide_index=True,
)

# Export Evidence Matrix
csv_download = table_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Evidence Matrix CSV",
    data=csv_download,
    file_name=f"whylens_contribution_{dimension_col.lower()}_{metric_col.lower()}.csv",
    mime="text/csv",
)

# -----------------------------------------------------------------------------
# Section 5: Data Cleansing & Integrity Audit
# -----------------------------------------------------------------------------
with st.expander("🛠️ Data Cleansing & Audit Trail", expanded=False):
    q = result.quality_stats
    audit_c1, audit_c2, audit_c3, audit_c4 = st.columns(4)
    audit_c1.metric("Raw Rows Processed", f"{q.total_raw_rows:,}")
    audit_c2.metric("Invalid Dates Dropped", f"{q.dropped_invalid_dates:,}")
    audit_c3.metric("Missing Metrics Handled", f"{q.missing_metrics_handled:,}")
    audit_c4.metric("Unassigned Dimension Rows", f"{q.unassigned_dimensions_handled:,}")

    st.write(
        f"- **Period A Active Records**: {q.period_a_rows:,} rows\n"
        f"- **Period B Active Records**: {q.period_b_rows:,} rows\n"
        f"- **Zero-Denominator Defense**: Handled automatically. Segments with Period A = 0 output NaN/% change without zero-division exceptions."
    )

# -----------------------------------------------------------------------------
# Section 6: Mandatory Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-card">
        ⚖️ <strong>Mathematical Contribution vs. Causation Disclaimer:</strong><br>
        <em>This tool calculates mathematical contribution, not causation. Domain context is required to establish true causal links.</em>
        A segment with an 85% mathematical contribution indicates where the numbers moved, but qualitative business context (e.g., supplier outages, pricing changes, or marketing pauses) is required to establish why the root cause occurred.
    </div>
    """,
    unsafe_allow_html=True,
)
