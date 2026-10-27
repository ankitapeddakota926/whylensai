"""
analyzer.py
===========
Core mathematical engine and data processing logic for WhyLens AI.
Explicitly computes mathematical contribution and root-cause decomposition between
two distinct time periods (Baseline vs. Comparison).

Key Mathematical Formulation:
- Total Absolute Change: ΔT = Total_B - Total_A
- Segment Absolute Change: ΔV_k = V_B,k - V_A,k
- Segment Percentage Change: %ΔV_k = (V_B,k - V_A,k) / V_A,k * 100
- Segment Contribution %: C_k = (ΔV_k / ΔT) * 100
  Notice that: Σ C_k = 100% (when ΔT != 0)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class QualityMetrics:
    """Summary of data cleansing and edge-case handling operations."""
    total_raw_rows: int
    dropped_invalid_dates: int
    missing_metrics_handled: int
    unassigned_dimensions_handled: int
    period_a_rows: int
    period_b_rows: int


@dataclass
class AnalysisResult:
    """Encapsulates the complete contribution decomposition and metrics."""
    date_col: str
    metric_col: str
    dimension_col: str
    period_a_range: Tuple[pd.Timestamp, pd.Timestamp]
    period_b_range: Tuple[pd.Timestamp, pd.Timestamp]
    
    total_period_a: float
    total_period_b: float
    total_abs_change: float
    total_pct_change: Optional[float]
    
    # Detailed segment breakdown DataFrame
    breakdown_df: pd.DataFrame
    
    # Cleansing & Quality statistics
    quality_stats: QualityMetrics
    
    # Top drivers
    top_negative_drivers: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    top_positive_drivers: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    
    # Executive narrative summary
    narrative_summary: str = ""


class ContributionAnalyzer:
    """
    Production-grade mathematical contribution analyzer for business metrics.
    """

    def __init__(
        self,
        handle_missing_metric: str = "fill_zero",
        unassigned_label: str = "(Unassigned)"
    ):
        """
        Parameters
        ----------
        handle_missing_metric : str
            Strategy for missing or NaN values in the metric column:
            'fill_zero': replace NaN with 0.0 (standard for retail transactions).
            'drop': drop rows where metric is NaN.
        unassigned_label : str
            Label assigned to blank or NaN categorical dimension values.
        """
        if handle_missing_metric not in ("fill_zero", "drop"):
            raise ValueError("handle_missing_metric must be either 'fill_zero' or 'drop'")
        self.handle_missing_metric = handle_missing_metric
        self.unassigned_label = unassigned_label

    def clean_and_validate(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        dimension_col: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Cleans data types, parses dates, handles missing values, and checks integrity.
        """
        if df.empty:
            raise ValueError("The provided dataset is empty.")

        for col in (date_col, metric_col, dimension_col):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is not present in the dataset.")

        df_clean = df.copy()
        raw_count = len(df_clean)

        # 1. Date parsing with graceful error coercion
        # Store internal parsed datetime to avoid modifying original date_col if date_col == dimension_col
        df_clean["_whylens_parsed_date_"] = pd.to_datetime(df_clean[date_col], errors="coerce")
        invalid_dates_mask = df_clean["_whylens_parsed_date_"].isna()
        dropped_invalid_dates = int(invalid_dates_mask.sum())
        if dropped_invalid_dates > 0:
            df_clean = df_clean.loc[~invalid_dates_mask].copy()

        # 2. Metric coercion to numeric
        df_clean[metric_col] = pd.to_numeric(df_clean[metric_col], errors="coerce")
        missing_metric_mask = df_clean[metric_col].isna()
        missing_metrics_count = int(missing_metric_mask.sum())

        if self.handle_missing_metric == "fill_zero":
            df_clean[metric_col] = df_clean[metric_col].fillna(0.0)
        else:
            df_clean = df_clean.loc[~missing_metric_mask].copy()

        # 3. Categorical dimension normalization
        # Cast to string first so assigning unassigned_label never fails on typed columns (e.g. datetime64[us], category, int)
        df_clean[dimension_col] = df_clean[dimension_col].astype(str).str.strip()
        unassigned_mask = df_clean[dimension_col].isin(["", "nan", "None", "<NA>", "null", "NaT"]) | df_clean[dimension_col].isna()
        unassigned_count = int(unassigned_mask.sum())
        df_clean.loc[unassigned_mask, dimension_col] = self.unassigned_label

        stats = {
            "total_raw_rows": raw_count,
            "dropped_invalid_dates": dropped_invalid_dates,
            "missing_metrics_handled": missing_metrics_count,
            "unassigned_dimensions_handled": unassigned_count,
        }

        return df_clean, stats

    def analyze(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        dimension_col: str,
        period_a_start: Union[str, pd.Timestamp, date, datetime, Any],
        period_a_end: Union[str, pd.Timestamp, date, datetime, Any],
        period_b_start: Union[str, pd.Timestamp, date, datetime, Any],
        period_b_end: Union[str, pd.Timestamp, date, datetime, Any],
    ) -> AnalysisResult:
        """
        Executes period-over-period contribution analysis.
        """
        # Step 1: Clean and validate inputs
        df_clean, stats = self.clean_and_validate(df, date_col, metric_col, dimension_col)

        # Normalize dates
        ts_a_start = pd.to_datetime(period_a_start)
        ts_a_end = pd.to_datetime(period_a_end)
        ts_b_start = pd.to_datetime(period_b_start)
        ts_b_end = pd.to_datetime(period_b_end)

        if ts_a_start > ts_a_end:
            raise ValueError(f"Period A start date ({ts_a_start.strftime('%Y-%m-%d')}) cannot be after end date ({ts_a_end.strftime('%Y-%m-%d')}).")
        if ts_b_start > ts_b_end:
            raise ValueError(f"Period B start date ({ts_b_start.strftime('%Y-%m-%d')}) cannot be after end date ({ts_b_end.strftime('%Y-%m-%d')}).")

        # Step 2: Slice periods (inclusive of start and end)
        # End dates include the full day up to 23:59:59.999999
        ts_a_end_inclusive = ts_a_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        ts_b_end_inclusive = ts_b_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        mask_a = (df_clean["_whylens_parsed_date_"] >= ts_a_start) & (df_clean["_whylens_parsed_date_"] <= ts_a_end_inclusive)
        mask_b = (df_clean["_whylens_parsed_date_"] >= ts_b_start) & (df_clean["_whylens_parsed_date_"] <= ts_b_end_inclusive)

        df_a = df_clean.loc[mask_a]
        df_b = df_clean.loc[mask_b]

        period_a_rows = len(df_a)
        period_b_rows = len(df_b)

        if period_a_rows == 0:
            raise ValueError(
                f"No records found in Period A ({ts_a_start.strftime('%Y-%m-%d')} to {ts_a_end.strftime('%Y-%m-%d')}). "
                "Please adjust the Baseline date range."
            )
        if period_b_rows == 0:
            raise ValueError(
                f"No records found in Period B ({ts_b_start.strftime('%Y-%m-%d')} to {ts_b_end.strftime('%Y-%m-%d')}). "
                "Please adjust the Comparison date range."
            )

        quality_metrics = QualityMetrics(
            total_raw_rows=stats["total_raw_rows"],
            dropped_invalid_dates=stats["dropped_invalid_dates"],
            missing_metrics_handled=stats["missing_metrics_handled"],
            unassigned_dimensions_handled=stats["unassigned_dimensions_handled"],
            period_a_rows=period_a_rows,
            period_b_rows=period_b_rows,
        )

        # Step 3: Aggregate metrics by breakdown dimension
        agg_a = df_a.groupby(dimension_col, observed=False)[metric_col].sum().rename("period_a_value")
        agg_b = df_b.groupby(dimension_col, observed=False)[metric_col].sum().rename("period_b_value")

        # Outer join preserves segments that existed only in Period A or only in Period B
        merged = pd.concat([agg_a, agg_b], axis=1, join="outer").fillna(0.0)
        merged.index.name = dimension_col
        merged = merged.reset_index()

        # Step 4: Overall Totals & Changes
        total_a = float(merged["period_a_value"].sum())
        total_b = float(merged["period_b_value"].sum())
        total_abs_change = total_b - total_a

        # Edge case: Period A total is zero
        if total_a != 0.0:
            total_pct_change = (total_abs_change / abs(total_a)) * 100.0
        else:
            total_pct_change = np.nan if total_abs_change != 0.0 else 0.0

        # Step 5: Segment-level Math
        merged["abs_change"] = merged["period_b_value"] - merged["period_a_value"]

        # Safe Percentage Change:
        # If Period A == 0: output NaN (or 0.0 if Period B is also 0)
        def calc_segment_pct(row):
            val_a = row["period_a_value"]
            val_b = row["period_b_value"]
            if val_a != 0.0:
                return ((val_b - val_a) / abs(val_a)) * 100.0
            else:
                return 0.0 if val_b == 0.0 else np.nan

        merged["pct_change"] = merged.apply(calc_segment_pct, axis=1)

        # Segment Contribution %:
        # C_k = (ΔV_k / ΔT) * 100
        # Edge case: If total_abs_change is 0, contribution is 0.0 to prevent ZeroDivisionError
        if total_abs_change != 0.0:
            merged["contribution_pct"] = (merged["abs_change"] / total_abs_change) * 100.0
        else:
            merged["contribution_pct"] = 0.0

        # Absolute Impact magnitude for ranking
        merged["abs_impact"] = merged["abs_change"].abs()

        # Segment Direction classification
        conditions = [
            merged["abs_change"] > 1e-6,
            merged["abs_change"] < -1e-6
        ]
        choices = ["Positive Impact", "Negative Impact"]
        merged["direction"] = np.select(conditions, choices, default="Neutral")

        # Sort primarily by absolute impact magnitude descending
        merged = merged.sort_values(by="abs_impact", ascending=False).reset_index(drop=True)

        # Step 6: Identify Top Positive and Negative Drivers
        pos_mask = merged["abs_change"] > 0
        neg_mask = merged["abs_change"] < 0

        top_pos = merged[pos_mask].sort_values(by="abs_change", ascending=False).copy()
        top_neg = merged[neg_mask].sort_values(by="abs_change", ascending=True).copy()

        # Step 7: Build Executive Narrative Summary
        narrative = self._generate_narrative(
            metric_col=metric_col,
            dimension_col=dimension_col,
            total_a=total_a,
            total_b=total_b,
            total_abs_change=total_abs_change,
            total_pct_change=total_pct_change,
            top_neg=top_neg,
            top_pos=top_pos,
            total_segments=len(merged)
        )

        return AnalysisResult(
            date_col=date_col,
            metric_col=metric_col,
            dimension_col=dimension_col,
            period_a_range=(ts_a_start, ts_a_end),
            period_b_range=(ts_b_start, ts_b_end),
            total_period_a=total_a,
            total_period_b=total_b,
            total_abs_change=total_abs_change,
            total_pct_change=total_pct_change,
            breakdown_df=merged,
            quality_stats=quality_metrics,
            top_negative_drivers=top_neg,
            top_positive_drivers=top_pos,
            narrative_summary=narrative,
        )

    @staticmethod
    def _generate_narrative(
        metric_col: str,
        dimension_col: str,
        total_a: float,
        total_b: float,
        total_abs_change: float,
        total_pct_change: Optional[float],
        top_neg: pd.DataFrame,
        top_pos: pd.DataFrame,
        total_segments: int
    ) -> str:
        """
        Synthesizes mathematical contribution results into an automated executive narrative.
        """
        pct_str = f"{total_pct_change:+.2f}%" if pd.notna(total_pct_change) else "N/A"
        dir_word = "increased" if total_abs_change > 0 else ("decreased" if total_abs_change < 0 else "remained flat")
        
        parts = []
        parts.append(
            f"Overall **{metric_col}** {dir_word} by **{total_abs_change:+,.2f}** ({pct_str}) "
            f"from Baseline (**{total_a:,.2f}**) to Comparison (**{total_b:,.2f}**)."
        )

        if abs(total_abs_change) < 1e-6:
            parts.append(
                "Because overall net change is effectively zero, segment-level swings fully offset each other."
            )
            return " ".join(parts)

        # Highlight negative drivers
        if not top_neg.empty:
            primary_neg = top_neg.iloc[0]
            dim_val = primary_neg[dimension_col]
            neg_change = primary_neg["abs_change"]
            neg_contrib = primary_neg["contribution_pct"]
            parts.append(
                f"The primary negative drag was **{dim_val}**, which dropped by **{neg_change:,.2f}**, "
                f"accounting for **{neg_contrib:.1f}%** of the total net shift."
            )

        # Highlight positive drivers
        if not top_pos.empty:
            primary_pos = top_pos.iloc[0]
            dim_val = primary_pos[dimension_col]
            pos_change = primary_pos["abs_change"]
            pos_contrib = primary_pos["contribution_pct"]
            parts.append(
                f"On the upside, **{dim_val}** provided the strongest positive growth with **{pos_change:+,.2f}** "
                f"({pos_contrib:.1f}% mathematical contribution)."
            )

        parts.append(
            f"A total of **{total_segments}** unique {dimension_col.lower()} segments were decomposed."
        )

        return " ".join(parts)


def prepare_waterfall_data(
    result: AnalysisResult,
    max_segments: int = 8
) -> Dict[str, Union[List[str], List[float], List[str]]]:
    """
    Prepares structured data for Plotly Waterfall charts.
    Consolidates smaller tail segments into an '(Other Segments)' bucket to ensure
    visual clarity while guaranteeing the chart begins at Baseline and ends at Comparison.
    """
    df = result.breakdown_df.copy()
    dim_col = result.dimension_col

    # Sort by absolute impact
    df_sorted = df.sort_values(by="abs_impact", ascending=False)

    if len(df_sorted) > max_segments:
        top_slice = df_sorted.iloc[:max_segments].copy()
        other_slice = df_sorted.iloc[max_segments:].copy()
        other_change = other_slice["abs_change"].sum()
        
        # Build other row
        other_row = pd.DataFrame([{
            dim_col: f"Other ({len(other_slice)} segments)",
            "abs_change": other_change,
        }])
        plot_df = pd.concat([top_slice, other_row], ignore_index=True)
    else:
        plot_df = df_sorted.copy()

    # Waterfall arrays:
    # 1. Baseline Total (absolute)
    # 2. Segment changes (relative)
    # 3. Comparison Total (total)
    x = ["Baseline (Period A)"]
    y = [result.total_period_a]
    measure = ["absolute"]
    text = [f"{result.total_period_a:,.1f}"]

    for _, row in plot_df.iterrows():
        x.append(str(row[dim_col]))
        y.append(float(row["abs_change"]))
        measure.append("relative")
        text.append(f"{row['abs_change']:+,.1f}")

    x.append("Comparison (Period B)")
    y.append(result.total_period_b)
    measure.append("total")
    text.append(f"{result.total_period_b:,.1f}")

    return {
        "x": x,
        "y": y,
        "measure": measure,
        "text": text,
    }
