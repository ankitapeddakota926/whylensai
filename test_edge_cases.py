"""
test_edge_cases.py
==================
Unit tests verifying all critical edge-case defenses for WhyLens AI:
- Zero denominator in % change (Period A is 0)
- Zero denominator in total change (Delta T is 0)
- Segment present in A but missing in B
- Segment present in B but missing in A
- Missing dates (NaT / invalid formats)
- Missing metric values (NaN)
- Zero rows in period
"""

import unittest
import numpy as np
import pandas as pd
from analyzer import ContributionAnalyzer

class TestContributionAnalyzerEdgeCases(unittest.TestCase):

    def setUp(self):
        self.analyzer = ContributionAnalyzer()

    def test_zero_baseline_segment_pct_change(self):
        """If Period A was 0, % change should not crash the app, output NaN/0."""
        data = pd.DataFrame([
            {"Date": "2024-01-01", "Region": "NewMarket", "Revenue": 0.0},
            {"Date": "2024-02-01", "Region": "NewMarket", "Revenue": 500.0},
            {"Date": "2024-01-01", "Region": "OldMarket", "Revenue": 1000.0},
            {"Date": "2024-02-01", "Region": "OldMarket", "Revenue": 1200.0},
        ])
        result = self.analyzer.analyze(
            df=data,
            date_col="Date",
            metric_col="Revenue",
            dimension_col="Region",
            period_a_start="2024-01-01",
            period_a_end="2024-01-15",
            period_b_start="2024-02-01",
            period_b_end="2024-02-15"
        )
        new_mkt = result.breakdown_df[result.breakdown_df["Region"] == "NewMarket"].iloc[0]
        # Period A is 0, Period B is 500 -> % change should be NaN, not crash
        self.assertTrue(np.isnan(new_mkt["pct_change"]))
        self.assertEqual(new_mkt["abs_change"], 500.0)
        # Total change = (500 + 1200) - (0 + 1000) = 700.0
        # Contribution = 500 / 700 = 71.43%
        self.assertAlmostEqual(new_mkt["contribution_pct"], (500.0 / 700.0) * 100.0)

    def test_zero_total_change_no_crash(self):
        """If total absolute change is 0 (offsetting shifts), contribution % should not raise ZeroDivisionError."""
        data = pd.DataFrame([
            {"Date": "2024-01-01", "Region": "North", "Revenue": 100.0},
            {"Date": "2024-01-01", "Region": "South", "Revenue": 100.0},
            # North gains 50, South loses 50 -> net change = 0
            {"Date": "2024-02-01", "Region": "North", "Revenue": 150.0},
            {"Date": "2024-02-01", "Region": "South", "Revenue": 50.0},
        ])
        result = self.analyzer.analyze(
            df=data,
            date_col="Date",
            metric_col="Revenue",
            dimension_col="Region",
            period_a_start="2024-01-01",
            period_a_end="2024-01-02",
            period_b_start="2024-02-01",
            period_b_end="2024-02-02"
        )
        self.assertEqual(result.total_abs_change, 0.0)
        # Contribution % should be 0.0 without ZeroDivisionError
        for contrib in result.breakdown_df["contribution_pct"]:
            self.assertEqual(contrib, 0.0)
        self.assertIn("effectively zero", result.narrative_summary)

    def test_segment_absent_in_one_period(self):
        """Discontinued or new segment should be preserved via outer join."""
        data = pd.DataFrame([
            {"Date": "2024-01-01", "Category": "DiscontinuedItem", "Sales": 300.0},
            {"Date": "2024-02-01", "Category": "BrandNewItem", "Sales": 400.0},
        ])
        result = self.analyzer.analyze(
            df=data,
            date_col="Date",
            metric_col="Sales",
            dimension_col="Category",
            period_a_start="2024-01-01",
            period_a_end="2024-01-05",
            period_b_start="2024-02-01",
            period_b_end="2024-02-05"
        )
        cats = set(result.breakdown_df["Category"])
        self.assertIn("DiscontinuedItem", cats)
        self.assertIn("BrandNewItem", cats)
        
        disc = result.breakdown_df[result.breakdown_df["Category"] == "DiscontinuedItem"].iloc[0]
        self.assertEqual(disc["period_b_value"], 0.0)
        self.assertEqual(disc["abs_change"], -300.0)

    def test_missing_dates_and_metrics_graceful(self):
        """Malformed dates and NaN metrics are cleaned and logged without crashing."""
        data = pd.DataFrame([
            {"Date": "2024-01-01", "Region": "East", "Sales": 100.0},
            {"Date": "INVALID_DATE_FORMAT", "Region": "East", "Sales": 200.0},
            {"Date": "2024-02-01", "Region": "East", "Sales": np.nan},
            {"Date": "2024-02-01", "Region": "East", "Sales": 150.0},
        ])
        result = self.analyzer.analyze(
            df=data,
            date_col="Date",
            metric_col="Sales",
            dimension_col="Region",
            period_a_start="2024-01-01",
            period_a_end="2024-01-05",
            period_b_start="2024-02-01",
            period_b_end="2024-02-05"
        )
        self.assertEqual(result.quality_stats.dropped_invalid_dates, 1)
        self.assertEqual(result.quality_stats.missing_metrics_handled, 1)
        # Period A: 100.0, Period B: 0.0 + 150.0 = 150.0
        self.assertEqual(result.total_period_a, 100.0)
        self.assertEqual(result.total_period_b, 150.0)

    def test_empty_period_raises_clear_error(self):
        """If a selected date range has 0 rows, raise a clear informative ValueError."""
        data = pd.DataFrame([
            {"Date": "2024-01-01", "Region": "East", "Sales": 100.0},
        ])
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze(
                df=data,
                date_col="Date",
                metric_col="Sales",
                dimension_col="Region",
                period_a_start="2024-05-01",  # No rows here
                period_a_end="2024-05-10",
                period_b_start="2024-01-01",
                period_b_end="2024-01-05"
            )
        self.assertIn("No records found in Period A", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
