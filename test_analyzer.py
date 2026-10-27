import pandas as pd
from analyzer import ContributionAnalyzer, prepare_waterfall_data

df = pd.read_csv('synthetic_retail_data.csv')
print(f"Read dataset: {len(df)} rows")

analyzer = ContributionAnalyzer()
res = analyzer.analyze(
    df=df,
    date_col='Order_Date',
    metric_col='Sales',
    dimension_col='Region',
    period_a_start='2024-03-01',
    period_a_end='2024-03-31',
    period_b_start='2024-04-01',
    period_b_end='2024-04-30'
)

print('\n--- REGION ANALYSIS (Month 3 vs Month 4) ---')
print(f'Total A: {res.total_period_a:,.2f}')
print(f'Total B: {res.total_period_b:,.2f}')
print(f'Total Change: {res.total_abs_change:+,.2f} ({res.total_pct_change:+.2f}%)')
print('\nBreakdown Matrix:')
print(res.breakdown_df[['Region', 'period_a_value', 'period_b_value', 'abs_change', 'pct_change', 'contribution_pct']])
print('\nNarrative:')
print(res.narrative_summary)

wf = prepare_waterfall_data(res)
print('\nWaterfall payload x:', wf['x'])
print('Waterfall payload measure:', wf['measure'])
