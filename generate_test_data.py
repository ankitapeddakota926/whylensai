"""
generate_test_data.py
======================
Generates synthetic retail order dataset for testing WhyLens AI.
Simulates realistic daily transactions across 6 months with intentional edge cases
and an injected root-cause event (massive drop in West - Electronics during Month 4).
"""

import csv
import random
from datetime import datetime, timedelta
import os

def generate_retail_dataset(
    output_filename: str = "synthetic_retail_data.csv",
    start_date_str: str = "2024-01-01",
    days: int = 182,  # ~6 months (Jan 1, 2024 to June 30, 2024)
    seed: int = 42
) -> str:
    """
    Generates synthetic daily retail sales data.
    
    Injected Scenarios:
    1. Realistic daily sales distribution across 4 regions and 5 categories.
    2. Intentional Root-Cause: In Month 4 (April 2024), 'Electronics' sales in the 'West'
       region plummet by ~85% due to a simulated regional warehouse failure / supply disruption.
    3. Edge Cases:
       - Missing values (empty/NaN) in Sales, Quantity, Region, Category.
       - Zero-sales days for specific segments.
       - Missing/malformed dates to test data cleansing.
    """
    random.seed(seed)
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Grocery"]
    
    # Baseline average daily sales per transaction by category
    base_category_sales = {
        "Electronics": (250.0, 1200.0),
        "Clothing": (40.0, 220.0),
        "Home & Kitchen": (60.0, 450.0),
        "Books": (15.0, 90.0),
        "Grocery": (25.0, 180.0)
    }
    
    # Regional baseline activity multiplier
    region_multiplier = {
        "North": 1.05,
        "South": 0.95,
        "East": 1.15,
        "West": 1.20
    }
    
    rows = []
    
    for day_idx in range(days):
        current_date = start_date + timedelta(days=day_idx)
        date_str = current_date.strftime("%Y-%m-%d")
        month = current_date.month
        
        # Determine if this day is in Month 4 (April: Month of the West-Electronics crisis)
        is_month_4 = (month == 4)
        
        # Generate transactions for each region and category
        for reg in regions:
            for cat in categories:
                # Number of orders for this region/cat on this day
                num_orders = random.randint(2, 6)
                
                # Weekend boost
                if current_date.weekday() in (5, 6):
                    num_orders += random.randint(1, 3)
                    
                # SIMULATED ANOMALY / ROOT CAUSE:
                # In Month 4, West Electronics experiences severe disruption
                if is_month_4 and reg == "West" and cat == "Electronics":
                    # 75% of days have 0 orders; others have drastically reduced order volume
                    if random.random() < 0.75:
                        num_orders = 0
                    else:
                        num_orders = 1
                
                # Check for zero-sales day injection (occasional 0 transactions)
                if num_orders == 0:
                    # Record a row with 0 sales / 0 quantity to test zero handling
                    rows.append({
                        "Order_Date": date_str,
                        "Region": reg,
                        "Category": cat,
                        "Sales": 0.0,
                        "Quantity": 0
                    })
                    continue
                
                for _ in range(num_orders):
                    min_val, max_val = base_category_sales[cat]
                    mult = region_multiplier[reg]
                    
                    # Compute normal sales
                    sales_val = round(random.uniform(min_val, max_val) * mult, 2)
                    qty = random.randint(1, 8)
                    
                    # West Electronics severe price/volume drop if an order does occur in Month 4
                    if is_month_4 and reg == "West" and cat == "Electronics":
                        sales_val = round(sales_val * 0.15, 2)
                        qty = 1
                        
                    # INJECT EDGE CASES (< 2.5% frequency):
                    # 1. Null / NaN Sales
                    sales_entry = sales_val
                    qty_entry = qty
                    reg_entry = reg
                    cat_entry = cat
                    order_date_entry = date_str
                    
                    edge_dice = random.random()
                    if edge_dice < 0.012:
                        # Missing sales value
                        sales_entry = ""
                    elif edge_dice < 0.020:
                        # Zero sales despite positive quantity (e.g. promotional free item)
                        sales_entry = 0.0
                    elif edge_dice < 0.025:
                        # Missing quantity
                        qty_entry = ""
                    elif edge_dice < 0.030:
                        # Missing region
                        reg_entry = ""
                    elif edge_dice < 0.034:
                        # Missing category
                        cat_entry = ""
                    elif edge_dice < 0.038:
                        # Malformed date string (edge case for date parser)
                        order_date_entry = "INVALID_DATE"
                        
                    rows.append({
                        "Order_Date": order_date_entry,
                        "Region": reg_entry,
                        "Category": cat_entry,
                        "Sales": sales_entry,
                        "Quantity": qty_entry
                    })
                    
    # Write to CSV
    output_path = os.path.abspath(output_filename)
    fieldnames = ["Order_Date", "Region", "Category", "Sales", "Quantity"]
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} records in: {output_path}")
    print(f"Date Range: {start_date_str} to {(start_date + timedelta(days=days-1)).strftime('%Y-%m-%d')}")
    print("Injected Root Cause: West Region 'Electronics' sales suffered an ~85% drop during Month 4 (April).")
    print("Injected Edge Cases: Missing Sales, Zero Sales, Missing Dimensions, Malformed Dates.")
    
    return output_path

if __name__ == "__main__":
    generate_retail_dataset()
