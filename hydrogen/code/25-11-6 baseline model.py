#%% Define key input/output paths
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Automatically locate project root (folder containing this script)
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(current_path)

# Define subfolders relative to project root
data_path = os.path.join(project_path, 'data')
results_path = os.path.join(project_path, 'results')

rrp_path_VIC  = os.path.join(data_path, 'AEMO RRP data', 'VIC 2023 data')
rrp_files_VIC = glob.glob(os.path.join(rrp_path_VIC, "PRICE_AND_DEMAND_2023**_VIC1.csv"))
assert len(rrp_files_VIC) > 0, f"No RRP files in {rrp_path_VIC}"

rrp_path_QLD  = os.path.join(data_path, 'AEMO RRP data', 'QLD 2023 data')
rrp_files_QLD = glob.glob(os.path.join(rrp_path_QLD, "PRICE_AND_DEMAND_2023**_QLD1.csv"))
assert len(rrp_files_QLD) > 0, f"No RRP files in {rrp_path_QLD}"

# Create results folder if missing
os.makedirs(results_path, exist_ok=True)

#%% plot fonts
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define optimized figure sizes for large fonts
heatmap_size = (10, 6)         # For heatmaps with colorbars
figure_size = (10, 6)          # Default size
subplot_size_1_2 = (12, 6)
long_hoz_figsize = (14, 6)    # Proportionally scaled
two_by_two_figsize = (14, 12) # Proportionally scaled

#%% Define Electrolyser Parameters (Region-Specific)
"""
Electrolyser techno-economic parameters for VIC and QLD regions.
Parameters differ by region to reflect local conditions and technologies.
"""
# VIC (Victoria) Parameters
PARAMS_VIC = {
    'CAPEX': 950,              # $/kW
    'LIFETIME_HOURS': 80000,   # hours (~9.1 years)
    'EFFICIENCY': 49,          # kWh/kg H₂ (LHV basis)
    'VOM': 4.5,               # $/MWh
    'H2_PRICE': 5.0,          # $/kg
    'CAPACITY_MW': 100         # MW
}

# QLD (Queensland) Parameters
PARAMS_QLD = {
    'CAPEX': 1050,            # $/kW
    'LIFETIME_HOURS': 75000,  # hours (~8.6 years)
    'EFFICIENCY': 51,         # kWh/kg H₂ (LHV basis)
    'VOM': 5.5,              # $/MWh
    'H2_PRICE': 5.2,         # $/kg
    'CAPACITY_MW': 100        # MW
}

print("="*80)
print("GRID-CONNECTED HYDROGEN ELECTROLYSER TECHNO-ECONOMIC MODEL")
print("="*80)
print("\nRegion-Specific Electrolyser Parameters:")
print("\n{:<25} {:>12} {:>12}".format("Parameter", "VIC", "QLD"))
print("-" * 50)
print("{:<25} {:>11}  {:>11}".format("CAPEX ($/kW)", f"${PARAMS_VIC['CAPEX']}", f"${PARAMS_QLD['CAPEX']}"))
print("{:<25} {:>11,}  {:>11,}".format("Lifetime (hours)", PARAMS_VIC['LIFETIME_HOURS'], PARAMS_QLD['LIFETIME_HOURS']))
print("{:<25} {:>11.1f}  {:>11.1f}".format("Lifetime (years)", PARAMS_VIC['LIFETIME_HOURS']/8760, PARAMS_QLD['LIFETIME_HOURS']/8760))
print("{:<25} {:>11}  {:>11}".format("Efficiency (kWh/kg H₂)", PARAMS_VIC['EFFICIENCY'], PARAMS_QLD['EFFICIENCY']))
print("{:<25} {:>11}  {:>11}".format("Variable O&M ($/MWh)", PARAMS_VIC['VOM'], PARAMS_QLD['VOM']))
print("{:<25} {:>11}  {:>11}".format("H₂ Price ($/kg)", PARAMS_VIC['H2_PRICE'], PARAMS_QLD['H2_PRICE']))
print("{:<25} {:>11}  {:>11}".format("Capacity (MW)", PARAMS_VIC['CAPACITY_MW'], PARAMS_QLD['CAPACITY_MW']))
print("="*80)

#%% Load AEMO RRP Data
"""
Load hourly Regional Reference Price (RRP) data from AEMO for Victoria and Queensland.
RRP represents the spot electricity price in $/MWh.
"""
print("\nLoading AEMO RRP Data...")

# Load Victoria data
print(f"  - Loading {len(rrp_files_VIC)} VIC files...")
df_vic_list = []
for file in rrp_files_VIC:
    df_temp = pd.read_csv(file)
    df_vic_list.append(df_temp)
df_vic = pd.concat(df_vic_list, ignore_index=True)

# Load Queensland data
print(f"  - Loading {len(rrp_files_QLD)} QLD files...")
df_qld_list = []
for file in rrp_files_QLD:
    df_temp = pd.read_csv(file)
    df_qld_list.append(df_temp)
df_qld = pd.concat(df_qld_list, ignore_index=True)

# Parse dates and sort
df_vic['SETTLEMENTDATE'] = pd.to_datetime(df_vic['SETTLEMENTDATE'])
df_qld['SETTLEMENTDATE'] = pd.to_datetime(df_qld['SETTLEMENTDATE'])
df_vic = df_vic.sort_values('SETTLEMENTDATE').reset_index(drop=True)
df_qld = df_qld.sort_values('SETTLEMENTDATE').reset_index(drop=True)

# Extract RRP (electricity price in $/MWh)
df_vic['price'] = df_vic['RRP']
df_qld['price'] = df_qld['RRP']

# Handle negative prices (cap at 0 for operational logic - can't run with negative input cost)
# Note: In reality, negative prices incentivize operation, but for LCOH we'll treat as $0 minimum
df_vic['price_operational'] = df_vic['price'].clip(lower=0)
df_qld['price_operational'] = df_qld['price'].clip(lower=0)

print(f"  ✓ VIC: {len(df_vic)} hourly records loaded")
print(f"  ✓ QLD: {len(df_qld)} hourly records loaded")
print(f"  - Date range: {df_vic['SETTLEMENTDATE'].min()} to {df_vic['SETTLEMENTDATE'].max()}")

#%% Calculate Core Economics
"""
Core calculation logic for grid-connected electrolyser operation:
1. Calculate depreciation cost per MWh (region-specific)
2. Determine maximum viable electricity price threshold (region-specific)
3. Make hourly operation decisions based on price vs threshold
"""
print("\nCalculating Economics...")

# Step 1: Depreciation cost per MWh (region-specific)
# Depreciation spreads CAPEX over lifetime operating hours
depreciation_per_mwh_vic = (PARAMS_VIC['CAPEX'] * 1000) / PARAMS_VIC['LIFETIME_HOURS']
depreciation_per_mwh_qld = (PARAMS_QLD['CAPEX'] * 1000) / PARAMS_QLD['LIFETIME_HOURS']

print(f"\n  VIC Depreciation: ${depreciation_per_mwh_vic:.2f}/MWh")
print(f"  QLD Depreciation: ${depreciation_per_mwh_qld:.2f}/MWh")

# Step 2: Maximum viable electricity price (region-specific)
# This is the break-even price where revenue from H₂ equals all costs
# Formula: H₂_price / specific_electricity_consumption - depreciation - VOM
# Where specific consumption = efficiency (kWh/kg) = MWh per 1000 kg

max_viable_price_vic = (1000 * PARAMS_VIC['H2_PRICE'] / PARAMS_VIC['EFFICIENCY']) - depreciation_per_mwh_vic - PARAMS_VIC['VOM']
max_viable_price_qld = (1000 * PARAMS_QLD['H2_PRICE'] / PARAMS_QLD['EFFICIENCY']) - depreciation_per_mwh_qld - PARAMS_QLD['VOM']

print(f"\n  VIC Maximum viable price: ${max_viable_price_vic:.2f}/MWh")
print(f"      (Operate when electricity < ${max_viable_price_vic:.2f}/MWh)")
print(f"  QLD Maximum viable price: ${max_viable_price_qld:.2f}/MWh")
print(f"      (Operate when electricity < ${max_viable_price_qld:.2f}/MWh)")

# Step 3: Hourly operation decisions (using region-specific thresholds)
df_vic['operate'] = df_vic['price_operational'] < max_viable_price_vic
df_qld['operate'] = df_qld['price_operational'] < max_viable_price_qld

print(f"\n  VIC Operations:")
print(f"    - Hours operating: {df_vic['operate'].sum():,} / {len(df_vic):,}")
print(f"    - Utilisation: {df_vic['operate'].mean()*100:.1f}%")

print(f"\n  QLD Operations:")
print(f"    - Hours operating: {df_qld['operate'].sum():,} / {len(df_qld):,}")
print(f"    - Utilisation: {df_qld['operate'].mean()*100:.1f}%")

#%% Calculate LCOH and Performance Metrics
"""
Calculate Levelised Cost of Hydrogen (LCOH) for both regions.
LCOH = (Total electricity cost + Total depreciation + Total O&M) / Total H₂ produced

Also compute key performance metrics:
- Utilisation factor
- Total H₂ production
- Average operating price
"""

def calculate_lcoh_metrics(df, region_name, params, depreciation_per_mwh):
    """
    Calculate LCOH and performance metrics for a given region using region-specific parameters.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: 'price_operational', 'operate'
    region_name : str
        Name of region (for printing)
    params : dict
        Region-specific electrolyser parameters (CAPEX, EFFICIENCY, VOM, etc.)
    depreciation_per_mwh : float
        Pre-calculated depreciation cost per MWh for this region

    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    # Operating hours
    operating_hours = df['operate'].sum()
    total_hours = len(df)
    utilisation = (operating_hours / total_hours) * 100

    # Energy consumed (MWh)
    energy_consumed_mwh = operating_hours * params['CAPACITY_MW']

    # Hydrogen produced (kg)
    # H₂ production = Energy consumed / Efficiency
    h2_produced_kg = (energy_consumed_mwh * 1000) / params['EFFICIENCY']  # Convert MWh to kWh
    h2_produced_tonnes = h2_produced_kg / 1000

    # Cost components
    electricity_cost = (df.loc[df['operate'], 'price_operational'] * params['CAPACITY_MW']).sum()
    depreciation_cost = depreciation_per_mwh * energy_consumed_mwh
    vom_cost = params['VOM'] * energy_consumed_mwh
    total_cost = electricity_cost + depreciation_cost + vom_cost

    # LCOH calculation
    if h2_produced_kg > 0:
        lcoh = total_cost / h2_produced_kg
    else:
        lcoh = np.inf

    # Cost breakdown
    lcoh_electricity = electricity_cost / h2_produced_kg if h2_produced_kg > 0 else 0
    lcoh_depreciation = depreciation_cost / h2_produced_kg if h2_produced_kg > 0 else 0
    lcoh_vom = vom_cost / h2_produced_kg if h2_produced_kg > 0 else 0

    # Average electricity price when operating
    avg_operating_price = df.loc[df['operate'], 'price_operational'].mean()

    return {
        'region': region_name,
        'operating_hours': operating_hours,
        'total_hours': total_hours,
        'utilisation_%': utilisation,
        'energy_consumed_mwh': energy_consumed_mwh,
        'h2_produced_kg': h2_produced_kg,
        'h2_produced_tonnes': h2_produced_tonnes,
        'electricity_cost_$': electricity_cost,
        'depreciation_cost_$': depreciation_cost,
        'vom_cost_$': vom_cost,
        'total_cost_$': total_cost,
        'lcoh_$/kg': lcoh,
        'lcoh_electricity_$/kg': lcoh_electricity,
        'lcoh_depreciation_$/kg': lcoh_depreciation,
        'lcoh_vom_$/kg': lcoh_vom,
        'avg_operating_price_$/mwh': avg_operating_price,
    }

# Calculate for both regions
print("\n" + "="*80)
print("RESULTS - VIC")
print("="*80)
metrics_vic = calculate_lcoh_metrics(df_vic, 'VIC', PARAMS_VIC, depreciation_per_mwh_vic)
for key, value in metrics_vic.items():
    if key == 'region':
        continue
    if isinstance(value, float):
        if 'cost' in key or 'lcoh' in key:
            print(f"  {key}: ${value:,.2f}")
        elif '%' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:,.2f}")
    else:
        print(f"  {key}: {value:,}")

print("\n" + "="*80)
print("RESULTS - QLD")
print("="*80)
metrics_qld = calculate_lcoh_metrics(df_qld, 'QLD', PARAMS_QLD, depreciation_per_mwh_qld)
for key, value in metrics_qld.items():
    if key == 'region':
        continue
    if isinstance(value, float):
        if 'cost' in key or 'lcoh' in key:
            print(f"  {key}: ${value:,.2f}")
        elif '%' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:,.2f}")
    else:
        print(f"  {key}: {value:,}")

#%% Engineering Validation Checks
"""
Verify that results are within expected ranges for grid-connected electrolysers.
"""
print("\n" + "="*80)
print("ENGINEERING VALIDATION")
print("="*80)

# LCOH range check (typical grid-connected: $3-8/kg)
for metrics in [metrics_vic, metrics_qld]:
    region = metrics['region']
    lcoh = metrics['lcoh_$/kg']
    if 3 <= lcoh <= 8:
        status = "✓ REASONABLE"
    elif lcoh < 3:
        status = "⚠ VERY LOW (check prices/parameters)"
    else:
        status = "⚠ HIGH (limited operation due to high prices)"
    print(f"{region} LCOH: ${lcoh:.2f}/kg - {status}")

# Utilisation check (price-arbitrage typically 30-60%)
for metrics in [metrics_vic, metrics_qld]:
    region = metrics['region']
    util = metrics['utilisation_%']
    if 30 <= util <= 60:
        status = "✓ TYPICAL for price-arbitrage operation"
    elif util < 30:
        status = "⚠ LOW - prices rarely profitable"
    else:
        status = "✓ HIGH - good price conditions"
    print(f"{region} Utilisation: {util:.1f}% - {status}")

# Cost component breakdown
for metrics in [metrics_vic, metrics_qld]:
    region = metrics['region']
    elec_pct = (metrics['lcoh_electricity_$/kg'] / metrics['lcoh_$/kg']) * 100
    depr_pct = (metrics['lcoh_depreciation_$/kg'] / metrics['lcoh_$/kg']) * 100
    vom_pct = (metrics['lcoh_vom_$/kg'] / metrics['lcoh_$/kg']) * 100
    print(f"\n{region} LCOH Breakdown:")
    print(f"  - Electricity: {elec_pct:.1f}%")
    print(f"  - Depreciation: {depr_pct:.1f}%")
    print(f"  - Variable O&M: {vom_pct:.1f}%")
    if 60 <= elec_pct <= 80:
        print(f"  ✓ Electricity dominance is typical")
    else:
        print(f"  ⚠ Unusual cost distribution - verify parameters")

#%% Export Results to CSV
"""
Export detailed results and summary statistics to CSV files.
"""
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Summary comparison table
summary_df = pd.DataFrame([metrics_vic, metrics_qld])
summary_file = os.path.join(results_path, 'lcoh_summary_vic_qld.csv')
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary saved: {summary_file}")

# Detailed hourly results for VIC
df_vic_export = df_vic[['SETTLEMENTDATE', 'RRP', 'price_operational', 'operate']].copy()
df_vic_export['region'] = 'VIC'
df_vic_export['energy_consumed_mw'] = df_vic_export['operate'] * PARAMS_VIC['CAPACITY_MW']
df_vic_export['h2_produced_kg'] = (df_vic_export['energy_consumed_mw'] * 1000) / PARAMS_VIC['EFFICIENCY']
vic_detail_file = os.path.join(results_path, 'hourly_results_vic.csv')
df_vic_export.to_csv(vic_detail_file, index=False)
print(f"✓ VIC hourly results saved: {vic_detail_file}")

# Detailed hourly results for QLD
df_qld_export = df_qld[['SETTLEMENTDATE', 'RRP', 'price_operational', 'operate']].copy()
df_qld_export['region'] = 'QLD'
df_qld_export['energy_consumed_mw'] = df_qld_export['operate'] * PARAMS_QLD['CAPACITY_MW']
df_qld_export['h2_produced_kg'] = (df_qld_export['energy_consumed_mw'] * 1000) / PARAMS_QLD['EFFICIENCY']
qld_detail_file = os.path.join(results_path, 'hourly_results_qld.csv')
df_qld_export.to_csv(qld_detail_file, index=False)
print(f"✓ QLD hourly results saved: {qld_detail_file}")

#%% Generate Visualizations
"""
Create comparison plots for VIC vs QLD analysis.
"""
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Electricity price distribution comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=subplot_size_1_2)

ax1.hist(df_vic['price'].clip(-100, 300), bins=50, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(max_viable_price_vic, color='red', linestyle='--', linewidth=2, label=f'Threshold: ${max_viable_price_vic:.0f}/MWh')
ax1.set_xlabel('Electricity Price ($/MWh)')
ax1.set_ylabel('Frequency (hours)')
ax1.set_title('VIC Electricity Price Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(df_qld['price'].clip(-100, 300), bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(max_viable_price_qld, color='red', linestyle='--', linewidth=2, label=f'Threshold: ${max_viable_price_qld:.0f}/MWh')
ax2.set_xlabel('Electricity Price ($/MWh)')
ax2.set_ylabel('Frequency (hours)')
ax2.set_title('QLD Electricity Price Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
price_dist_file = os.path.join(results_path, 'price_distribution_comparison.png')
plt.savefig(price_dist_file, dpi=300, bbox_inches='tight')
print(f"✓ Price distribution plot saved: {price_dist_file}")
plt.close()

# 2. LCOH and Utilisation Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=subplot_size_1_2)

regions = ['VIC', 'QLD']
lcoh_values = [metrics_vic['lcoh_$/kg'], metrics_qld['lcoh_$/kg']]
util_values = [metrics_vic['utilisation_%'], metrics_qld['utilisation_%']]

bars1 = ax1.bar(regions, lcoh_values, color=['blue', 'green'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('LCOH ($/kg H₂)')
ax1.set_title('Levelised Cost of Hydrogen')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, lcoh_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'${val:.2f}/kg', ha='center', va='bottom', fontweight='bold')

bars2 = ax2.bar(regions, util_values, color=['blue', 'green'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Utilisation Factor (%)')
ax2.set_title('Electrolyser Utilisation')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, util_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
comparison_file = os.path.join(results_path, 'lcoh_utilisation_comparison.png')
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"✓ LCOH/Utilisation comparison saved: {comparison_file}")
plt.close()

# 3. LCOH Component Breakdown (Stacked Bar Chart)
fig, ax = plt.subplots(figsize=figure_size)

elec_costs = [metrics_vic['lcoh_electricity_$/kg'], metrics_qld['lcoh_electricity_$/kg']]
depr_costs = [metrics_vic['lcoh_depreciation_$/kg'], metrics_qld['lcoh_depreciation_$/kg']]
vom_costs = [metrics_vic['lcoh_vom_$/kg'], metrics_qld['lcoh_vom_$/kg']]

width = 0.5
x = np.arange(len(regions))

p1 = ax.bar(x, elec_costs, width, label='Electricity', color='#1f77b4')
p2 = ax.bar(x, depr_costs, width, bottom=elec_costs, label='Depreciation', color='#ff7f0e')
p3 = ax.bar(x, vom_costs, width, bottom=np.array(elec_costs)+np.array(depr_costs), label='Variable O&M', color='#2ca02c')

ax.set_ylabel('LCOH Component ($/kg H₂)')
ax.set_title('LCOH Cost Breakdown by Region')
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
breakdown_file = os.path.join(results_path, 'lcoh_cost_breakdown.png')
plt.savefig(breakdown_file, dpi=300, bbox_inches='tight')
print(f"✓ Cost breakdown plot saved: {breakdown_file}")
plt.close()

# 4. Monthly Operating Profile
df_vic['month'] = df_vic['SETTLEMENTDATE'].dt.to_period('M')
df_qld['month'] = df_qld['SETTLEMENTDATE'].dt.to_period('M')

monthly_vic = df_vic.groupby('month')['operate'].mean() * 100
monthly_qld = df_qld.groupby('month')['operate'].mean() * 100

fig, ax = plt.subplots(figsize=long_hoz_figsize)
x_months = range(len(monthly_vic))
ax.plot(x_months, monthly_vic.values, marker='o', linewidth=2, markersize=8, label='VIC', color='blue')
ax.plot(x_months, monthly_qld.values, marker='s', linewidth=2, markersize=8, label='QLD', color='green')
ax.set_xlabel('Month (2023)')
ax.set_ylabel('Utilisation Factor (%)')
ax.set_title('Monthly Electrolyser Utilisation')
ax.set_xticks(x_months)
ax.set_xticklabels([str(m) for m in monthly_vic.index], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
monthly_file = os.path.join(results_path, 'monthly_utilisation.png')
plt.savefig(monthly_file, dpi=300, bbox_inches='tight')
print(f"✓ Monthly utilisation plot saved: {monthly_file}")
plt.close()

# 5. Cumulative H₂ Production Over Time
df_vic['h2_cumulative'] = (df_vic['operate'] * PARAMS_VIC['CAPACITY_MW'] * 1000 / PARAMS_VIC['EFFICIENCY']).cumsum() / 1000  # tonnes
df_qld['h2_cumulative'] = (df_qld['operate'] * PARAMS_QLD['CAPACITY_MW'] * 1000 / PARAMS_QLD['EFFICIENCY']).cumsum() / 1000  # tonnes

fig, ax = plt.subplots(figsize=long_hoz_figsize)
ax.plot(df_vic['SETTLEMENTDATE'], df_vic['h2_cumulative'], linewidth=2, label='VIC', color='blue')
ax.plot(df_qld['SETTLEMENTDATE'], df_qld['h2_cumulative'], linewidth=2, label='QLD', color='green')
ax.set_xlabel('Date (2023)')
ax.set_ylabel('Cumulative H₂ Production (tonnes)')
ax.set_title('Cumulative Hydrogen Production')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
cumulative_file = os.path.join(results_path, 'cumulative_h2_production.png')
plt.savefig(cumulative_file, dpi=300, bbox_inches='tight')
print(f"✓ Cumulative production plot saved: {cumulative_file}")
plt.close()

print("\n" + "="*80)
print("MODEL EXECUTION COMPLETE")
print("="*80)
print(f"\nAll results saved to: {results_path}")
print("\nKey Findings:")
print(f"  VIC: LCOH = ${metrics_vic['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_vic['utilisation_%']:.1f}%")
print(f"  QLD: LCOH = ${metrics_qld['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_qld['utilisation_%']:.1f}%")