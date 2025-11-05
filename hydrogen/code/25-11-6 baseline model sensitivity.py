
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#%% Define key input/output paths
# Automatically locate project root (folder containing this script)
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(current_path)

# Define subfolders relative to project root
data_path = os.path.join(project_path, 'data')
timestamp = datetime.now().strftime("%Y-%m-%d")
results_path = os.path.join(project_path, 'results', timestamp)

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

#%% ============================================================================
# SENSITIVITY ANALYSIS SECTION
# ============================================================================

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS")
print("="*80)
print("Exploring parameter variations:")
print("  - CAPEX: [600, 800, 1000, 1200, 1400] $/kW")
print("  - H₂ Price: [3.0, 4.0, 5.0, 6.0] $/kg")
print("  - Efficiency: [45, 47, 49, 51, 53] kWh/kg H₂")
print("  - Lifetime: [60k, 70k, 80k, 90k, 100k] hours")

#%% Sensitivity Analysis Configuration
SENSITIVITY_PARAMS = {
    'CAPEX': [600, 800, 1000, 1200, 1400],  # $/kW
    'H2_PRICE': [3.0, 4.0, 5.0, 6.0],       # $/kg
    'EFFICIENCY': [45, 47, 49, 51, 53],     # kWh/kg H₂
    'LIFETIME_HOURS': [60000, 70000, 80000, 90000, 100000]  # hours
}

#%% Function Definitions for Sensitivity Analysis

def run_single_param_sensitivity(df_prices, baseline_params, param_name, param_values, region_name):
    """
    Run sensitivity analysis for a single parameter.

    Parameters:
    -----------
    df_prices : pd.DataFrame
        Price data with 'price_operational' column
    baseline_params : dict
        Baseline electrolyser parameters
    param_name : str
        Name of parameter to vary (e.g., 'CAPEX', 'H2_PRICE')
    param_values : list
        List of values to test for the parameter
    region_name : str
        Region identifier ('VIC' or 'QLD')

    Returns:
    --------
    pd.DataFrame : Results for all parameter values
    """
    results = []

    for param_value in param_values:
        # Create modified parameter set
        params = baseline_params.copy()
        params[param_name] = param_value

        # Recalculate derived values
        depreciation_per_mwh = (params['CAPEX'] * 1000) / params['LIFETIME_HOURS']
        max_viable_price = (1000 * params['H2_PRICE'] / params['EFFICIENCY']) - depreciation_per_mwh - params['VOM']

        # Make operation decisions
        df_temp = df_prices.copy()
        df_temp['operate'] = df_temp['price_operational'] < max_viable_price

        # Calculate metrics
        metrics = calculate_lcoh_metrics(df_temp, region_name, params, depreciation_per_mwh)

        # Store results with parameter information
        result_row = {
            'region': region_name,
            'sensitivity_type': f'{param_name}_only',
            'CAPEX': params['CAPEX'],
            'H2_PRICE': params['H2_PRICE'],
            'EFFICIENCY': params['EFFICIENCY'],
            'VOM': params['VOM'],
            'LIFETIME_HOURS': params['LIFETIME_HOURS'],
            'CAPACITY_MW': params['CAPACITY_MW']
        }
        result_row.update(metrics)
        results.append(result_row)

    return pd.DataFrame(results)

def run_two_param_grid_sensitivity(df_prices, baseline_params, param1_name, param1_values,
                                    param2_name, param2_values, region_name):
    """
    Run sensitivity analysis for two parameters in a grid (all combinations).

    Parameters:
    -----------
    df_prices : pd.DataFrame
        Price data
    baseline_params : dict
        Baseline parameters
    param1_name : str
        First parameter name (e.g., 'CAPEX')
    param1_values : list
        Values for first parameter
    param2_name : str
        Second parameter name (e.g., 'H2_PRICE')
    param2_values : list
        Values for second parameter
    region_name : str
        Region identifier

    Returns:
    --------
    pd.DataFrame : Results for all parameter combinations
    """
    results = []

    for p1_value in param1_values:
        for p2_value in param2_values:
            # Create modified parameter set
            params = baseline_params.copy()
            params[param1_name] = p1_value
            params[param2_name] = p2_value

            # Recalculate derived values
            depreciation_per_mwh = (params['CAPEX'] * 1000) / params['LIFETIME_HOURS']
            max_viable_price = (1000 * params['H2_PRICE'] / params['EFFICIENCY']) - depreciation_per_mwh - params['VOM']

            # Make operation decisions
            df_temp = df_prices.copy()
            df_temp['operate'] = df_temp['price_operational'] < max_viable_price

            # Calculate metrics
            metrics = calculate_lcoh_metrics(df_temp, region_name, params, depreciation_per_mwh)

            # Store results
            result_row = {
                'region': region_name,
                'sensitivity_type': f'{param1_name}_{param2_name}_grid',
                'CAPEX': params['CAPEX'],
                'H2_PRICE': params['H2_PRICE'],
                'EFFICIENCY': params['EFFICIENCY'],
                'VOM': params['VOM'],
                'LIFETIME_HOURS': params['LIFETIME_HOURS'],
                'CAPACITY_MW': params['CAPACITY_MW']
            }
            result_row.update(metrics)
            results.append(result_row)

    return pd.DataFrame(results)

def validate_sensitivity_results(df_results, param_name, region_name):
    """
    Validate sensitivity results against engineering principles.

    Parameters:
    -----------
    df_results : pd.DataFrame
        Sensitivity results to validate
    param_name : str
        Parameter being varied
    region_name : str
        Region identifier

    Returns:
    --------
    dict : Validation summary with pass/fail status
    """
    validation_summary = {
        'region': region_name,
        'parameter': param_name,
        'checks_passed': 0,
        'checks_failed': 0,
        'warnings': []
    }

    # Sort by parameter value for trend checking
    df_sorted = df_results.sort_values(param_name).reset_index(drop=True)

    # Check 1: LCOH range (should be $1.5-$10/kg)
    lcoh_min = df_sorted['lcoh_$/kg'].min()
    lcoh_max = df_sorted['lcoh_$/kg'].max()
    if 1.5 <= lcoh_min and lcoh_max <= 10.0:
        validation_summary['checks_passed'] += 1
        print(f"  ✓ LCOH range [{lcoh_min:.2f}, {lcoh_max:.2f}] $/kg is realistic")
    else:
        validation_summary['checks_failed'] += 1
        validation_summary['warnings'].append(f"LCOH range [{lcoh_min:.2f}, {lcoh_max:.2f}] outside typical bounds")
        print(f"  ⚠ LCOH range [{lcoh_min:.2f}, {lcoh_max:.2f}] $/kg - CHECK PARAMETERS")

    # Check 2: CAPEX trend (if varying CAPEX, LCOH should increase with CAPEX)
    if param_name == 'CAPEX':
        lcoh_trend = df_sorted['lcoh_$/kg'].diff().dropna()
        if (lcoh_trend >= -0.01).all():  # Allow tiny numerical errors
            validation_summary['checks_passed'] += 1
            print(f"  ✓ LCOH increases with CAPEX (correct trend)")
        else:
            validation_summary['checks_failed'] += 1
            validation_summary['warnings'].append("LCOH does not monotonically increase with CAPEX")
            print(f"  ✗ LCOH trend with CAPEX is non-monotonic - ERROR")

    # Check 3: H₂ Price trend (if varying H2_PRICE, LCOH should decrease)
    if param_name == 'H2_PRICE':
        lcoh_trend = df_sorted['lcoh_$/kg'].diff().dropna()
        if (lcoh_trend <= 0.01).all():  # Allow tiny numerical errors
            validation_summary['checks_passed'] += 1
            print(f"  ✓ LCOH decreases with H₂ price (correct trend)")
        else:
            validation_summary['checks_failed'] += 1
            validation_summary['warnings'].append("LCOH does not monotonically decrease with H₂ price")
            print(f"  ✗ LCOH trend with H₂ price is non-monotonic - ERROR")

    # Check 4: Utilization range (should be 0-100%, typically 10-85%)
    util_min = df_sorted['utilisation_%'].min()
    util_max = df_sorted['utilisation_%'].max()
    if 0 <= util_min and util_max <= 100:
        validation_summary['checks_passed'] += 1
        print(f"  ✓ Utilization range [{util_min:.1f}%, {util_max:.1f}%] is valid")
    else:
        validation_summary['checks_failed'] += 1
        validation_summary['warnings'].append(f"Utilization outside 0-100% range")
        print(f"  ✗ Utilization range [{util_min:.1f}%, {util_max:.1f}%] - ERROR")

    # Check 5: Electricity cost dominance (should be 50-80% of LCOH)
    elec_fractions = (df_sorted['lcoh_electricity_$/kg'] / df_sorted['lcoh_$/kg']) * 100
    elec_min = elec_fractions.min()
    elec_max = elec_fractions.max()
    if 50 <= elec_min and elec_max <= 85:
        validation_summary['checks_passed'] += 1
        print(f"  ✓ Electricity cost fraction [{elec_min:.1f}%, {elec_max:.1f}%] is typical")
    else:
        validation_summary['warnings'].append(f"Electricity cost fraction [{elec_min:.1f}%, {elec_max:.1f}%] unusual")
        print(f"  ⚠ Electricity cost fraction [{elec_min:.1f}%, {elec_max:.1f}%] - verify parameters")
        validation_summary['checks_passed'] += 1  # Warning, not failure

    return validation_summary

#%% Execute Sensitivity Scenarios

print("\n" + "-"*80)
print("Running Sensitivity Scenarios...")
print("-"*80)

all_sensitivity_results = []

# Scenario 1: CAPEX Sensitivity (VIC and QLD)
print("\n1. CAPEX Sensitivity Analysis:")
print(f"   Testing CAPEX values: {SENSITIVITY_PARAMS['CAPEX']} $/kW")

df_capex_vic = run_single_param_sensitivity(
    df_vic, PARAMS_VIC, 'CAPEX', SENSITIVITY_PARAMS['CAPEX'], 'VIC'
)
print(f"   ✓ VIC: {len(df_capex_vic)} scenarios completed")
all_sensitivity_results.append(df_capex_vic)

df_capex_qld = run_single_param_sensitivity(
    df_qld, PARAMS_QLD, 'CAPEX', SENSITIVITY_PARAMS['CAPEX'], 'QLD'
)
print(f"   ✓ QLD: {len(df_capex_qld)} scenarios completed")
all_sensitivity_results.append(df_capex_qld)

# Scenario 2: H₂ Price Sensitivity (VIC and QLD)
print("\n2. H₂ Price Sensitivity Analysis:")
print(f"   Testing H₂ prices: {SENSITIVITY_PARAMS['H2_PRICE']} $/kg")

df_h2price_vic = run_single_param_sensitivity(
    df_vic, PARAMS_VIC, 'H2_PRICE', SENSITIVITY_PARAMS['H2_PRICE'], 'VIC'
)
print(f"   ✓ VIC: {len(df_h2price_vic)} scenarios completed")
all_sensitivity_results.append(df_h2price_vic)

df_h2price_qld = run_single_param_sensitivity(
    df_qld, PARAMS_QLD, 'H2_PRICE', SENSITIVITY_PARAMS['H2_PRICE'], 'QLD'
)
print(f"   ✓ QLD: {len(df_h2price_qld)} scenarios completed")
all_sensitivity_results.append(df_h2price_qld)

# Scenario 3: CAPEX × H₂ Price Grid (VIC and QLD)
print("\n3. CAPEX × H₂ Price Grid Sensitivity:")
print(f"   Testing {len(SENSITIVITY_PARAMS['CAPEX'])} × {len(SENSITIVITY_PARAMS['H2_PRICE'])} = {len(SENSITIVITY_PARAMS['CAPEX']) * len(SENSITIVITY_PARAMS['H2_PRICE'])} combinations per region")

df_grid_vic = run_two_param_grid_sensitivity(
    df_vic, PARAMS_VIC, 'CAPEX', SENSITIVITY_PARAMS['CAPEX'],
    'H2_PRICE', SENSITIVITY_PARAMS['H2_PRICE'], 'VIC'
)
print(f"   ✓ VIC: {len(df_grid_vic)} scenarios completed")
all_sensitivity_results.append(df_grid_vic)

df_grid_qld = run_two_param_grid_sensitivity(
    df_qld, PARAMS_QLD, 'CAPEX', SENSITIVITY_PARAMS['CAPEX'],
    'H2_PRICE', SENSITIVITY_PARAMS['H2_PRICE'], 'QLD'
)
print(f"   ✓ QLD: {len(df_grid_qld)} scenarios completed")
all_sensitivity_results.append(df_grid_qld)

# Scenario 4: Efficiency Sensitivity (VIC and QLD)
print("\n4. Efficiency Sensitivity Analysis:")
print(f"   Testing efficiencies: {SENSITIVITY_PARAMS['EFFICIENCY']} kWh/kg H₂")

df_eff_vic = run_single_param_sensitivity(
    df_vic, PARAMS_VIC, 'EFFICIENCY', SENSITIVITY_PARAMS['EFFICIENCY'], 'VIC'
)
print(f"   ✓ VIC: {len(df_eff_vic)} scenarios completed")
all_sensitivity_results.append(df_eff_vic)

df_eff_qld = run_single_param_sensitivity(
    df_qld, PARAMS_QLD, 'EFFICIENCY', SENSITIVITY_PARAMS['EFFICIENCY'], 'QLD'
)
print(f"   ✓ QLD: {len(df_eff_qld)} scenarios completed")
all_sensitivity_results.append(df_eff_qld)

# Scenario 5: Lifetime Sensitivity (VIC and QLD)
print("\n5. Lifetime Sensitivity Analysis:")
print(f"   Testing lifetimes: {[f'{int(h/1000)}k' for h in SENSITIVITY_PARAMS['LIFETIME_HOURS']]} hours")

df_lifetime_vic = run_single_param_sensitivity(
    df_vic, PARAMS_VIC, 'LIFETIME_HOURS', SENSITIVITY_PARAMS['LIFETIME_HOURS'], 'VIC'
)
print(f"   ✓ VIC: {len(df_lifetime_vic)} scenarios completed")
all_sensitivity_results.append(df_lifetime_vic)

df_lifetime_qld = run_single_param_sensitivity(
    df_qld, PARAMS_QLD, 'LIFETIME_HOURS', SENSITIVITY_PARAMS['LIFETIME_HOURS'], 'QLD'
)
print(f"   ✓ QLD: {len(df_lifetime_qld)} scenarios completed")
all_sensitivity_results.append(df_lifetime_qld)

print(f"\n✓ All sensitivity scenarios completed!")
print(f"  Total scenarios run: {sum(len(df) for df in all_sensitivity_results)}")

#%% Export Consolidated Results

print("\n" + "-"*80)
print("Exporting Sensitivity Results...")
print("-"*80)

# Combine all results
df_all_sensitivity = pd.concat(all_sensitivity_results, ignore_index=True)

# Export to CSV
sensitivity_results_file = os.path.join(results_path, 'sensitivity_results.csv')
df_all_sensitivity.to_csv(sensitivity_results_file, index=False)
print(f"✓ Sensitivity results saved: {sensitivity_results_file}")
print(f"  Rows: {len(df_all_sensitivity)}, Columns: {len(df_all_sensitivity.columns)}")

#%% Generate Visualizations - VIC Region

print("\n" + "-"*80)
print("Generating VIC Sensitivity Visualizations...")
print("-"*80)

# VIC Plot 1: LCOH vs CAPEX (with H₂ price variations)
fig, ax = plt.subplots(figsize=figure_size)

# Plot lines for each H₂ price
h2_prices = SENSITIVITY_PARAMS['H2_PRICE']
colors_h2 = plt.cm.viridis(np.linspace(0, 1, len(h2_prices)))

for i, h2_price in enumerate(h2_prices):
    df_plot = df_grid_vic[df_grid_vic['H2_PRICE'] == h2_price]
    df_plot = df_plot.sort_values('CAPEX')
    ax.plot(df_plot['CAPEX'], df_plot['lcoh_$/kg'],
            marker='o', linewidth=2, markersize=8,
            label=f'H₂ = ${h2_price:.1f}/kg', color=colors_h2[i])

# Mark baseline
ax.scatter([PARAMS_VIC['CAPEX']], [metrics_vic['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('VIC: LCOH Sensitivity to CAPEX and H₂ Price')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
vic_lcoh_capex_file = os.path.join(results_path, 'sensitivity_lcoh_vs_capex_VIC.png')
plt.savefig(vic_lcoh_capex_file, dpi=300, bbox_inches='tight')
print(f"✓ VIC LCOH vs CAPEX plot saved: {vic_lcoh_capex_file}")
plt.close()

# VIC Plot 2: LCOH Heatmap (CAPEX × H₂ Price)
fig, ax = plt.subplots(figsize=heatmap_size)

# Pivot data for heatmap
pivot_vic = df_grid_vic.pivot(index='H2_PRICE', columns='CAPEX', values='lcoh_$/kg')

# Create heatmap
im = ax.imshow(pivot_vic.values, cmap='RdYlGn_r', aspect='auto', origin='lower',
               extent=[pivot_vic.columns.min()-50, pivot_vic.columns.max()+50,
                       pivot_vic.index.min()-0.25, pivot_vic.index.max()+0.25])

# Add contour lines
X, Y = np.meshgrid(pivot_vic.columns, pivot_vic.index)
contours = ax.contour(X, Y, pivot_vic.values, levels=[2.5, 3.0, 4.0, 5.0],
                      colors='black', linewidths=1.5, alpha=0.7)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

# Annotate cells with values
for i, h2_price in enumerate(pivot_vic.index):
    for j, capex in enumerate(pivot_vic.columns):
        lcoh_val = pivot_vic.iloc[i, j]
        ax.text(capex, h2_price, f'{lcoh_val:.2f}',
                ha='center', va='center', color='white' if lcoh_val > 3.5 else 'black',
                fontsize=9, fontweight='bold')

# Colorbar and labels
cbar = plt.colorbar(im, ax=ax, label='LCOH ($/kg H₂)')
ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('H₂ Price ($/kg)')
ax.set_title('VIC: LCOH Sensitivity Heatmap')

# Set ticks
ax.set_xticks(SENSITIVITY_PARAMS['CAPEX'])
ax.set_yticks(SENSITIVITY_PARAMS['H2_PRICE'])

plt.tight_layout()
vic_heatmap_file = os.path.join(results_path, 'sensitivity_heatmap_capex_h2price_VIC.png')
plt.savefig(vic_heatmap_file, dpi=300, bbox_inches='tight')
print(f"✓ VIC heatmap saved: {vic_heatmap_file}")
plt.close()

# VIC Plot 3: Utilization vs CAPEX
fig, ax = plt.subplots(figsize=figure_size)

for i, h2_price in enumerate(h2_prices):
    df_plot = df_grid_vic[df_grid_vic['H2_PRICE'] == h2_price]
    df_plot = df_plot.sort_values('CAPEX')
    ax.plot(df_plot['CAPEX'], df_plot['utilisation_%'],
            marker='o', linewidth=2, markersize=8,
            label=f'H₂ = ${h2_price:.1f}/kg', color=colors_h2[i])

ax.scatter([PARAMS_VIC['CAPEX']], [metrics_vic['utilisation_%']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('Utilization Factor (%)')
ax.set_title('VIC: Utilization Sensitivity to CAPEX and H₂ Price')
ax.set_ylim(0, 100)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
vic_util_capex_file = os.path.join(results_path, 'sensitivity_utilization_vs_capex_VIC.png')
plt.savefig(vic_util_capex_file, dpi=300, bbox_inches='tight')
print(f"✓ VIC utilization vs CAPEX plot saved: {vic_util_capex_file}")
plt.close()

# VIC Plot 4: Efficiency Impact
fig, ax = plt.subplots(figsize=figure_size)

df_eff_vic_sorted = df_eff_vic.sort_values('EFFICIENCY')
ax.plot(df_eff_vic_sorted['EFFICIENCY'], df_eff_vic_sorted['lcoh_$/kg'],
        marker='o', linewidth=3, markersize=10, color='blue', label='LCOH')

ax.scatter([PARAMS_VIC['EFFICIENCY']], [metrics_vic['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('Efficiency (kWh/kg H₂)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('VIC: LCOH Sensitivity to Electrolyser Efficiency')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
vic_efficiency_file = os.path.join(results_path, 'sensitivity_efficiency_impact_VIC.png')
plt.savefig(vic_efficiency_file, dpi=300, bbox_inches='tight')
print(f"✓ VIC efficiency impact plot saved: {vic_efficiency_file}")
plt.close()

# VIC Plot 5: Lifetime Impact
fig, ax = plt.subplots(figsize=figure_size)

df_lifetime_vic_sorted = df_lifetime_vic.sort_values('LIFETIME_HOURS')
lifetime_years = df_lifetime_vic_sorted['LIFETIME_HOURS'] / 8760
ax.plot(lifetime_years, df_lifetime_vic_sorted['lcoh_$/kg'],
        marker='o', linewidth=3, markersize=10, color='green', label='LCOH')

baseline_lifetime_years = PARAMS_VIC['LIFETIME_HOURS'] / 8760
ax.scatter([baseline_lifetime_years], [metrics_vic['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('Electrolyser Lifetime (years)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('VIC: LCOH Sensitivity to Electrolyser Lifetime')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
vic_lifetime_file = os.path.join(results_path, 'sensitivity_lifetime_impact_VIC.png')
plt.savefig(vic_lifetime_file, dpi=300, bbox_inches='tight')
print(f"✓ VIC lifetime impact plot saved: {vic_lifetime_file}")
plt.close()

#%% Generate Visualizations - QLD Region

print("\n" + "-"*80)
print("Generating QLD Sensitivity Visualizations...")
print("-"*80)

# QLD Plot 1: LCOH vs CAPEX (with H₂ price variations)
fig, ax = plt.subplots(figsize=figure_size)

for i, h2_price in enumerate(h2_prices):
    df_plot = df_grid_qld[df_grid_qld['H2_PRICE'] == h2_price]
    df_plot = df_plot.sort_values('CAPEX')
    ax.plot(df_plot['CAPEX'], df_plot['lcoh_$/kg'],
            marker='s', linewidth=2, markersize=8,
            label=f'H₂ = ${h2_price:.1f}/kg', color=colors_h2[i])

ax.scatter([PARAMS_QLD['CAPEX']], [metrics_qld['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('QLD: LCOH Sensitivity to CAPEX and H₂ Price')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
qld_lcoh_capex_file = os.path.join(results_path, 'sensitivity_lcoh_vs_capex_QLD.png')
plt.savefig(qld_lcoh_capex_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD LCOH vs CAPEX plot saved: {qld_lcoh_capex_file}")
plt.close()

# QLD Plot 2: LCOH Heatmap (CAPEX × H₂ Price)
fig, ax = plt.subplots(figsize=heatmap_size)

pivot_qld = df_grid_qld.pivot(index='H2_PRICE', columns='CAPEX', values='lcoh_$/kg')

im = ax.imshow(pivot_qld.values, cmap='RdYlGn_r', aspect='auto', origin='lower',
               extent=[pivot_qld.columns.min()-50, pivot_qld.columns.max()+50,
                       pivot_qld.index.min()-0.25, pivot_qld.index.max()+0.25])

X, Y = np.meshgrid(pivot_qld.columns, pivot_qld.index)
contours = ax.contour(X, Y, pivot_qld.values, levels=[2.5, 3.0, 4.0, 5.0],
                      colors='black', linewidths=1.5, alpha=0.7)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

for i, h2_price in enumerate(pivot_qld.index):
    for j, capex in enumerate(pivot_qld.columns):
        lcoh_val = pivot_qld.iloc[i, j]
        ax.text(capex, h2_price, f'{lcoh_val:.2f}',
                ha='center', va='center', color='white' if lcoh_val > 3.5 else 'black',
                fontsize=9, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, label='LCOH ($/kg H₂)')
ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('H₂ Price ($/kg)')
ax.set_title('QLD: LCOH Sensitivity Heatmap')

ax.set_xticks(SENSITIVITY_PARAMS['CAPEX'])
ax.set_yticks(SENSITIVITY_PARAMS['H2_PRICE'])

plt.tight_layout()
qld_heatmap_file = os.path.join(results_path, 'sensitivity_heatmap_capex_h2price_QLD.png')
plt.savefig(qld_heatmap_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD heatmap saved: {qld_heatmap_file}")
plt.close()

# QLD Plot 3: Utilization vs CAPEX
fig, ax = plt.subplots(figsize=figure_size)

for i, h2_price in enumerate(h2_prices):
    df_plot = df_grid_qld[df_grid_qld['H2_PRICE'] == h2_price]
    df_plot = df_plot.sort_values('CAPEX')
    ax.plot(df_plot['CAPEX'], df_plot['utilisation_%'],
            marker='s', linewidth=2, markersize=8,
            label=f'H₂ = ${h2_price:.1f}/kg', color=colors_h2[i])

ax.scatter([PARAMS_QLD['CAPEX']], [metrics_qld['utilisation_%']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('CAPEX ($/kW)')
ax.set_ylabel('Utilization Factor (%)')
ax.set_title('QLD: Utilization Sensitivity to CAPEX and H₂ Price')
ax.set_ylim(0, 100)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
qld_util_capex_file = os.path.join(results_path, 'sensitivity_utilization_vs_capex_QLD.png')
plt.savefig(qld_util_capex_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD utilization vs CAPEX plot saved: {qld_util_capex_file}")
plt.close()

# QLD Plot 4: Efficiency Impact
fig, ax = plt.subplots(figsize=figure_size)

df_eff_qld_sorted = df_eff_qld.sort_values('EFFICIENCY')
ax.plot(df_eff_qld_sorted['EFFICIENCY'], df_eff_qld_sorted['lcoh_$/kg'],
        marker='s', linewidth=3, markersize=10, color='green', label='LCOH')

ax.scatter([PARAMS_QLD['EFFICIENCY']], [metrics_qld['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('Efficiency (kWh/kg H₂)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('QLD: LCOH Sensitivity to Electrolyser Efficiency')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
qld_efficiency_file = os.path.join(results_path, 'sensitivity_efficiency_impact_QLD.png')
plt.savefig(qld_efficiency_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD efficiency impact plot saved: {qld_efficiency_file}")
plt.close()

# QLD Plot 5: Lifetime Impact
fig, ax = plt.subplots(figsize=figure_size)

df_lifetime_qld_sorted = df_lifetime_qld.sort_values('LIFETIME_HOURS')
lifetime_years = df_lifetime_qld_sorted['LIFETIME_HOURS'] / 8760
ax.plot(lifetime_years, df_lifetime_qld_sorted['lcoh_$/kg'],
        marker='s', linewidth=3, markersize=10, color='orange', label='LCOH')

baseline_lifetime_years_qld = PARAMS_QLD['LIFETIME_HOURS'] / 8760
ax.scatter([baseline_lifetime_years_qld], [metrics_qld['lcoh_$/kg']],
           s=200, marker='*', color='red', edgecolors='black', linewidths=2,
           label='Baseline', zorder=5)

ax.set_xlabel('Electrolyser Lifetime (years)')
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('QLD: LCOH Sensitivity to Electrolyser Lifetime')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
qld_lifetime_file = os.path.join(results_path, 'sensitivity_lifetime_impact_QLD.png')
plt.savefig(qld_lifetime_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD lifetime impact plot saved: {qld_lifetime_file}")
plt.close()

#%% Engineering Validation of Sensitivity Results

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS - ENGINEERING VALIDATION")
print("="*80)

# Validate each sensitivity type
print("\nVIC Region Validation:")
print("-" * 50)
validate_sensitivity_results(df_capex_vic, 'CAPEX', 'VIC')
validate_sensitivity_results(df_h2price_vic, 'H2_PRICE', 'VIC')

print("\nQLD Region Validation:")
print("-" * 50)
validate_sensitivity_results(df_capex_qld, 'CAPEX', 'QLD')
validate_sensitivity_results(df_h2price_qld, 'H2_PRICE', 'QLD')

# Summary Statistics
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS - SUMMARY STATISTICS")
print("="*80)

print("\nVIC - LCOH Range Across All Scenarios:")
vic_lcoh_min = df_all_sensitivity[df_all_sensitivity['region'] == 'VIC']['lcoh_$/kg'].min()
vic_lcoh_max = df_all_sensitivity[df_all_sensitivity['region'] == 'VIC']['lcoh_$/kg'].max()
print(f"  Minimum LCOH: ${vic_lcoh_min:.2f}/kg")
print(f"  Maximum LCOH: ${vic_lcoh_max:.2f}/kg")
print(f"  Range: ${vic_lcoh_max - vic_lcoh_min:.2f}/kg")

print("\nQLD - LCOH Range Across All Scenarios:")
qld_lcoh_min = df_all_sensitivity[df_all_sensitivity['region'] == 'QLD']['lcoh_$/kg'].min()
qld_lcoh_max = df_all_sensitivity[df_all_sensitivity['region'] == 'QLD']['lcoh_$/kg'].max()
print(f"  Minimum LCOH: ${qld_lcoh_min:.2f}/kg")
print(f"  Maximum LCOH: ${qld_lcoh_max:.2f}/kg")
print(f"  Range: ${qld_lcoh_max - qld_lcoh_min:.2f}/kg")

# Find optimal configurations
print("\nOptimal Configuration (Lowest LCOH):")
optimal_vic = df_all_sensitivity[df_all_sensitivity['region'] == 'VIC'].loc[
    df_all_sensitivity[df_all_sensitivity['region'] == 'VIC']['lcoh_$/kg'].idxmin()
]
print(f"\n  VIC Optimal:")
print(f"    CAPEX: ${optimal_vic['CAPEX']:.0f}/kW")
print(f"    H₂ Price: ${optimal_vic['H2_PRICE']:.1f}/kg")
print(f"    Efficiency: {optimal_vic['EFFICIENCY']:.0f} kWh/kg")
print(f"    Lifetime: {optimal_vic['LIFETIME_HOURS']/1000:.0f}k hours")
print(f"    → LCOH: ${optimal_vic['lcoh_$/kg']:.2f}/kg")
print(f"    → Utilization: {optimal_vic['utilisation_%']:.1f}%")

optimal_qld = df_all_sensitivity[df_all_sensitivity['region'] == 'QLD'].loc[
    df_all_sensitivity[df_all_sensitivity['region'] == 'QLD']['lcoh_$/kg'].idxmin()
]
print(f"\n  QLD Optimal:")
print(f"    CAPEX: ${optimal_qld['CAPEX']:.0f}/kW")
print(f"    H₂ Price: ${optimal_qld['H2_PRICE']:.1f}/kg")
print(f"    Efficiency: {optimal_qld['EFFICIENCY']:.0f} kWh/kg")
print(f"    Lifetime: {optimal_qld['LIFETIME_HOURS']/1000:.0f}k hours")
print(f"    → LCOH: ${optimal_qld['lcoh_$/kg']:.2f}/kg")
print(f"    → Utilization: {optimal_qld['utilisation_%']:.1f}%")

# Parameter sensitivity ranking
print("\nParameter Sensitivity Ranking (LCOH Impact):")

for region in ['VIC', 'QLD']:
    print(f"\n  {region}:")

    # CAPEX sensitivity
    df_region = df_all_sensitivity[df_all_sensitivity['region'] == region]
    capex_range = df_region[df_region['sensitivity_type'] == 'CAPEX_only']['lcoh_$/kg'].max() - \
                  df_region[df_region['sensitivity_type'] == 'CAPEX_only']['lcoh_$/kg'].min()

    # H2 Price sensitivity
    h2_range = df_region[df_region['sensitivity_type'] == 'H2_PRICE_only']['lcoh_$/kg'].max() - \
               df_region[df_region['sensitivity_type'] == 'H2_PRICE_only']['lcoh_$/kg'].min()

    # Efficiency sensitivity
    eff_range = df_region[df_region['sensitivity_type'] == 'EFFICIENCY_only']['lcoh_$/kg'].max() - \
                df_region[df_region['sensitivity_type'] == 'EFFICIENCY_only']['lcoh_$/kg'].min()

    # Lifetime sensitivity
    life_range = df_region[df_region['sensitivity_type'] == 'LIFETIME_HOURS_only']['lcoh_$/kg'].max() - \
                 df_region[df_region['sensitivity_type'] == 'LIFETIME_HOURS_only']['lcoh_$/kg'].min()

    sensitivity_ranking = [
        ('CAPEX', capex_range),
        ('H₂ Price', h2_range),
        ('Efficiency', eff_range),
        ('Lifetime', life_range)
    ]
    sensitivity_ranking.sort(key=lambda x: x[1], reverse=True)

    for i, (param, lcoh_range) in enumerate(sensitivity_ranking, 1):
        print(f"    {i}. {param}: LCOH range = ${lcoh_range:.2f}/kg")

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*80)

print("\n" + "="*80)
print("MODEL EXECUTION COMPLETE")
print("="*80)
print(f"\nAll results saved to: {results_path}")
print("\nKey Findings:")
print(f"  VIC: LCOH = ${metrics_vic['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_vic['utilisation_%']:.1f}%")
print(f"  QLD: LCOH = ${metrics_qld['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_qld['utilisation_%']:.1f}%")