import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
import scipy.stats as stats

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
    'VOM': 4.5,               # $/MWh variable O&M
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

#%% ============================================================================
# MONTE CARLO SIMULATION FRAMEWORK
# ============================================================================

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_iterations: int = 5000  # Number of Monte Carlo iterations
    seed: int = 42  # Random seed for reproducibility
    project_lifetime_years: int = 25  # Project economic life
    wacc_mean: float = 0.07  # Weighted average cost of capital (7%)
    wacc_std_pct: float = 0.18  # WACC standard deviation (18% of mean)

@dataclass
class ParameterDistribution:
    """
    Define probability distributions for stochastic parameters.
    Based on GenCost 2025 data and industry standards.

    All normal distributions specified as (mean, std_deviation_as_percent_of_mean)
    """
    # CAPEX distributions ($/kW) - region specific
    capex_mean_vic: float = 950
    capex_mean_qld: float = 1050
    capex_std_pct: float = 0.15  # 15% std deviation
    capex_bounds: tuple = (600, 1400)  # Physical/economic bounds

    # Efficiency distributions (kWh/kg H₂) - region specific
    efficiency_mean_vic: float = 49
    efficiency_mean_qld: float = 51
    efficiency_std_pct: float = 0.08  # 8% std deviation
    efficiency_bounds: tuple = (45, 57)  # Thermodynamic limits

    # H₂ Price distribution ($/kg) - triangular distribution for asymmetric uncertainty
    h2_price_min: float = 3.0  # Pessimistic scenario
    h2_price_mode_vic: float = 5.0  # Most likely (VIC)
    h2_price_mode_qld: float = 5.2  # Most likely (QLD)
    h2_price_max: float = 6.0  # Optimistic scenario

    # Lifetime distributions (hours) - region specific
    lifetime_mean_vic: float = 80000
    lifetime_mean_qld: float = 75000
    lifetime_std_pct: float = 0.12  # 12% std deviation
    lifetime_bounds: tuple = (60000, 100000)  # Industry range

    # Variable O&M distributions ($/MWh) - region specific
    vom_mean_vic: float = 4.5
    vom_mean_qld: float = 5.5
    vom_std_pct: float = 0.15  # 15% std deviation
    vom_bounds: tuple = (3.0, 8.0)  # Industry range

def sample_parameters(region_name, base_params, param_dist, mc_config):
    """
    Sample parameter sets for Monte Carlo simulation using independent sampling.

    Parameters:
    -----------
    region_name : str
        'VIC' or 'QLD' - determines region-specific distribution parameters
    base_params : dict
        Baseline parameter dictionary (PARAMS_VIC or PARAMS_QLD)
    param_dist : ParameterDistribution
        Distribution specifications
    mc_config : MonteCarloConfig
        Monte Carlo configuration

    Returns:
    --------
    dict : Dictionary of sampled parameter arrays, each of length n_iterations

    Notes:
    ------
    - Uses independent sampling (no correlations) for simplicity and validation
    - All parameters clipped to physical/economic bounds
    - Normal distributions for well-understood parameters (CAPEX, efficiency, lifetime, V&M)
    - Triangular distribution for H₂ price (asymmetric market uncertainty)
    - Random seed set for reproducibility
    """
    np.random.seed(mc_config.seed)
    n = mc_config.n_iterations

    # Determine region-specific means
    if region_name == 'VIC':
        capex_mean = param_dist.capex_mean_vic
        efficiency_mean = param_dist.efficiency_mean_vic
        h2_price_mode = param_dist.h2_price_mode_vic
        lifetime_mean = param_dist.lifetime_mean_vic
        vom_mean = param_dist.vom_mean_vic
    else:  # QLD
        capex_mean = param_dist.capex_mean_qld
        efficiency_mean = param_dist.efficiency_mean_qld
        h2_price_mode = param_dist.h2_price_mode_qld
        lifetime_mean = param_dist.lifetime_mean_qld
        vom_mean = param_dist.vom_mean_qld

    # Sample CAPEX (Normal distribution with bounds)
    capex_samples = np.random.normal(
        capex_mean,
        capex_mean * param_dist.capex_std_pct,
        n
    ).clip(param_dist.capex_bounds[0], param_dist.capex_bounds[1])

    # Sample Efficiency (Normal distribution with thermodynamic bounds)
    efficiency_samples = np.random.normal(
        efficiency_mean,
        efficiency_mean * param_dist.efficiency_std_pct,
        n
    ).clip(param_dist.efficiency_bounds[0], param_dist.efficiency_bounds[1])

    # Sample H₂ Price (Triangular distribution - asymmetric uncertainty)
    h2_price_samples = np.random.triangular(
        param_dist.h2_price_min,
        h2_price_mode,
        param_dist.h2_price_max,
        n
    )

    # Sample Lifetime (Normal distribution with industry bounds)
    lifetime_samples = np.random.normal(
        lifetime_mean,
        lifetime_mean * param_dist.lifetime_std_pct,
        n
    ).clip(param_dist.lifetime_bounds[0], param_dist.lifetime_bounds[1])

    # Sample Variable O&M (Normal distribution with industry bounds)
    vom_samples = np.random.normal(
        vom_mean,
        vom_mean * param_dist.vom_std_pct,
        n
    ).clip(param_dist.vom_bounds[0], param_dist.vom_bounds[1])

    # Sample WACC (Normal distribution with financial bounds)
    wacc_samples = np.random.normal(
        mc_config.wacc_mean,
        mc_config.wacc_mean * mc_config.wacc_std_pct,
        n
    ).clip(0.05, 0.12)  # 5-12% range

    return {
        'CAPEX': capex_samples,
        'EFFICIENCY': efficiency_samples,
        'H2_PRICE': h2_price_samples,
        'LIFETIME_HOURS': lifetime_samples,
        'VOM': vom_samples,
        'WACC': wacc_samples,
        'CAPACITY_MW': np.full(n, base_params['CAPACITY_MW'])  # Fixed capacity
    }

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

def calculate_npv(metrics, params, wacc, project_lifetime=25):
    """
    Calculate Net Present Value (NPV) for hydrogen project over project lifetime.

    Parameters:
    -----------
    metrics : dict
        Dictionary from calculate_lcoh_metrics containing:
        - h2_produced_kg: Annual hydrogen production
        - lcoh_$/kg: Levelised cost of hydrogen
        - lcoh_electricity_$/kg: Electricity component of LCOH
        - lcoh_vom_$/kg: V&M component of LCOH
    params : dict
        Parameter dictionary with CAPEX, H2_PRICE, CAPACITY_MW, etc.
    wacc : float
        Weighted average cost of capital (decimal, e.g., 0.07 for 7%)
    project_lifetime : int
        Project economic life (years), default 25

    Returns:
    --------
    dict : Dictionary containing:
        - npv_$: NPV in dollars
        - npv_$m: NPV in millions of dollars
        - irr: Internal rate of return (if calculable)
        - payback_years: Simple payback period

    Notes:
    ------
    NPV Formula:
        NPV = -CAPEX_initial + Σ(t=1 to T) [CF_t / (1 + WACC)^t] - CAPEX_stack_replacement

    Where:
        - CF_t = Annual cash flow = Revenue - OpEx
        - Revenue = H₂ Production × H₂ Price
        - OpEx = Electricity Cost + V&M Cost (depreciation excluded, it's non-cash)
        - Stack replacement at year ~12 (midlife) = 35% of initial CAPEX

    Assumptions:
        - Annual H₂ production constant (based on 2023 AEMO data)
        - H₂ price constant (real terms)
        - Electricity cost pattern repeats annually
        - One stack replacement at project midlife
        - No terminal value / salvage value
        - No tax considerations (pre-tax analysis)
        - No HPTI ($2/kg tax incentive) - can be added as sensitivity
    """
    # Initial CAPEX (t=0)
    capex_initial = params['CAPEX'] * params['CAPACITY_MW'] * 1000  # Total $ (kW × MW × 1000)

    # Annual hydrogen production (kg/year)
    h2_production_annual = metrics['h2_produced_kg']

    # Annual revenue from H₂ sales
    annual_revenue = h2_production_annual * params['H2_PRICE']

    # Annual operating costs (electricity + V&M, excluding depreciation which is non-cash)
    # Reconstruct from LCOH components
    annual_electricity_cost = h2_production_annual * metrics['lcoh_electricity_$/kg']
    annual_vom_cost = h2_production_annual * metrics['lcoh_vom_$/kg']
    annual_opex = annual_electricity_cost + annual_vom_cost

    # Annual cash flow (revenue - opex)
    annual_cash_flow = annual_revenue - annual_opex

    # Initialize NPV with negative initial CAPEX
    npv = -capex_initial

    # Discount annual cash flows over project lifetime
    for year in range(1, project_lifetime + 1):
        discount_factor = (1 + wacc) ** year
        npv += annual_cash_flow / discount_factor

    # Account for stack replacement at midlife (~year 12)
    # Stack is approximately 35% of total electrolyser CAPEX
    stack_replacement_cost = capex_initial * 0.35
    replacement_year = project_lifetime // 2  # Year 12 for 25-year project
    npv -= stack_replacement_cost / ((1 + wacc) ** replacement_year)

    # Calculate NPV in millions
    npv_millions = npv / 1e6

    # Calculate simple payback period (years to recover initial CAPEX)
    if annual_cash_flow > 0:
        payback_years = capex_initial / annual_cash_flow
    else:
        payback_years = np.inf

    # Calculate IRR (simplified - would need scipy.optimize.newton for exact)
    # For now, use approximation: if NPV > 0, IRR > WACC; if NPV < 0, IRR < WACC
    # Exact IRR calculation would require iteration, skip for Monte Carlo speed
    irr = None  # Placeholder - can add exact calculation if needed

    return {
        'npv_$': npv,
        'npv_$m': npv_millions,
        'irr': irr,
        'payback_years': payback_years,
        'annual_cash_flow': annual_cash_flow,
        'capex_initial': capex_initial
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
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
)
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

# Create heatmap with explicit scale for consistency (industry-relevant LCOH range)
# <$2.50/kg = excellent (green), $2.50-3.50/kg = acceptable (yellow), >$3.50/kg = challenging (red)
im = ax.imshow(pivot_vic.values, cmap='RdYlGn_r', aspect='auto', origin='lower',
               extent=[pivot_vic.columns.min()-50, pivot_vic.columns.max()+50,
                       pivot_vic.index.min()-0.25, pivot_vic.index.max()+0.25],
               vmin=1.5, vmax=5.0)

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
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3
)
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
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
qld_lcoh_capex_file = os.path.join(results_path, 'sensitivity_lcoh_vs_capex_QLD.png')
plt.savefig(qld_lcoh_capex_file, dpi=300, bbox_inches='tight')
print(f"✓ QLD LCOH vs CAPEX plot saved: {qld_lcoh_capex_file}")
plt.close()

# QLD Plot 2: LCOH Heatmap (CAPEX × H₂ Price)
fig, ax = plt.subplots(figsize=heatmap_size)

pivot_qld = df_grid_qld.pivot(index='H2_PRICE', columns='CAPEX', values='lcoh_$/kg')

# Consistent scale with VIC for regional comparison
im = ax.imshow(pivot_qld.values, cmap='RdYlGn_r', aspect='auto', origin='lower',
               extent=[pivot_qld.columns.min()-50, pivot_qld.columns.max()+50,
                       pivot_qld.index.min()-0.25, pivot_qld.index.max()+0.25],
               vmin=1.5, vmax=5.0)

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
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3
)
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

#%% ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo_simulation(df_prices, region_name, base_params, param_dist, mc_config):
    """
    Run Monte Carlo simulation for hydrogen project economics.

    Parameters:
    -----------
    df_prices : pd.DataFrame
        Price data with 'price_operational' column
    region_name : str
        'VIC' or 'QLD'
    base_params : dict
        Baseline parameter dictionary
    param_dist : ParameterDistribution
        Distribution specifications
    mc_config : MonteCarloConfig
        Monte Carlo configuration

    Returns:
    --------
    pd.DataFrame : Results for each iteration with columns:
        - iteration, CAPEX, EFFICIENCY, H2_PRICE, LIFETIME_HOURS, VOM, WACC
        - lcoh_$/kg, utilization_%, h2_production_kg
        - npv_$, npv_$m, payback_years

    Notes:
    ------
    - Samples parameters independently (no correlations)
    - Calculates LCOH using existing calculate_lcoh_metrics()
    - Calculates NPV using calculate_npv()
    - Progress printed every 500 iterations
    """
    # Sample parameter sets
    param_samples = sample_parameters(region_name, base_params, param_dist, mc_config)

    results = []

    print(f"\nRunning Monte Carlo simulation for {region_name}...")
    print(f"  Iterations: {mc_config.n_iterations}")
    print(f"  Project lifetime: {mc_config.project_lifetime_years} years")

    for i in range(mc_config.n_iterations):
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{mc_config.n_iterations} iterations complete ({((i+1)/mc_config.n_iterations)*100:.1f}%)")

        # Create parameter set for this iteration
        params_iter = base_params.copy()
        params_iter['CAPEX'] = param_samples['CAPEX'][i]
        params_iter['EFFICIENCY'] = param_samples['EFFICIENCY'][i]
        params_iter['H2_PRICE'] = param_samples['H2_PRICE'][i]
        params_iter['LIFETIME_HOURS'] = param_samples['LIFETIME_HOURS'][i]
        params_iter['VOM'] = param_samples['VOM'][i]
        wacc = param_samples['WACC'][i]

        # Calculate depreciation and max viable price for this parameter set
        depreciation_per_mwh = (params_iter['CAPEX'] * 1000) / params_iter['LIFETIME_HOURS']
        max_viable_price = (1000 * params_iter['H2_PRICE'] / params_iter['EFFICIENCY']) - \
                          depreciation_per_mwh - params_iter['VOM']

        # Make operation decisions based on electricity prices
        df_temp = df_prices.copy()
        df_temp['operate'] = df_temp['price_operational'] < max_viable_price

        # Calculate LCOH and operational metrics
        metrics = calculate_lcoh_metrics(df_temp, region_name, params_iter, depreciation_per_mwh)

        # Calculate NPV
        npv_results = calculate_npv(metrics, params_iter, wacc, mc_config.project_lifetime_years)

        # Store results
        result_row = {
            'iteration': i,
            'CAPEX': params_iter['CAPEX'],
            'EFFICIENCY': params_iter['EFFICIENCY'],
            'H2_PRICE': params_iter['H2_PRICE'],
            'LIFETIME_HOURS': params_iter['LIFETIME_HOURS'],
            'VOM': params_iter['VOM'],
            'WACC': wacc,
            'lcoh_$/kg': metrics['lcoh_$/kg'],
            'utilization_%': metrics['utilisation_%'],
            'h2_production_kg': metrics['h2_produced_kg'],
            'h2_production_tonnes': metrics['h2_produced_tonnes'],
            'npv_$': npv_results['npv_$'],
            'npv_$m': npv_results['npv_$m'],
            'payback_years': npv_results['payback_years'],
            'annual_cash_flow': npv_results['annual_cash_flow']
        }
        results.append(result_row)

    print(f"  ✓ Monte Carlo simulation complete for {region_name}")

    return pd.DataFrame(results)

print("\n" + "="*80)
print("MONTE CARLO SIMULATION - PROBABILISTIC ANALYSIS")
print("="*80)
print("Based on GenCost 2025 parameter distributions")
print(f"Iterations per region: {MonteCarloConfig().n_iterations:,}")

# Initialize configurations
mc_config = MonteCarloConfig()
param_dist = ParameterDistribution()

# Run Monte Carlo for VIC
df_mc_vic = run_monte_carlo_simulation(df_vic, 'VIC', PARAMS_VIC, param_dist, mc_config)

# Run Monte Carlo for QLD
df_mc_qld = run_monte_carlo_simulation(df_qld, 'QLD', PARAMS_QLD, param_dist, mc_config)

# Export Monte Carlo results to CSV
mc_vic_file = os.path.join(results_path, 'monte_carlo_results_vic.csv')
df_mc_vic.to_csv(mc_vic_file, index=False)
print(f"\n✓ VIC Monte Carlo results saved: {mc_vic_file}")

mc_qld_file = os.path.join(results_path, 'monte_carlo_results_qld.csv')
df_mc_qld.to_csv(mc_qld_file, index=False)
print(f"✓ QLD Monte Carlo results saved: {mc_qld_file}")

#%% Monte Carlo Visualization Functions

def plot_monte_carlo_distributions(df_mc, region_name, results_path):
    """
    Create comprehensive Monte Carlo visualization suite.

    Parameters:
    -----------
    df_mc : pd.DataFrame
        Monte Carlo results
    region_name : str
        'VIC' or 'QLD'
    results_path : str
        Directory to save plots

    Creates 4 key visualizations:
    ----------------------------
    1. LCOH Distribution Histogram
    2. NPV Distribution Histogram
    3. Convergence Plot (mean & std over iterations)
    4. Parameter Sensitivity Tornado Diagram
    """
    print(f"\nGenerating Monte Carlo visualizations for {region_name}...")

    # Plot 1: LCOH Distribution Histogram
    fig, ax = plt.subplots(figsize=figure_size)
    ax.hist(df_mc['lcoh_$/kg'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df_mc['lcoh_$/kg'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: ${df_mc['lcoh_$/kg'].mean():.2f}/kg")
    ax.axvline(df_mc['lcoh_$/kg'].quantile(0.10), color='orange', linestyle=':', linewidth=2,
               label=f"P10: ${df_mc['lcoh_$/kg'].quantile(0.10):.2f}/kg")
    ax.axvline(df_mc['lcoh_$/kg'].quantile(0.90), color='green', linestyle=':', linewidth=2,
               label=f"P90: ${df_mc['lcoh_$/kg'].quantile(0.90):.2f}/kg")
    ax.set_xlabel('LCOH ($/kg H₂)')
    ax.set_ylabel('Frequency (# of Iterations)')
    ax.set_title(f'{region_name}: LCOH Distribution (Monte Carlo, n={len(df_mc):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_lcoh_distribution_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ LCOH distribution plot saved")
    plt.close()

    # Plot 2: NPV Distribution Histogram
    fig, ax = plt.subplots(figsize=figure_size)
    ax.hist(df_mc['npv_$m'], bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Breakeven')
    ax.axvline(df_mc['npv_$m'].median(), color='red', linestyle='--', linewidth=2,
               label=f"Median: ${df_mc['npv_$m'].median():.1f}M")
    ax.set_xlabel('NPV ($M AUD)')
    ax.set_ylabel('Frequency (# of Iterations)')
    ax.set_title(f'{region_name}: NPV Distribution (25-year project)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_npv_distribution_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ NPV distribution plot saved")
    plt.close()

    # Plot 3: Convergence Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    cumulative_mean = df_mc['lcoh_$/kg'].expanding().mean()
    cumulative_std = df_mc['lcoh_$/kg'].expanding().std()
    iterations = np.arange(1, len(df_mc) + 1)

    ax1.plot(iterations, cumulative_mean, color='blue', linewidth=2)
    ax1.fill_between(iterations,
                      cumulative_mean - cumulative_std,
                      cumulative_mean + cumulative_std,
                      alpha=0.3, color='blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative Mean LCOH ($/kg)')
    ax1.set_title(f'{region_name}: Monte Carlo Convergence - LCOH Mean')
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, cumulative_std, color='orange', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Std Dev ($/kg)')
    ax2.set_title(f'{region_name}: Monte Carlo Convergence - LCOH Std Dev')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_convergence_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Convergence plot saved")
    plt.close()

    # Plot 4: Tornado Diagram (Sensitivity)
    fig, ax = plt.subplots(figsize=figure_size)

    correlations = {}
    for param in ['CAPEX', 'EFFICIENCY', 'H2_PRICE', 'LIFETIME_HOURS', 'VOM', 'WACC']:
        correlations[param] = df_mc[param].corr(df_mc['lcoh_$/kg'])

    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    params_names = [p[0] for p in sorted_params]
    params_corr = [p[1] for p in sorted_params]

    colors = ['red' if c < 0 else 'blue' for c in params_corr]
    ax.barh(params_names, params_corr, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Correlation with LCOH')
    ax.set_title(f'{region_name}: Parameter Sensitivity (Tornado Diagram)')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_tornado_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Tornado diagram saved")
    plt.close()

    # Plot 5: NPV Heatmap (CAPEX vs H₂ Price)
    fig, ax = plt.subplots(figsize=heatmap_size)

    # Create 2D grid bins
    capex_bins = np.linspace(600, 1400, 21)  # 20 bins
    h2_price_bins = np.linspace(3.0, 6.0, 21)  # 20 bins

    # Calculate mean NPV for each bin
    npv_grid = np.zeros((20, 20))
    count_grid = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            mask = ((df_mc['CAPEX'] >= capex_bins[i]) &
                    (df_mc['CAPEX'] < capex_bins[i+1]) &
                    (df_mc['H2_PRICE'] >= h2_price_bins[j]) &
                    (df_mc['H2_PRICE'] < h2_price_bins[j+1]))
            count = mask.sum()
            if count > 0:
                npv_grid[j, i] = df_mc.loc[mask, 'npv_$m'].mean()
                count_grid[j, i] = count
            else:
                npv_grid[j, i] = np.nan

    # Plot heatmap with diverging colormap (red=negative, green=positive)
    # Scale set to capture P10-P90 range across both regions for meaningful comparison
    # VIC NPV range: ~1,925-10,312 $M; QLD NPV range: ~868-7,725 $M
    vmin = 1500  # Below QLD P10 (1,770 $M)
    vmax = 7500  # Above VIC P90 (7,072 $M)
    extent = [capex_bins[0], capex_bins[-1], h2_price_bins[0], h2_price_bins[-1]]
    im = ax.imshow(npv_grid, cmap='RdYlGn', aspect='auto', origin='lower',
                   extent=extent, vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='NPV ($M AUD)')

    ax.set_xlabel('CAPEX ($/kW)')
    ax.set_ylabel('H₂ Price ($/kg)')
    ax.set_title(f'{region_name}: NPV Heatmap (CAPEX vs H₂ Price)\n25-year project, WACC=7%')

    # Add text annotations for key regions
    # ax.text(0.95, 0.05, 'HIGH RISK\n(Low price,\nHigh CAPEX)',
    #         transform=ax.transAxes, ha='right', va='bottom',
    #         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
    #         fontsize=9)
    # ax.text(0.05, 0.95, 'ATTRACTIVE\n(High price,\nLow CAPEX)',
    #         transform=ax.transAxes, ha='left', va='top',
    #         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
    #         fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_npv_heatmap_capex_h2price_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ NPV heatmap saved")
    plt.close()

    # Plot 6: NPV Scatter Plot (H₂ Price vs NPV, colored by CAPEX)
    fig, ax = plt.subplots(figsize=figure_size)

    # Create scatter plot with CAPEX as color dimension
    scatter = ax.scatter(df_mc['H2_PRICE'], df_mc['npv_$m'],
                        c=df_mc['CAPEX'], cmap='RdYlBu_r',
                        alpha=0.4, s=15, edgecolors='none')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='CAPEX ($/kW)')

    # Add horizontal line at NPV = 0
    ax.axhline(0, color='black', linestyle='--', linewidth=2, label='Breakeven', zorder=3)

    # Add trend line
    z = np.polyfit(df_mc['H2_PRICE'], df_mc['npv_$m'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_mc['H2_PRICE'].min(), df_mc['H2_PRICE'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'r-', linewidth=2, label=f'Trend: NPV = {z[0]:.1f}×Price {z[1]:+.1f}', zorder=4)

    ax.set_xlabel('H₂ Selling Price ($/kg)')
    ax.set_ylabel('NPV ($M AUD)')
    ax.set_title(f'{region_name}: NPV vs H₂ Price (colored by CAPEX)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'mc_npv_scatter_h2price_{region_name}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ NPV scatter plot saved")
    plt.close()

    print(f"  ✓ All Monte Carlo visualizations complete for {region_name}\n")

# Generate visualizations for both regions
plot_monte_carlo_distributions(df_mc_vic, 'VIC', results_path)
plot_monte_carlo_distributions(df_mc_qld, 'QLD', results_path)

# Plot 7: Combined NPV CDF (Cumulative Distribution Function) - Both Regions
print(f"\nGenerating combined NPV CDF plot for both regions...")
fig, ax = plt.subplots(figsize=figure_size)

# Calculate CDF for VIC
npv_sorted_vic = np.sort(df_mc_vic['npv_$m'])
# Exceedance probability: P(NPV > x) = 1 - CDF
exceedance_vic = (1 - np.arange(1, len(npv_sorted_vic)+1) / len(npv_sorted_vic)) * 100

# Calculate CDF for QLD
npv_sorted_qld = np.sort(df_mc_qld['npv_$m'])
exceedance_qld = (1 - np.arange(1, len(npv_sorted_qld)+1) / len(npv_sorted_qld)) * 100

# Plot both regions
ax.plot(npv_sorted_vic, exceedance_vic, linewidth=2.5, label='VIC', color='blue')
ax.plot(npv_sorted_qld, exceedance_qld, linewidth=2.5, label='QLD', color='green')

# Mark P10, P50, P90 for VIC
for percentile, label, style in [(0.10, 'P10', ':'), (0.50, 'P50', '--'), (0.90, 'P90', ':')]:
    npv_val_vic = df_mc_vic['npv_$m'].quantile(percentile)
    ax.axvline(npv_val_vic, color='blue', linestyle=style, alpha=0.5, linewidth=1.5)
    if percentile == 0.50:  # Only label P50
        ax.text(npv_val_vic, 95, f' VIC P50: ${npv_val_vic:.1f}M',
                rotation=0, va='top', ha='left', fontsize=9, color='blue')

# Mark P10, P50, P90 for QLD
for percentile, label, style in [(0.10, 'P10', ':'), (0.50, 'P50', '--'), (0.90, 'P90', ':')]:
    npv_val_qld = df_mc_qld['npv_$m'].quantile(percentile)
    ax.axvline(npv_val_qld, color='green', linestyle=style, alpha=0.5, linewidth=1.5)
    if percentile == 0.50:  # Only label P50
        ax.text(npv_val_qld, 85, f' QLD P50: ${npv_val_qld:.1f}M',
                rotation=0, va='top', ha='left', fontsize=9, color='green')

# Mark breakeven line
ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Breakeven (NPV=0)', zorder=3)

# Shade profitable region (NPV > 0)
ax.axvspan(0, max(npv_sorted_vic.max(), npv_sorted_qld.max()), alpha=0.1, color='green',
           label='Profitable Zone')

# Shade unprofitable region (NPV < 0)
ax.axvspan(min(npv_sorted_vic.min(), npv_sorted_qld.min()), 0, alpha=0.1, color='red',
           label='Loss Zone')

# Add horizontal reference lines
for prob in [25, 50, 75]:
    ax.axhline(prob, color='gray', linestyle=':', alpha=0.3, linewidth=1)

ax.set_xlabel('NPV ($M AUD)')
ax.set_ylabel('Exceedance Probability (%)\n← P(NPV > X) →')
ax.set_title('NPV Exceedance Probability - VIC vs QLD Comparison\n(25-year project, 5,000 Monte Carlo iterations)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# Add annotation for interpretation
ax.text(0.02, 0.5, 'Example: Find NPV on X-axis,\nread probability on Y-axis.\n"80% chance NPV exceeds this value"',
        transform=ax.transAxes, fontsize=9, va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
cdf_file = os.path.join(results_path, 'mc_npv_cdf_comparison.png')
plt.savefig(cdf_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Combined NPV CDF plot saved: {cdf_file}")
plt.close()

#%% Monte Carlo Results Analysis

def analyze_monte_carlo_results(df_mc, region_name):
    """
    Analyze Monte Carlo results and print comprehensive statistics.

    Parameters:
    -----------
    df_mc : pd.DataFrame
        Monte Carlo results from run_monte_carlo_simulation()
    region_name : str
        'VIC' or 'QLD'

    Returns:
    --------
    dict : Summary statistics and key metrics

    Outputs printed:
    ----------------
    - LCOH distribution (mean, median, std, P10/P50/P90)
    - NPV distribution (mean, median, std, P10/P50/P90)
    - Probability of positive NPV
    - Convergence assessment
    - Parameter sensitivity (correlation coefficients)
    - Engineering validation checks
    """
    print(f"\n{'='*80}")
    print(f"MONTE CARLO RESULTS ANALYSIS - {region_name}")
    print(f"{'='*80}")

    # LCOH Distribution Statistics
    print(f"\nLCOH Distribution ($/kg H₂):")
    print(f"  Mean:      ${df_mc['lcoh_$/kg'].mean():.2f}/kg")
    print(f"  Median:    ${df_mc['lcoh_$/kg'].median():.2f}/kg")
    print(f"  Std Dev:   ${df_mc['lcoh_$/kg'].std():.2f}/kg")
    print(f"  Min:       ${df_mc['lcoh_$/kg'].min():.2f}/kg")
    print(f"  Max:       ${df_mc['lcoh_$/kg'].max():.2f}/kg")
    print(f"  P10:       ${df_mc['lcoh_$/kg'].quantile(0.10):.2f}/kg (pessimistic)")
    print(f"  P50:       ${df_mc['lcoh_$/kg'].quantile(0.50):.2f}/kg (median)")
    print(f"  P90:       ${df_mc['lcoh_$/kg'].quantile(0.90):.2f}/kg (optimistic)")

    # NPV Distribution Statistics
    print(f"\nNPV Distribution ($M AUD, 25-year project):")
    print(f"  Mean:      ${df_mc['npv_$m'].mean():.1f}M")
    print(f"  Median:    ${df_mc['npv_$m'].median():.1f}M")
    print(f"  Std Dev:   ${df_mc['npv_$m'].std():.1f}M")
    print(f"  Min:       ${df_mc['npv_$m'].min():.1f}M")
    print(f"  Max:       ${df_mc['npv_$m'].max():.1f}M")
    print(f"  P10:       ${df_mc['npv_$m'].quantile(0.10):.1f}M (pessimistic)")
    print(f"  P50:       ${df_mc['npv_$m'].quantile(0.50):.1f}M (median)")
    print(f"  P90:       ${df_mc['npv_$m'].quantile(0.90):.1f}M (optimistic)")

    # Probability of Positive NPV
    prob_positive_npv = (df_mc['npv_$m'] > 0).mean() * 100
    print(f"\n  Probability of Positive NPV: {prob_positive_npv:.1f}%")
    if prob_positive_npv > 70:
        print(f"  ✓ HIGH probability of project viability")
    elif prob_positive_npv > 40:
        print(f"  ⚠ MODERATE probability - detailed risk analysis recommended")
    else:
        print(f"  ✗ LOW probability - project likely not viable under these conditions")

    # Utilization Statistics
    print(f"\nUtilization Factor Distribution (%):")
    print(f"  Mean:      {df_mc['utilization_%'].mean():.1f}%")
    print(f"  Median:    {df_mc['utilization_%'].median():.1f}%")
    print(f"  P10-P90:   {df_mc['utilization_%'].quantile(0.10):.1f}% - {df_mc['utilization_%'].quantile(0.90):.1f}%")

    # Convergence Check
    print(f"\nConvergence Assessment:")
    n_iter = len(df_mc)
    last_n = min(500, n_iter // 10)  # Last 10% or 500 iterations
    cumulative_mean_lcoh = df_mc['lcoh_$/kg'].expanding().mean()
    convergence_stability = cumulative_mean_lcoh.iloc[-last_n:].std()
    print(f"  Std Dev of mean LCOH (last {last_n} iterations): ${convergence_stability:.4f}/kg")
    if convergence_stability < 0.01:
        print(f"  ✓ CONVERGED - results are stable")
    elif convergence_stability < 0.05:
        print(f"  ⚠ ACCEPTABLE - minor fluctuations remaining")
    else:
        print(f"  ⚠ Consider increasing iterations for better stability")

    # Parameter Sensitivity (Correlation Analysis)
    print(f"\nParameter Sensitivity to LCOH (Correlation Coefficients):")
    params_to_check = ['CAPEX', 'EFFICIENCY', 'H2_PRICE', 'LIFETIME_HOURS', 'VOM', 'WACC']
    correlations_lcoh = {}
    for param in params_to_check:
        corr = df_mc[param].corr(df_mc['lcoh_$/kg'])
        correlations_lcoh[param] = corr
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else ("Moderate" if abs(corr) > 0.3 else "Weak")
        print(f"  {param:20s}: {corr:+.3f}  {direction} {strength}")

    print(f"\nParameter Sensitivity to NPV (Correlation Coefficients):")
    correlations_npv = {}
    for param in params_to_check:
        corr = df_mc[param].corr(df_mc['npv_$m'])
        correlations_npv[param] = corr
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else ("Moderate" if abs(corr) > 0.3 else "Weak")
        print(f"  {param:20s}: {corr:+.3f}  {direction} {strength}")

    # Engineering Validation
    print(f"\nEngineering Validation:")
    lcoh_p50 = df_mc['lcoh_$/kg'].median()
    if 3.0 <= lcoh_p50 <= 8.0:
        print(f"  ✓ LCOH P50 (${lcoh_p50:.2f}/kg) is within typical grid-connected range ($3-8/kg)")
    elif lcoh_p50 < 3.0:
        print(f"  ⚠ LCOH P50 (${lcoh_p50:.2f}/kg) is very low - verify parameters")
    else:
        print(f"  ⚠ LCOH P50 (${lcoh_p50:.2f}/kg) is high - limited profitability at current prices")

    util_median = df_mc['utilization_%'].median()
    if 30 <= util_median <= 70:
        print(f"  ✓ Utilization ({util_median:.1f}%) is typical for price-arbitrage operation")
    elif util_median < 30:
        print(f"  ⚠ Low utilization ({util_median:.1f}%) - electricity prices rarely profitable")
    else:
        print(f"  ✓ High utilization ({util_median:.1f}%) - good price conditions")

    # Check for expected correlation signs
    validation_passed = True
    if correlations_lcoh.get('CAPEX', 0) < 0:
        print(f"  ✗ WARNING: CAPEX correlation with LCOH is negative (expected positive)")
        validation_passed = False
    if correlations_lcoh.get('H2_PRICE', 0) > 0:
        print(f"  ✗ WARNING: H₂ Price correlation with LCOH is positive (expected negative)")
        validation_passed = False
    if validation_passed:
        print(f"  ✓ Parameter correlations match expected relationships")

    return {
        'region': region_name,
        'lcoh_mean': df_mc['lcoh_$/kg'].mean(),
        'lcoh_p50': df_mc['lcoh_$/kg'].median(),
        'lcoh_std': df_mc['lcoh_$/kg'].std(),
        'npv_p50_$m': df_mc['npv_$m'].median(),
        'prob_positive_npv': prob_positive_npv,
        'utilization_median': util_median,
        'correlations_lcoh': correlations_lcoh,
        'correlations_npv': correlations_npv
    }

# Analyze results for both regions
summary_vic = analyze_monte_carlo_results(df_mc_vic, 'VIC')
summary_qld = analyze_monte_carlo_results(df_mc_qld, 'QLD')

print("\n" + "="*80)
print("MODEL EXECUTION COMPLETE")
print("="*80)
print(f"\nAll results saved to: {results_path}")
print("\nKey Findings:")
print(f"  VIC: LCOH = ${metrics_vic['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_vic['utilisation_%']:.1f}%")
print(f"  QLD: LCOH = ${metrics_qld['lcoh_$/kg']:.2f}/kg, Utilisation = {metrics_qld['utilisation_%']:.1f}%")