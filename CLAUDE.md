# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced fork of NREL's Hybrid Optimization and Performance Platform (HOPP) with advanced multi-wind farm simulation, optimization capabilities, and hydrogen techno-economic modeling. The codebase combines renewable energy system modeling (wind, solar, battery, CSP) with sophisticated optimization algorithms for hybrid plant design and operation.

Key enhancements over base HOPP:
- Multi-wind farm support via `MultiWindPlant` class allowing simulation of multiple geographically distributed wind farms
- Advanced optimization algorithms (Nelder-Mead, Differential Evolution, Genetic Algorithms) in `hopp/optimization/`
- Parallel processing optimization with LRU caching
- Custom system optimizer in `hopp/optimization/system_optimizer.py`
- Standalone optimization test scripts (`test.py`, `test2.py`) demonstrating real-world optimization scenarios
- **Hydrogen techno-economic modeling** in `hydrogen/` directory for grid-connected electrolyser analysis with AEMO pricing data

## Development Commands

### Environment Setup
```bash
# Create conda environment (recommended)
conda create --name hopp python=3.11 -y
conda activate hopp

# Install solver dependencies (required for dispatch optimization)
conda install -y -c conda-forge coin-or-cbc=2.10.8 glpk

# Install package in development mode
pip install -e ".[develop]"

# For examples (Jupyter notebooks)
pip install -e ".[examples]"
```

### Testing
```bash
# Run full test suite
# Note: The pyproject.toml specifies test paths as test/hopp/ and test/greenheart/
pytest test/hopp

# Run specific test module
pytest test/hopp/test_hybrid.py

# Run tests matching pattern
pytest test/hopp/test_layout.py -k multi_wind

# Run with verbose output
pytest test/hopp -v

# Run single test function
pytest test/hopp/test_battery_dispatch.py::test_battery_dispatch
```

**Important**: This project uses `test/hopp/` (not `tests/hopp/`) as specified in pyproject.toml. If the test directory doesn't exist yet, tests may need to be created.

### Documentation
```bash
# Build documentation (Jupyter Book format)
jupyter-book build docs/

# Documentation is located in docs/_build/html/
```

### Running Examples
```bash
# Run standalone optimization test scripts
python test.py              # Comprehensive hybrid optimization with multi-wind ratios
                           # - Optimizes for multiple demand-met targets (75%, 80%, 85%, etc.)
                           # - Tests various turbine distribution ratios between sub-farms
                           # - Outputs results to Excel with organized worksheets
                           # - Uses LRU caching for performance

python test2.py             # Dual-location wind farm demonstration
                           # - Downloads wind/solar resource data for separate locations
                           # - Demonstrates MultiWindPlant configuration
                           # - Includes visualization of generation profiles
                           # - Shows battery dispatch optimization

# Run Jupyter notebooks
jupyter lab examples/       # Opens example notebooks in browser
```

**Security Note**: Both test scripts contain hardcoded NREL API keys. For production use, always use environment variables or .env files instead:
```bash
export NREL_API_KEY=your_key_here
export NREL_API_EMAIL=your.email@example.com
```

## Architecture Overview

### Core Simulation Architecture

**HybridSimulation** (`hopp/simulation/hybrid_simulation.py`) is the main orchestrator:
- Manages multiple power sources (PV, Wind, Battery, CSP, Wave, Tidal, Grid)
- Coordinates simulation sequence: `simulate_power` → `calculate_installed_cost` → `calculate_financials` → `simulate_financials`
- Uses PySAM models under the hood for energy calculations
- Configuration driven via YAML files (see `examples/inputs/*.yaml`)

**HoppInterface** (`hopp/simulation/hopp_interface.py`):
- High-level wrapper around HybridSimulation for easy initialization from YAML
- Primary entry point for running simulations
- Usage pattern:
  ```python
  from hopp.simulation import HoppInterface

  hi = HoppInterface("config.yaml")
  hi.simulate(project_life=25)  # Run 25-year simulation
  hybrid_plant = hi.system       # Access HybridSimulation instance
  ```
- Automatically loads configuration, initializes technologies, and sets up site information

**Technology Hierarchy**:
```
PowerSource (base class)
├── PVPlant / DetailedPVPlant
├── WindPlant
│   └── MultiWindPlant (enhanced for multi-location simulation)
├── Battery / BatteryStateless
├── TowerPlant / TroughPlant (CSP)
├── MHKWavePlant / MHKTidalPlant
└── Grid
```

Each technology implements:
- `simulate_power()` - generates hourly generation profiles
- `calculate_installed_cost()` - computes CAPEX
- Financial methods for LCOE/NPV calculations

### Multi-Wind Farm Architecture

**MultiWindPlant** (`hopp/simulation/technologies/wind/multi_wind_plant.py`):
- Subclass of `PowerSource` (parallel to `WindPlant`, not inheriting from it)
- Manages multiple sub-wind farms with independent configurations
- Each sub-farm can have different: resource files, hub heights, turbine ratings, number of turbines, locations
- Aggregates generation across all sub-farms
- Configuration via `MultiWindConfig` class with lists of parameters for each sub-farm

Key design: `HybridSimulation` calls `simulate_power()` on each power source sequentially, so `MultiWindPlant` internally manages its sub-farm array but presents a single interface.

### Optimization Architecture

**SystemOptimizer** (`hopp/optimization/system_optimizer.py`):
- High-level interface for optimizing hybrid system configurations
- Supports three optimization methods:
  - `optimize_system()` - Nelder-Mead simplex method (scipy.optimize.minimize)
  - `optimize_system_de()` - Differential Evolution
  - `optimize_system_ga()` - Genetic Algorithm
- Optimizes system parameters: PV size, wind turbines, battery capacity (kWh/kW), grid interconnect
- Integrates with `LoadAnalyzer` for demand-met targets and flexible load modeling
- Uses `EconomicCalculator` for LCOE and financial metrics
- Key methods:
  - `set_turbine_ratio(ratio)` - Sets turbine distribution ratio for multi-wind farms
  - `objective_function(x)` - Calculates LCOE and system metrics
  - `penalized_objective(x)` - Adds penalty terms for constraint violations
  - `round_battery_capacity(capacity)` - Rounds to nearest MWh (1000 kWh increments)

**Optimization Framework** (`hopp/optimization/optimizer/`):
- Multiple optimizer implementations: CEM, DCEM, CMA-ES, GA, SPSA, and variants (IDCEM, IPDCEM, IWDCEM, KFDCEM)
- Ask-tell interface for parallel optimization (`ask_tell_optimizer.py`)
- Driver classes (`hopp/optimization/driver/`) for serial and parallel execution

### Dispatch Optimization

**HybridDispatch** (`hopp/simulation/technologies/dispatch/hybrid_dispatch.py`):
- Uses Pyomo for optimization-based dispatch of battery and grid
- Formulated as mixed-integer linear program (MILP)
- Solves for optimal battery charge/discharge and grid import/export given:
  - Generation profiles (wind, solar)
  - Load profiles
  - Electricity prices (time-of-use rates)
  - Battery constraints
- Requires CBC or GLPK solver (installed via conda)

**Key files**:
- `hybrid_dispatch_builder_solver.py` - constructs and solves dispatch problem
- `grid_dispatch.py` - grid-specific dispatch logic
- `dispatch_problem_state.py` - maintains state between time steps

### Configuration System

**ConfigManager** (`hopp/utilities/config_manager.py`):
- Loads YAML configurations with `!include` support (via `pyyaml-include`)
- Validates configurations against expected schemas
- Example structure:
  ```yaml
  site: !include site_definition.yaml
  technologies:
    pv:
      system_capacity_kw: 50000
    wind:
      num_turbines: 10
      turbine_rating_kw: 5000
  ```

### Resource Data

**ResourceDataManager** (`hopp/simulation/resource_files/resource_data_manager.py`):
- Downloads solar and wind resource data from NREL APIs
- Requires `NREL_API_KEY` and `NREL_API_EMAIL` environment variables
- Caches data locally in `hopp/simulation/resource_files/`
- Supports custom resource files (CSV format with specific column structure)
- Methods:
  - `download_solar_data(latitude, longitude, year)` - Downloads NSRDB solar resource data
  - `download_wind_data(latitude, longitude, start_date, end_date)` - Downloads Wind Toolkit data

### Load Analysis

**LoadAnalyzer** (`hopp/optimization/load_analyzer.py`):
- Analyzes system performance against load requirements
- Supports flexible load modeling with configurable load reduction
- Integrates with `SystemOptimizer` for demand-met target optimization
- Key parameters:
  - `enable_flexible_load` - Allows load to be reduced during optimization
  - `max_load_reduction_percentage` - Maximum allowable load reduction (e.g., 0.2 = 20%)
- Methods:
  - `calculate_performance_metrics(df, project_lifetime)` - Calculates demand-met percentage, load served, etc.
  - Returns metrics including: Total Load Served, Demand Met %, Excess Generation, etc.

### Hydrogen Techno-Economic Modeling

**Separate from HOPP Core** - The `hydrogen/` directory contains standalone analysis scripts:

**Architecture**:
- **Not integrated with HOPP simulation engine** - runs independently
- Uses pandas for data manipulation and analysis
- Reads AEMO (Australian Energy Market Operator) spot price data from CSV files
- Calculates electrolyser economics based on grid electricity prices

**Key Components** (`hydrogen/code/`):
- Region-specific parameter definitions (VIC, QLD) including:
  - CAPEX ($/kW)
  - Lifetime (hours)
  - Efficiency (kWh/kg H₂)
  - Variable O&M ($/MWh)
  - Hydrogen selling price ($/kg)
- Price threshold optimization for electrolyser dispatch
- LCOH calculation with component breakdown (electricity, depreciation, VOM)
- Utilization analysis and sensitivity studies

**Data Requirements**:
- AEMO RRP data in `hydrogen/data/AEMO RRP data/[REGION] [YEAR] data/`
- CSV format with columns: `SETTLEMENTDATE`, `RRP` (Regional Reference Price in $/MWh)
- Files named like `PRICE_AND_DEMAND_[YEAR]**_[REGION].csv`

**Output**:
- Results saved to `hydrogen/results/`
- Includes utilization metrics, cost breakdowns, and validation checks
- Matplotlib visualizations of price distributions and operating strategies

## Important Design Patterns

### Technology Simulation Flow
All power sources follow this sequence (called by `HybridSimulation`):
1. Initialize with site info and configuration
2. Call `simulate_power()` to generate hourly generation array
3. Call `calculate_installed_cost()` for CAPEX
4. Call `calculate_financials()` with financial parameters
5. Optionally run dispatch optimization
6. Extract results via technology-specific output attributes

### Multi-Wind Sub-Farm Configuration
When working with `MultiWindPlant`, sub-farm parameters are lists:
```python
config = MultiWindConfig(
    resource_files=["site1.csv", "site2.csv"],
    sub_num_turbines=[10, 15],
    sub_turbine_ratings=[5000, 6000],  # kW
    sub_hub_heights=[90, 100],  # meters
    sub_lats=[39.7, 40.1],
    sub_lons=[-105.2, -104.8]
)
```
Missing parameters inherit from main config. Validation ensures list lengths match.

### Optimization Parameter Encoding
Standard optimization vector format (used in `SystemOptimizer`):
```python
x = [pv_size_kw, num_turbines, battery_kwh, battery_kw, grid_interconnect_kw]
```
Battery capacity is rounded to nearest MWh (1000 kWh increments). All parameters cast to integers except battery capacity.

### Data Caching
- Optimization scripts use `functools.lru_cache` on expensive simulation calls
- Must be careful with cache invalidation when parameters change
- Parallel optimization benefits significantly from caching repeated evaluations

## Common Workflows

### Optimizing Multi-Wind Farm Configurations
To optimize hybrid systems with multiple wind farms and varying turbine distributions:

```python
from hopp.optimization.system_optimizer import SystemOptimizer
from hopp.tools.analysis import EconomicCalculator

# Initialize optimizer
optimizer = SystemOptimizer(
    yaml_file_path="config.yaml",
    economic_calculator=EconomicCalculator(project_lifetime=25),
    enable_flexible_load=False
)

# Set turbine distribution ratio between sub-farms
# e.g., 0.4 means 40% of total turbines in first farm, 60% in second
optimizer.set_turbine_ratio(0.4)

# Define optimization bounds: [pv_kw, num_turbines, battery_kwh, battery_kw, grid_kw]
bounds = [(0, 100000), (0, 100), (0, 50000), (0, 10000), (0, 150000)]

# Run optimization
result = optimizer.optimize_system(bounds, initial_conditions=[[10000, 20, 5000, 1000, 50000]])

# Access results
print(f"Optimal PV: {result['PV Capacity (kW)']} kW")
print(f"Optimal Wind: {result['Wind Turbine Capacity (kW)']} kW")
print(f"LCOE: {result['LCOE ($/kWh)']} $/kWh")
```

**Key Implementation Details**:
- The optimizer updates `config['technologies']['wind']['sub_num_turbines'][0]` based on the ratio
- Battery capacity is rounded to nearest MWh (1000 kWh) increments
- Configuration is saved to YAML file between iterations
- LRU caching can significantly improve performance for repeated evaluations

### Adding a New Technology
1. Create class in `hopp/simulation/technologies/<tech_name>/`
2. Inherit from `PowerSource` base class
3. Implement required methods: `simulate_power()`, `calculate_installed_cost()`, financial methods
4. Create corresponding `<Tech>Config` class using `attrs.define`
5. Add to `TechnologiesConfig` in `hybrid_simulation.py`
6. Update type union `PowerSourceTypes`
7. Add tests in `tests/hopp/test_<tech_name>.py`

### Modifying Optimization Objectives
When changing what the optimizer minimizes:
1. Update `objective_function()` in `SystemOptimizer` or create subclass
2. Ensure penalty terms for constraint violations (e.g., demand-met target)
3. Update bounds in optimization script
4. Verify gradient behavior if using gradient-based methods
5. Test with multiple initial conditions for global optimum

### Working with Dispatch
Dispatch requires:
1. Hourly load profile (8760 array)
2. Hourly generation profiles from technologies
3. Battery model with capacity/power constraints
4. Grid pricing structure (from `utility_rate.py`)
5. Solver installed (CBC or GLPK)

To modify dispatch logic:
- Edit constraints in `hybrid_dispatch.py::dispatch_block_rule()`
- Modify objective function in same method
- Add new decision variables to Pyomo model
- Update port connections if changing power flow topology

### Working with Hydrogen Models

The hydrogen scripts are **separate from HOPP** and use a different architecture:

```python
# Example: Modifying electrolyser parameters for a new region
PARAMS_NEW_REGION = {
    'CAPEX': 1000,              # $/kW - electrolyser capital cost
    'LIFETIME_HOURS': 80000,    # hours of operation before replacement
    'EFFICIENCY': 50,           # kWh/kg H₂ (LHV basis)
    'VOM': 5.0,                # $/MWh - variable O&M
    'H2_PRICE': 5.5,           # $/kg - hydrogen selling price
    'CAPACITY_MW': 100         # MW - electrolyser rated capacity
}

# Calculate depreciation per MWh
depreciation_per_mwh = (PARAMS_NEW_REGION['CAPEX'] * PARAMS_NEW_REGION['CAPACITY_MW'] * 1000) / \
                       (PARAMS_NEW_REGION['LIFETIME_HOURS'] / 1000)
```

**Key workflow steps**:
1. Load AEMO RRP data for the region of interest
2. Define electrolyser parameters (CAPEX, efficiency, lifetime)
3. Iterate over price thresholds to find optimal dispatch strategy
4. Calculate LCOH including electricity, depreciation, and O&M costs
5. Run sensitivity analysis on key parameters
6. Export results to `hydrogen/results/` for visualization

**Important notes**:
- Hydrogen models do not use YAML configuration files
- No integration with HOPP's HybridSimulation or optimization framework
- Suitable for grid-connected electrolyser analysis with time-varying electricity prices
- For renewable-powered hydrogen (PV+Wind+Electrolyser), use HOPP's main simulation engine instead

### NREL API Key Setup
Resource downloads require API credentials:
```bash
# Method 1: Environment variables (preferred for CI/CD)
export NREL_API_KEY=your_key_here
export NREL_API_EMAIL=your.email@example.com

# Method 2: .env file in project root (for local development)
echo "NREL_API_KEY=your_key_here" > .env
echo "NREL_API_EMAIL=your.email@example.com" >> .env
```
Never commit `.env` files. They are git-ignored by default.

### Running Hydrogen Analysis Scripts
The `hydrogen/` directory contains standalone techno-economic analysis for grid-connected electrolysers:

```bash
# Navigate to hydrogen code directory
cd hydrogen/code

# Run baseline model for AEMO regions (VIC and QLD)
python "25-11-6 baseline model.py"

# Run sensitivity analysis on electrolyser parameters
python "25-11-6 baseline model sensitivity.py"
```

**Key features of hydrogen scripts**:
- Analyzes grid-connected electrolyser economics using AEMO spot pricing data
- Calculates Levelized Cost of Hydrogen (LCOH) with breakdown by electricity, depreciation, and O&M
- Region-specific parameters for Victoria (VIC) and Queensland (QLD)
- Optimizes electrolyser dispatch based on electricity price thresholds
- Outputs include utilization rates, hydrogen production volumes, and cost metrics
- Requires AEMO RRP (Regional Reference Price) CSV files in `hydrogen/data/AEMO RRP data/`

## Testing Considerations

- Mock API responses should be in `test/hopp/api_responses/` (if directory exists) - use for tests requiring external data
- Input fixtures (YAML configs, resource CSVs) should live in `test/hopp/inputs/`
- Tests should clean up any output files created
- Use `pytest.mark.dependency` for tests with prerequisites
- Use `pytest.mark.parametrize` for testing multiple configurations
- Long-running optimization tests should have shorter variants or be marked for optional runs
- **Note**: Test directory (`test/hopp/`) may not exist yet - check before running tests

## Dependencies and Solver Requirements

- **Python**: 3.10-3.11 required (3.12+ not officially supported due to PySAM constraints)
  - **Warning**: If using Python 3.13+, you may encounter compatibility issues with PySAM
  - Use conda to create environment with specific Python version: `conda create --name hopp python=3.11`
- **PySAM** (>=6.0.0): NREL's System Advisor Model Python wrapper, core simulation engine
- **Pyomo** (>=6.1.2): Optimization modeling, used for dispatch optimization
- **FLORIS** (>=4.0): Wind farm wake modeling (optional, for advanced wind modeling)
- **CBC/GLPK**: MILP solvers for dispatch optimization (must be conda-installed, not pip)
  - Install via: `conda install -y -c conda-forge coin-or-cbc=2.10.8 glpk`
- **pyyaml-include**: Enables `!include` directives in YAML configuration files
- **attrs**: Used for configuration class definitions with validation

## File Organization

- `hopp/simulation/` - Core simulation engine and hybrid orchestration
  - `hopp/simulation/technologies/` - Power source implementations (PV, wind, battery, CSP, etc.)
    - `hopp/simulation/technologies/wind/` - Wind plant implementations including `MultiWindPlant`
  - `hopp/simulation/resource_files/` - Resource data manager and cached resource files
- `hopp/optimization/` - Optimization algorithms, system optimizer, and load analyzer
  - `hopp/optimization/system_optimizer.py` - Main system optimization interface
  - `hopp/optimization/load_analyzer.py` - Load analysis with flexible demand management
  - `hopp/optimization/optimizer/` - Advanced optimization algorithms (GA, CEM, DCEM, CMA-ES, SPSA)
  - `hopp/optimization/driver/` - Serial and parallel optimization drivers
- `hopp/tools/` - Analysis and utility tools
  - `hopp/tools/analysis/` - Economic calculators, BOS costing
  - `hopp/tools/layout/` - Flicker analysis, layout optimization
  - `hopp/tools/dispatch/` - Dispatch plotting tools
- `hopp/utilities/` - Configuration management and helper utilities
- `hydrogen/` - **Hydrogen techno-economic modeling (separate from HOPP base)**
  - `hydrogen/code/` - Electrolyser analysis scripts for grid-connected hydrogen production
  - `hydrogen/data/` - AEMO Regional Reference Price (RRP) data for VIC/QLD regions
  - `hydrogen/results/` - Analysis outputs and visualizations
- `examples/` - Jupyter notebooks demonstrating features
- `examples/inputs/` - YAML configuration files
- `test.py`, `test2.py` - Standalone optimization demonstration scripts
- `test/hopp/` - Test suite (pyproject.toml specifies this path, not `tests/`)
- `output/` and `log/` - Runtime outputs (git-ignored)

## Output and Results Structure

Simulation outputs follow technology-specific patterns:
- `system.pv.generation_profile` - hourly array (kW)
- `system.wind.generation_profile` - hourly array (kW)
- `system.battery.Outputs.gen` - battery generation/consumption array
- `system.grid.generation_profile` - net grid interaction
- Financial metrics accessed via `system.<tech>.financial_model.Outputs`

When working with results, note:
- Arrays are NumPy arrays, typically length 8760 (annual hourly)
- Power values typically in kW, energy in kWh
- Financial outputs from PySAM have specific naming conventions (see PySAM docs)

## Known Issues and Gotchas

- **Python Version Compatibility**:
  - Project officially supports Python 3.10-3.11 only (per pyproject.toml: `requires-python = ">=3.10, <3.12"`)
  - **CRITICAL**: Python 3.12+ and 3.13+ are NOT supported due to PySAM compatibility issues
  - Current environment may be Python 3.13 - you MUST create a conda environment with Python 3.11 for HOPP to work:
    ```bash
    conda create --name hopp python=3.11 -y
    conda activate hopp
    ```
  - Many features will fail or produce incorrect results with Python 3.12+
- **Windows CBC**: Manual installation required on Windows (conda install may not work)
  - See: https://github.com/coin-or/Cbc for manual installation instructions
- **Cache invalidation**: LRU cache in optimization scripts doesn't track YAML config changes
  - If you modify YAML files directly, clear cache or restart Python session
- **API rate limits**: NREL API has rate limits, cache resource files locally
  - Downloaded files are cached in `hopp/simulation/resource_files/`
- **Hardcoded API Keys**: test.py and test2.py contain hardcoded API keys
  - Remove these before committing or sharing code
  - Use environment variables instead
- **Memory usage**: Large optimization runs can consume significant memory, especially with parallel evaluation
  - Monitor memory usage with large turbine counts or long simulation periods
- **YAML includes**: Relative paths in `!include` directives are relative to the including file
  - Use absolute paths or ensure correct relative path from YAML file location
- **Battery rounding**: SystemOptimizer rounds battery capacity to nearest MWh, which may affect optimal solutions
  - This is intentional to avoid overly precise capacity values
- **Dispatch convergence**: Some configurations may cause dispatch optimization to be infeasible, check solver status
  - Ensure battery capacity and grid interconnect are reasonable for the load
- **Excel Output Limitations**: test.py uses openpyxl for Excel output
  - Worksheet names are limited to 31 characters
  - Large result sets may cause Excel file size to grow significantly

## Configuration Validation

When creating new YAML configs:
- Site info must include: latitude, longitude, year, elevation
- Each technology requires minimum parameters (system size or number of units)
- Financial parameters can be omitted for generation-only studies
- Grid interconnect must be >= maximum combined output to avoid curtailment
- Resource data must match site location and year

## Performance Optimization Tips

- Use parallel optimization drivers for expensive objective functions
- Cache simulation results when evaluating similar configurations
  - Example: `from functools import lru_cache` with `@lru_cache(maxsize=None)`
- Reduce time resolution for initial optimization sweeps (use representative days)
- Profile optimization scripts to identify bottlenecks (often in PySAM calls)
- Consider surrogate models for nested optimization problems
- Use `run_in_background=True` for long-running simulations if monitoring output

## Quick Reference

### Essential Import Patterns
```python
from hopp.simulation import HoppInterface
from hopp.optimization.system_optimizer import SystemOptimizer
from hopp.optimization.load_analyzer import LoadAnalyzer
from hopp.tools.analysis import EconomicCalculator
from hopp.utilities.config_manager import ConfigManager
from hopp.simulation.resource_files import ResourceDataManager
from hopp.utilities.keys import set_developer_nrel_gov_key
```

### Common Configuration Paths
- Optimization parameters: `config['technologies'][<tech>]['<param>']`
- Site location: `config['site']['data']['lat']`, `config['site']['data']['lon']`
- Resource files: `config['site']['solar_resource_file']`, `config['site']['wind_resource_files']`
- Load profile: `config['site']['desired_schedule']`
- Grid interconnect: `config['technologies']['grid']['interconnect_kw']`

### Accessing Simulation Results
```python
hi = HoppInterface("config.yaml")
hi.simulate(25)
plant = hi.system

# Generation profiles (hourly kW arrays)
pv_gen = plant.generation_profile.pv
wind_gen = plant.generation_profile.wind
battery_gen = plant.generation_profile.battery
grid_gen = plant.generation_profile.grid

# Total generation (kWh)
total_pv = np.sum(plant.generation_profile.pv)

# Capacity factors
pv_cf = plant.pv.capacity_factor
wind_cf = plant.wind.capacity_factor

# Financial metrics
lcoe = plant.grid.financial_model.Outputs.lcoe_nom  # or via EconomicCalculator
npv = plant.grid.financial_model.Outputs.project_return_aftertax_npv
```

### Multi-Wind Farm Configuration Pattern
```yaml
technologies:
  wind:
    model_name: multi_wind  # Critical: triggers MultiWindPlant
    num_turbines: 30  # Total turbines across all farms
    sub_num_turbines: [12, 18]  # Distribution: 12 in farm 1, 18 in farm 2
    sub_hub_heights: [90, 100]  # Different hub heights per farm
    sub_lats: [39.7, 40.1]  # Separate locations
    sub_lons: [-105.2, -104.8]

site:
  wind_resource_files:  # List of resource files
    - path/to/site1_wind.csv
    - path/to/site2_wind.csv
```

### Optimization Vector Format
```python
# Standard format used by SystemOptimizer
x = [
    pv_size_kw,           # Index 0: PV system capacity
    num_turbines,         # Index 1: Total wind turbines
    battery_kwh,          # Index 2: Battery energy capacity (rounded to MWh)
    battery_kw,           # Index 3: Battery power capacity
    grid_interconnect_kw  # Index 4: Grid interconnection limit
]
```

### Troubleshooting Checklist

**HOPP Simulation Issues**:
1. **Import errors**: Check Python version (should be 3.10 or 3.11, NOT 3.12+ or 3.13)
2. **PySAM errors**: Verify you're using Python 3.10 or 3.11 (`python --version`), create new conda env if needed
3. **Simulation fails**: Verify YAML configuration has all required fields
4. **Resource download fails**: Check NREL API key environment variables
5. **Dispatch optimization fails**: Ensure CBC/GLPK solver installed via conda
6. **Results seem wrong**: Check that battery rounding isn't affecting optimization
7. **Memory issues**: Reduce simulation timesteps or use smaller turbine counts for testing

**Hydrogen Model Issues**:
1. **FileNotFoundError for RRP data**: Ensure AEMO CSV files are in `hydrogen/data/AEMO RRP data/[REGION] [YEAR] data/`
2. **Empty results**: Check that CSV files have correct column names (`SETTLEMENTDATE`, `RRP`)
3. **LCOH is infinity**: Electrolyser never operates - try higher price threshold or check data
4. **Script crashes**: Hydrogen scripts don't need HOPP dependencies, but require pandas, numpy, matplotlib
