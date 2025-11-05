# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced fork of NREL's Hybrid Optimization and Performance Platform (HOPP) with advanced multi-wind farm simulation and optimization capabilities. The codebase combines renewable energy system modeling (wind, solar, battery, CSP) with sophisticated optimization algorithms for hybrid plant design and operation.

Key enhancements over base HOPP:
- Multi-wind farm support via `MultiWindPlant` class allowing simulation of multiple geographically distributed wind farms
- Advanced optimization algorithms (Nelder-Mead, Differential Evolution, Genetic Algorithms) in `hopp/tools/optimization/`
- Parallel processing optimization with LRU caching
- Custom system optimizer in `hopp/tools/optimization/system_optimizer.py`

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
pytest tests/hopp

# Run specific test module
pytest tests/hopp/test_hybrid.py

# Run tests matching pattern
pytest tests/hopp/test_layout.py -k multi_wind

# Run with verbose output
pytest tests/hopp -v

# Run single test function
pytest tests/hopp/test_battery_dispatch.py::test_battery_dispatch
```

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
python test2.py             # Dual-location wind farm optimization

# Run Jupyter notebooks
jupyter lab examples/       # Opens example notebooks in browser
```

## Architecture Overview

### Core Simulation Architecture

**HybridSimulation** (`hopp/simulation/hybrid_simulation.py`) is the main orchestrator:
- Manages multiple power sources (PV, Wind, Battery, CSP, Wave, Tidal, Grid)
- Coordinates simulation sequence: `simulate_power` → `calculate_installed_cost` → `calculate_financials` → `simulate_financials`
- Uses PySAM models under the hood for energy calculations
- Configuration driven via YAML files (see `examples/inputs/*.yaml`)

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

**SystemOptimizer** (`hopp/tools/optimization/system_optimizer.py`):
- High-level interface for optimizing hybrid system configurations
- Supports three optimization methods:
  - `optimize_system()` - Nelder-Mead simplex method
  - `optimize_system_de()` - Differential Evolution
  - `optimize_system_ga()` - Genetic Algorithm
- Optimizes system parameters: PV size, wind turbines, battery capacity (kWh/kW), grid interconnect
- Integrates with `LoadAnalyzer` for demand-met targets and flexible load modeling
- Uses `EconomicCalculator` for LCOE and financial metrics

**Optimization Framework** (`hopp/tools/optimization/optimizer/`):
- Multiple optimizer implementations: CEM, DCEM, CMA-ES, GA, SPSA, and variants (IDCEM, IPDCEM, IWDCEM, KFDCEM)
- Ask-tell interface for parallel optimization (`ask_tell_optimizer.py`)
- Driver classes (`hopp/tools/optimization/driver/`) for serial and parallel execution

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

## Testing Considerations

- Mock API responses are in `tests/hopp/api_responses/` - use for tests requiring external data
- Input fixtures (YAML configs, resource CSVs) live in `tests/hopp/inputs/`
- Tests should clean up any output files created
- Use `pytest.mark.dependency` for tests with prerequisites
- Use `pytest.mark.parametrize` for testing multiple configurations
- Long-running optimization tests should have shorter variants or be marked for optional runs

## Dependencies and Solver Requirements

- **PySAM** (>=6.0.0): NREL's System Advisor Model Python wrapper, core simulation engine
- **Pyomo** (>=6.1.2): Optimization modeling, used for dispatch
- **FLORIS** (>=4.0): Wind farm wake modeling (optional, for advanced wind modeling)
- **CBC/GLPK**: MILP solvers for dispatch optimization (must be conda-installed, not pip)
- Python 3.10-3.11 only (3.12+ not supported due to PySAM constraints)

## File Organization

- `hopp/simulation/technologies/` - Power source implementations
- `hopp/tools/optimization/` - Optimization algorithms and system optimizer
- `hopp/tools/analysis/` - Economic calculators, BOS costing
- `hopp/tools/layout/` - Flicker analysis, layout optimization
- `hopp/tools/resource/` - Resource data fetching and processing
- `examples/` - Jupyter notebooks demonstrating features
- `examples/inputs/` - YAML configuration files
- `tests/hopp/` - Test suite
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

- **Windows CBC**: Manual installation required on Windows (conda install may not work)
- **Cache invalidation**: LRU cache in optimization scripts doesn't track YAML config changes
- **API rate limits**: NREL API has rate limits, cache resource files locally
- **Memory usage**: Large optimization runs can consume significant memory, especially with parallel evaluation
- **YAML includes**: Relative paths in `!include` directives are relative to the including file
- **Battery rounding**: SystemOptimizer rounds battery capacity to nearest MWh, which may affect optimal solutions
- **Dispatch convergence**: Some configurations may cause dispatch optimization to be infeasible, check solver status

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
- Reduce time resolution for initial optimization sweeps (use representative days)
- Profile optimization scripts to identify bottlenecks (often in PySAM calls)
- Consider surrogate models for nested optimization problems
