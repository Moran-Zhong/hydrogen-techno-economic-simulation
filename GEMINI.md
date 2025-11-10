# Project Overview

This project is a techno-economic simulation for green hydrogen production. It is an enhanced version of NREL's [Hybrid Optimization and Performance Platform (HOPP)](https://github.com/NREL/HOPP), with additional features for multi-wind farm simulation and advanced optimization algorithms.

The core of the project is a Python-based simulation that assesses the economic feasibility of grid-connected hydrogen electrolyzers in Australia (specifically Victoria and Queensland). It uses real-world electricity price data from AEMO to model the operation of an electrolyzer and calculate key financial metrics like Levelized Cost of Hydrogen (LCOH), Net Present Value (NPV), and Internal Rate of Return (IRR).

The project includes:

*   A detailed techno-economic model for a hydrogen electrolyzer.
*   A sensitivity analysis to assess the impact of key parameters on project viability.
*   A Monte Carlo simulation to quantify the project's risk and return under uncertainty.
*   Data loading and processing for AEMO electricity price data.
*   Visualization of results using matplotlib.

# Building and Running

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/NREL/HOPP.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd HOPP
    ```
3.  Create and activate a virtual environment:
    ```bash
    conda create --name hopp python=3.11 -y
    conda activate hopp
    ```
4.  Install dependencies:
    ```bash
    conda install -y -c conda-forge coin-or-cbc=2.10.8 glpk
    pip install -e ".[develop]"
    ```

## Running the Simulation

The main simulation script is located at `hydrogen/code/25-11-10 Monte Carlo model.py`. To run the simulation, execute the following command from the project root:

```bash
python hydrogen/code/25-11-10 Monte Carlo model.py
```

The script will generate results in the `hydrogen/results` directory, including CSV files and plots.

## Running Tests

Tests are located in the `tests` directory and can be run using pytest:

```bash
pytest tests/hopp
```

# Development Conventions

*   **Coding Style:** The code follows standard Python conventions (PEP 8).
*   **Project Structure:** The project is organized into the following main directories:
    *   `hopp`: The core HOPP library code.
    *   `hydrogen`: The hydrogen techno-economic simulation code.
    *   `examples`: Jupyter notebooks and YAML files for usage examples.
    *   `docs`: Project documentation.
*   **Dependencies:** Project dependencies are managed in `pyproject.toml`.
*   **Data:** Input data is stored in the `hydrogen/data` directory.
*   **Results:** Simulation results are saved in the `hydrogen/results` directory.
