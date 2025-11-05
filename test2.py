#!/usr/bin/env python3
"""
Test configuration file transfer functionality for dual-location wind power model
"""

# Ensure loading the correct HOPP version

import pandas as pd
import numpy as np
import copy
from pathlib import Path
from hopp.simulation import HoppInterface
from hopp.optimization.system_optimizer import SystemOptimizer
from hopp.tools.analysis import EconomicCalculator
from hopp.utilities.config_manager import ConfigManager  # Modified import
from hopp.utilities.keys import set_developer_nrel_gov_key
from hopp.simulation.resource_files import ResourceDataManager
from hopp.tools.dispatch.plot_tools import (
    plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
)
# Import HOPP modules
from hopp.simulation import HoppInterface
import matplotlib.pyplot as plt
from hopp.optimization.system_optimizer import SystemOptimizer
from hopp.tools.analysis import EconomicCalculator
from hopp.utilities import ConfigManager
from hopp.utilities.keys import set_developer_nrel_gov_key
from hopp.simulation.resource_files import ResourceDataManager

# Set your NREL API key (required for downloading solar data)
set_developer_nrel_gov_key('uZQBIA4mfVd7kj9ofLFNY0rmEVy5qx1v2fBpTAIG')

# Initialize resource manager for downloading data
resource_manager = ResourceDataManager(
    api_key='uZQBIA4mfVd7kj9ofLFNY0rmEVy5qx1v2fBpTAIG',
    email='z5536449@ad.unsw.edu.au'
)

# Location information for solar
solar_latitude = -19.5825
solar_longitude = 146.8398

# Location information for wind (different location)

wind_latitude1 = -26.73274363843257
wind_longitude1 = 151.4722470563812

wind_latitude2 = -17.183333
wind_longitude2 = 145.366666

# Download resource data
solar_path = resource_manager.download_solar_data(
    latitude=solar_latitude,
    longitude=solar_longitude,
    year="2016"
)
wind_path1 = resource_manager.download_wind_data(
    latitude=wind_latitude1,
    longitude=wind_longitude1,
    start_date="20160101",
    end_date="20161231",
)
wind_path2 = resource_manager.download_wind_data(
    latitude=wind_latitude2,
    longitude=wind_longitude2,
    start_date="20160101",
    end_date="20161231",
)

wind_path = [wind_path1,wind_path2]

# read load data
csv_path = r'C:\Users\Chengxiang Xu\Desktop\Load data.csv'
csv_df = pd.read_csv(csv_path)
load_data = csv_df.iloc[:, 0].tolist()

# Use configuration file in workspace
yaml_file_path = str(r"D:\python\HOPP-3.2.0\examples\inputs\test-separated-simple2.yaml")
print(f"Using configuration file: {yaml_file_path}")
config = ConfigManager.load_yaml_safely(yaml_file_path)  # Use static method of class

# Update configuration with location and resource files
config['site']['data']['lat'] = solar_latitude
config['site']['data']['lon'] = solar_longitude
config['site']['solar_resource_file'] = solar_path
config['site']['wind_resource_files'] = wind_path
config['site']['desired_schedule'] = load_data

ConfigManager.save_yaml_safely(config, yaml_file_path)

# Initialize HoppInterface
hi = HoppInterface(yaml_file_path)

# Execute simulation
hi.simulate(25)

# Get simulation results
hybrid_plant = hi.system

print("Output after losses over gross output:",
      hybrid_plant.wind.value("annual_energy") / hybrid_plant.wind.value("annual_gross_energy"))

# Save results
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
revs = hybrid_plant.total_revenues
capacity = hybrid_plant.system_capacity_kw.wind
solar_capacity = hybrid_plant.pv.system_capacity_kw
lcoe = hybrid_plant.lcoe_nom
lcoe2 = hybrid_plant.lcoe_real
cost=hybrid_plant.wind.total_installed_cost
pct_timesteps_met = hybrid_plant.grid.time_load_met
grid = hybrid_plant.grid.generation_profile
wind = hybrid_plant.wind.generation_profile
solar = hybrid_plant.pv.generation_profile
battery = hybrid_plant.battery.generation_profile
load_data=hybrid_plant.site.desired_schedule
# Unmet load at each time step (unit kW or kWh, depending on configuration)
missed_load_series = hybrid_plant.grid.missed_load

# Total unmet load (cumulative for all time steps)
total_missed_load = missed_load_series.sum()
print(f"Total unmet load: {total_missed_load:.0f} kWh")

# Unmet load percentage
missed_pct = hybrid_plant.grid.missed_load_percentage
print(f"Unmet load percentage: {missed_pct :.2f}%")



    # Convert to numpy array for easier calculation
grid_array = np.array(grid)
    
    # Calculate count less than 3000
count_less_than_3000 = np.sum(grid_array < 3000)
total_timesteps = len(grid_array)
percentage = (count_less_than_3000/ total_timesteps) * 100
print(percentage)

print("\nAnnual Energies:")
print(annual_energies)

print("Net Present Values:")
print(npvs)

print("Total Revenues:")
print(revs)

print("wind capacity:")
print(capacity)
print(hybrid_plant.wind.total_installed_cost)
print("solar capacity:")
print(solar_capacity)
print(hybrid_plant.pv.total_installed_cost)
print("LCOE nom")
print(lcoe)

print("LCOE real")
print(lcoe2)
print("grid")
print(np.sum(grid))
print("load")
print(load_data[:30])
print("grid")
print(grid[:30])
print(min(grid))
print("wind")
print(wind[:30])
print("solar")
print(solar[:30])
print("battery")
print(battery[:30])
print((wind[:30]+solar[:30]+battery[:30]))



print(pct_timesteps_met)


