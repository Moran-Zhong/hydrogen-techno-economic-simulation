"""
Multi-wind farm management module, as a subclass of PowerSource, parallel to WindPlant

Note: HybridSimulation sequentially calls `simulate_power`,
`calculate_installed_cost`, `calculate_financials`, `simulate_financials`,
rather than calling the `simulate` method.
"""

from typing import List, Dict, Union, Iterable, Sequence, Optional
import copy
from attrs import define, field, Factory
import numpy as np
from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.simulation.technologies.sites.site_info import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.utilities.log import hybrid_logger as logger
from pathlib import Path
from shapely.geometry import Polygon

@define
class MultiWindConfig(WindConfig):
    """
    Multi-wind farm configuration class
    
    Args:
        sub_wind_resources: List of wind resource data for sub-wind farms
        resource_files: List of wind resource file paths
        sub_lats: List of latitudes for sub-wind farms (optional, defaults to main site latitude)
        sub_lons: List of longitudes for sub-wind farms (optional, defaults to main site longitude)
        sub_years: List of years for sub-wind farms (optional, defaults to main site year)
        sub_hub_heights: List of hub heights for sub-wind farms (optional, defaults to main site hub height)
        sub_turbine_ratings: List of turbine rated power for sub-wind farms (optional, defaults to main site turbine rated power)
        sub_num_turbines: List of number of turbines for sub-wind farms (optional, defaults to main site number of turbines)
    """
    # Override parent class model_name field, allow "multi_wind"
    model_name: str = field(
        default="multi_wind",
        converter=(str.strip, str.lower)
        # No contains validation, as this is only used to distinguish MultiWindConfig
    )
    
    sub_wind_resources: List[Dict] = field(default=Factory(list))  # List of wind resource data for sub-wind farms
    resource_files: List[Union[Path, str]] = field(default=Factory(list))  # List of wind resource file paths
    sub_lats: List[float] = field(default=Factory(list))  # List of latitudes for sub-wind farms
    sub_lons: List[float] = field(default=Factory(list))  # List of longitudes for sub-wind farms
    sub_years: List[int] = field(default=Factory(list))  # List of years for sub-wind farms
    sub_hub_heights: List[float] = field(default=Factory(list))  # List of hub heights for sub-wind farms
    sub_turbine_ratings: List[float] = field(default=Factory(list))  # List of turbine rated power for sub-wind farms
    sub_num_turbines: List[int] = field(default=Factory(list))  # List of number of turbines for sub-wind farms

    def __attrs_post_init__(self):
        """Validate configuration"""
        super().__attrs_post_init__()
        # No longer check model_name value
        
        # Validate sub-wind farm configuration
        if self.sub_wind_resources and self.resource_files:
            logger.warning("Both sub_wind_resources and resource_files provided, both will be used")
        
        # Calculate total number of sub-wind farms
        n_sub_farms = max(
            len(self.sub_wind_resources),
            len(self.resource_files)
        )
        
        if n_sub_farms == 0:
            return
            
        # Validate consistency of sub-wind farm position parameter lengths
        sub_params = [
            (self.sub_lats, "sub_lats"),
            (self.sub_lons, "sub_lons"),
            (self.sub_years, "sub_years"),
            (self.sub_hub_heights, "sub_hub_heights"),
            (self.sub_turbine_ratings, "sub_turbine_ratings"),
            (self.sub_num_turbines, "sub_num_turbines")
        ]
        
        # Check if lengths of non-empty parameter lists are consistent
        non_empty_params = [(param, name) for param, name in sub_params if param]
        if non_empty_params:
            first_len = len(non_empty_params[0][0])
            for param, name in non_empty_params[1:]:
                if len(param) != first_len:
                    raise ValueError(f"Length of {name}({len(param)}) is inconsistent with length of {non_empty_params[0][1]}({first_len}), "
                                    f"please ensure all sub-wind farm parameter lists have the same length")
            
            # Issue warning if parameter list length is less than total number of sub-wind farms
            if first_len < n_sub_farms:
                logger.warning(f"Sub-wind farm parameter list length({first_len}) is less than total sub-wind farms({n_sub_farms}), "
                              f"missing parameters will use main wind farm parameters")
        
        # Fill default values
        self._fill_default_sub_params(n_sub_farms)
    
    def _fill_default_sub_params(self, n_sub_farms):
        """
        Fill default parameters for sub-wind farms
        
        Args:
            n_sub_farms: Total number of sub-wind farms
        """
        if n_sub_farms == 0:
            return
        
        # If sub-wind farm location parameters are not provided, use main site location
        if not self.sub_lats:
            self.sub_lats = [None] * n_sub_farms
            logger.info(f"Sub-wind farm latitudes not provided, will use main wind farm latitude")
        elif len(self.sub_lats) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_lats)
            self.sub_lats.extend([None] * missing_count)
            logger.info(f"Sub-wind farm latitude list length insufficient, will use main wind farm latitude for {missing_count} sub-wind farms")
        
        if not self.sub_lons:
            self.sub_lons = [None] * n_sub_farms
            logger.info(f"Sub-wind farm longitudes not provided, will use main wind farm longitude")
        elif len(self.sub_lons) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_lons)
            self.sub_lons.extend([None] * missing_count)
            logger.info(f"Sub-wind farm longitude list length insufficient, will use main wind farm longitude for {missing_count} sub-wind farms")
        
        if not self.sub_years:
            self.sub_years = [None] * n_sub_farms
            logger.info(f"Sub-wind farm years not provided, will use main wind farm year")
        elif len(self.sub_years) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_years)
            self.sub_years.extend([None] * missing_count)
            logger.info(f"Sub-wind farm year list length insufficient, will use main wind farm year for {missing_count} sub-wind farms")
        
        if not self.sub_hub_heights:
            self.sub_hub_heights = [None] * n_sub_farms
            logger.info(f"Sub-wind farm hub heights not provided, will use main wind farm hub height")
        elif len(self.sub_hub_heights) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_hub_heights)
            self.sub_hub_heights.extend([None] * missing_count)
            logger.info(f"Sub-wind farm hub height list length insufficient, will use main wind farm hub height for {missing_count} sub-wind farms")
        
        if not self.sub_turbine_ratings:
            self.sub_turbine_ratings = [None] * n_sub_farms
            logger.info(f"Sub-wind farm turbine ratings not provided, will use main wind farm turbine rating")
        elif len(self.sub_turbine_ratings) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_turbine_ratings)
            self.sub_turbine_ratings.extend([None] * missing_count)
            logger.info(f"Sub-wind farm turbine rating list length insufficient, will use main wind farm turbine rating for {missing_count} sub-wind farms")
        
        if not self.sub_num_turbines:
            self.sub_num_turbines = [None] * n_sub_farms
            logger.info(f"Sub-wind farm number of turbines not provided, will use main wind farm number of turbines")
        elif len(self.sub_num_turbines) < n_sub_farms:
            missing_count = n_sub_farms - len(self.sub_num_turbines)
            self.sub_num_turbines.extend([None] * missing_count)
            logger.info(f"Sub-wind farm number of turbines list length insufficient, will use main wind farm number of turbines for {missing_count} sub-wind farms")
    
    @classmethod
    def from_dict(cls, data: dict):
        """Construct configuration from dictionary"""
        # Let parent class parse basic fields first
        config = super().from_dict(data)
        
        # Manually parse sub-wind farm specific fields
        config.sub_wind_resources = data.get("sub_wind_resources", [])
        config.resource_files = data.get("resource_files", [])
        config.sub_lats = data.get("sub_lats", [])
        config.sub_lons = data.get("sub_lons", [])
        config.sub_years = data.get("sub_years", [])
        config.sub_hub_heights = data.get("sub_hub_heights", [])
        config.sub_turbine_ratings = data.get("sub_turbine_ratings", [])
        config.sub_num_turbines = data.get("sub_num_turbines", [])
        
        return config
    
    def get_base_config_dict(self):
        """
        Get base configuration dictionary, removing multi-wind farm specific fields
        
        Returns:
            dict: Configuration dictionary without multi-wind farm specific fields
        """
        base_dict = self.as_dict()
        # Remove multi-wind farm specific fields
        multi_wind_fields = [
            'sub_wind_resources', 'resource_files', 'sub_lats', 'sub_lons', 
            'sub_years', 'sub_hub_heights', 'sub_turbine_ratings', 'sub_num_turbines'
        ]
        for field in multi_wind_fields:
            base_dict.pop(field, None)
        return base_dict

@define
class MultiWindPlant(PowerSource):
    """
    Multi-wind farm management class
    
    This class manages multiple wind farms and combines their outputs. In HybridSimulation, it behaves the same as a single WindPlant,
    but internally manages multiple wind farm instances.
    
    Note: HybridSimulation sequentially calls `simulate_power`, `calculate_installed_cost`,
    `calculate_financials`, `simulate_financials`, rather than calling the `simulate` method.
    """
    site: SiteInfo
    config: MultiWindConfig
    config_name: str = field(init=False, default="WindPowerSingleOwner")

    def __attrs_post_init__(self):
        """Initialize multi-wind farm"""
        # Initialize private attributes
        self._generation_profile = None
        self.outputs = {}
        
        # Store all wind farms
        self.plants = []
        
        # Create main wind farm
        main_base = self.config.get_base_config_dict()
        main_base["model_name"] = "pysam"
        main_config = WindConfig.from_dict(main_base)

        # Print main wind farm configuration
      #  print(f"[DEBUG] Main wind farm configuration: {main_config}")
        
        main_plant = WindPlant(self.site, main_config)
        self.plants.append(main_plant)
        
        # Create wind farms from resource file list
        if self.site.wind_resource_files:
            for i, wind_resource_file in enumerate(self.site.wind_resource_files[1:]):
                # Get sub-wind farm specific parameters
                idx = i
                lat = self.config.sub_lats[idx] if idx < len(self.config.sub_lats) and self.config.sub_lats[idx] is not None else self.site.lat
                lon = self.config.sub_lons[idx] if idx < len(self.config.sub_lons) and self.config.sub_lons[idx] is not None else self.site.lon
                year = self.config.sub_years[idx] if idx < len(self.config.sub_years) and self.config.sub_years[idx] is not None else self.site.year
                

                # Create site information (using specified resource file)
                site_data = {
                    "data": {
                        "lat": lat,
                        "lon": lon,
                        "year": year,
                        "site_details": {
                            "site_shape":     "rectangle",      # circle, rectangle, hexagon…
                            "site_area_km2": 100, # Required or provide site_area_km2
                            "aspect_ratio": 3.65
                    }
                    },

                    "wind_resource_files": [self.site.wind_resource_files[i+1]],  # Ensure format consistency with WindPlant
                    "solar": False,
                }
                

                farm_site = SiteInfo.from_dict(site_data)
                
                # Create sub-wind farm configuration
                farm_base = self.config.get_base_config_dict()
                farm_base["model_name"] = "pysam"

                # Override sub-wind farm specific parameters
                if idx < len(self.config.sub_hub_heights) and self.config.sub_hub_heights[idx] is not None:
                    farm_base["hub_height"] = self.config.sub_hub_heights[idx]
                
                if idx < len(self.config.sub_turbine_ratings) and self.config.sub_turbine_ratings[idx] is not None:
                    farm_base["turbine_rating_kw"] = self.config.sub_turbine_ratings[idx]
                
                if idx < len(self.config.sub_num_turbines) and self.config.sub_num_turbines[idx] is not None:
                    farm_base["num_turbines"] = self.config.sub_num_turbines[idx]
                
                farm_config = WindConfig.from_dict(farm_base)
                
                # Print sub-wind farm configuration
                # print(f"[DEBUG] Sub-wind farm {i+len(self.config.sub_wind_resources)} configuration: {farm_config}")
                
                # Create wind farm
                plant = WindPlant(farm_site, farm_config)
                self.plants.append(plant)
        
        # Initialize PowerSource base class using the first wind farm's model
        if len(self.plants) == 0:
            raise ValueError("Unable to create any wind farm")
        
        first_plant = self.plants[0]
        super().__init__(
            name="MultiWindPlant",
            site=self.site, 
            system_model=first_plant._system_model,
            financial_model=first_plant._financial_model
        )
        
        logger.info(f"Created MultiWindPlant with {len(self.plants)} wind farms")

    def _ensure_array(self, gen_max):
        """Ensure gen_max_feasible is an array"""
        arr = np.array(gen_max)
        if arr.ndim == 0:
            arr = np.ones(self.site.n_timesteps) * arr
        return arr
    
    def _merge_installed_cost(self):
        """
        Merge installation costs of all sub-wind farms
        
        This method is called after simulate_power to ensure total_installed_cost is available in the financial model
        """
        try:
            costs = [plant.total_installed_cost for plant in self.plants]
            total_cost = sum(costs)
            
            # Update both attribute and underlying PySAM model
            self.total_installed_cost = total_cost
            self._system_model.value("total_installed_cost", total_cost)
            
            logger.info(f"Merged installation costs of {len(self.plants)} wind farms, total: {total_cost}")
        except Exception as e:
            logger.error(f"Error merging installation costs: {e}")
            # Don't throw exception as this is not a critical error, HybridSimulation can continue

    def simulate_power(self, project_life: int = 25, lifetime_sim=False):
        """
        Simulate power generation of multiple wind farms and merge results
        
        Args:
            project_life: Project lifetime in years
            lifetime_sim: Whether to perform full lifetime simulation
            
        Returns:
            np.ndarray: Merged power generation curve
            
        Raises:
            ValueError: If power generation curve length doesn't match expectation
            RuntimeError: If sub-wind farm simulation fails
        """
        # Call each sub-wind farm
        failed_plants = []
        # print(f"[DEBUG] Number of wind farms: {len(self.plants)}")
        for i, plant in enumerate(self.plants):
            try:
                plant.simulate_power(project_life, lifetime_sim)
                if i == 0:
                    print(f"[DEBUG] Main wind farm configuration: hub_height={plant.config.hub_height}, num_turbines={plant.config.num_turbines}, turbine_rating={plant.config.turbine_rating_kw}")
                else:
                    print(f"[DEBUG] Sub-wind farm {i} configuration: hub_height={plant.config.hub_height}, num_turbines={plant.config.num_turbines}, turbine_rating={plant.config.turbine_rating_kw}")
            except Exception as e:
                logger.error(f"Sub-wind farm {i} simulation failed: {e}")
                failed_plants.append(i)
        
        # Debug output annual energy for each wind farm
        for i, plant in enumerate(self.plants):
            try:
                annual_energy = plant.annual_energy_kwh
                #print(f"[DEBUG] Sub-wind farm {i} annual energy: {annual_energy} kWh")
            except Exception as e:
                print(f"[DEBUG] Sub-wind farm {i} failed to get annual energy: {e}")
        
        if len(failed_plants) == len(self.plants):
            raise RuntimeError(f"All sub-wind farm simulations failed, cannot continue")
        
        # Output power generation curve for each wind farm
        for i, plant in enumerate(self.plants):
            try:
                generation = plant.generation_profile
                # if i == 0:
                #     print(f"[DEBUG] Main wind farm power curve: {generation}")
                # else:
                #     print(f"[DEBUG] Sub-wind farm {i} power curve: {generation}")
            except Exception as e:
                print(f"[DEBUG] Sub-wind farm {i} failed to get annual energy: {e}")
        
        if len(failed_plants) == len(self.plants):
            raise RuntimeError(f"All sub-wind farm simulations failed, cannot continue")



        # Merge power generation curves
        profiles = []
        for i, plant in enumerate(self.plants):
            try:
                if plant.generation_profile is not None:
                    profiles.append(plant.generation_profile)
                    # print(f"[DEBUG] Wind farm {i} power curve: {plant.generation_profile}")
                else:
                    print(f"[DEBUG] Wind farm {i} power curve is empty")
            except Exception as e:
                print(f"[DEBUG] Wind farm {i} failed to get power curve: {e}")
        
        if not profiles:
            raise RuntimeError("No available power generation curve data")

        ####### Add power generation curve visualization code
        if profiles and len(self.plants) > 1:
            try:
                import matplotlib.pyplot as plt

                # Create chart to display power generation curves of each wind farm
                plt.figure(figsize=(12, 8))

                # Plot power generation curve for each wind farm
                for i, (plant, profile) in enumerate(zip(self.plants, profiles)):
                    if profile is not None and len(profile) > 0:
                        # For clear visualization, only show data for the first 500 hours
                        hours_to_show = min(500, len(profile))
                        time_points = range(hours_to_show)
                        plt.plot(time_points, profile[:hours_to_show],
                                 label=f'wind farm {i} (hub height: {plant.config.hub_height}m, '
                                       f'turbine number: {plant.config.num_turbines})',
                                 alpha=0.7)

                # Plot total power generation curve
                if profiles:
                    total_profile = np.sum(profiles, axis=0)
                    hours_to_show = min(500, len(total_profile))
                    time_points = range(hours_to_show)
                    plt.plot(time_points, total_profile[:hours_to_show],
                             label='total generation', linewidth=2, color='black')

                plt.xlabel('time (h)')
                plt.ylabel('generated power (kW)')
                plt.title('Comparison of power generation curves of multiple wind farms')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save chart
                import os
                if not os.path.exists('output'):
                    os.makedirs('output')
                plt.savefig('output/multi_wind_generation_profiles.png', dpi=300, bbox_inches='tight')
                # print(f"[DEBUG] Power generation curve chart saved to output/multi_wind_generation_profiles.png")
                plt.close()

            except Exception as e:
                print(f"[DEBUG] Error generating power generation curve chart: {e}")

        if not profiles:
            raise RuntimeError("No available power generation curve data")

        self._generation_profile = np.sum(profiles, axis=0)
        # print(f"[DEBUG] Merged power generation curve: {self._generation_profile}")
        
        # Verify if time step length is consistent
        if len(self._generation_profile) != self.site.n_timesteps * project_life and \
           len(self._generation_profile) != self.site.n_timesteps:
            raise ValueError(f"Generated power curve length {len(self._generation_profile)} does not match expectation "
                            f"(should be {self.site.n_timesteps} or {self.site.n_timesteps * project_life})")
        
        # Merge maximum feasible power generation
        feas = [self._ensure_array(plant.gen_max_feasible) for plant in self.plants if hasattr(plant, 'gen_max_feasible')]
        if feas:
            self.gen_max_feasible = np.sum(feas, axis=0)
        
        # Calculate total capacity - must be set, otherwise PowerSource.generation_profile will throw NotImplementedError
        total_capacity = sum(plant.system_capacity_kw for plant in self.plants)
        self._system_model.value("system_capacity", total_capacity)
        
        # Calculate and store annual energy generation
        if hasattr(self, "outputs"):
            self.outputs["annual_energy_kwh"] = sum(plant.annual_energy_kwh for plant in self.plants)
        
        logger.info(f"Total annual energy generation of multi-wind farms: {sum(plant.annual_energy_kwh for plant in self.plants)/1000:.2f} MWh")
        return self._generation_profile
    
    def setup_performance_model(self):
        """Set up performance model, maintain consistency with PowerSource base class interface"""
        # Call performance model setup for each sub-wind farm
        for plant in self.plants:
            try:
                plant.setup_performance_model()
            except Exception as e:
                logger.warning(f"Sub-wind farm performance model setup failed: {e}")
    

    def value(self, param, value=None):
        """
        Get or set parameter value
        
        Handle special attributes like annual_energy_kwh, maintaining consistency with PowerSource.
        
        Args:
            param: Parameter name
            value: Parameter value (optional)
            
        Returns:
            Parameter value (if get operation)
        """
        # Special handling for certain parameters
        if param == "annual_energy_kwh" and value is None:
            return self.annual_energy_kwh
        
        # Set operation
        if value is not None:
            # Set parameter value for itself
            result = self._system_model.value(param, value)
            return result
        
        # Get operation
        return self._system_model.value(param)

    @property
    def system_capacity_kw(self):
        """System capacity [kW]"""
        try:
            return self._system_model.value("system_capacity")
        except:
            # If model value is not available, calculate directly
            return sum(plant.system_capacity_kw for plant in self.plants)
    
    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """Set system capacity"""
        # Update system capacity in the model
        self._system_model.value("system_capacity", size_kw)
        
        # Proportionally adjust capacity of each sub-wind farm
        current_total = sum(plant.system_capacity_kw for plant in self.plants)
        if current_total > 0:
            ratio = size_kw / current_total
            for plant in self.plants:
                new_cap = plant.system_capacity_kw * ratio
                plant.system_capacity_kw = new_cap

    @property
    def generation_profile(self):
        """Power generation curve [kW]"""
        if hasattr(self, "_generation_profile") and self._generation_profile is not None:
            return self._generation_profile
        
        # If not yet simulated, return zero array
        return np.zeros(self.site.n_timesteps)
    
    @property
    def annual_energy_kwh(self):
        """Annual energy generation [kWh]"""
        if hasattr(self, "outputs") and "annual_energy_kwh" in self.outputs:
            return self.outputs["annual_energy_kwh"]
        
        # If outputs not initialized, calculate directly
        return sum(plant.annual_energy_kwh for plant in self.plants) 