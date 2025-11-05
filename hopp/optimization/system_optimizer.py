"""
System optimization utilities for HOPP.
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from hopp.simulation import HoppInterface
from hopp.utilities import ConfigManager
from hopp.tools.analysis import EconomicCalculator
from .load_analyzer import LoadAnalyzer
from scipy.optimize import differential_evolution
import random
from deap import base, creator, tools, algorithms


class SystemOptimizer:
    """Handles system optimization and configuration."""
    
    def __init__(self, 
                 yaml_file_path: str, 
                 economic_calculator: EconomicCalculator,
                 enable_flexible_load: bool = True,
                 max_load_reduction_percentage: float = 0.2,
                 pv_preference_weight: float = 0.0):
        """Initialize SystemOptimizer."""
        self.yaml_file_path = yaml_file_path
        self.economic_calculator = economic_calculator
        self.config_manager = ConfigManager()
        self.load_analyzer = LoadAnalyzer(
            enable_flexible_load=enable_flexible_load,
            max_load_reduction_percentage=max_load_reduction_percentage
        )
        self.turbine_ratio = 1.0
        self.target_demand_met = None
        self.demand_met_tolerance = None
        self.pv_preference_weight = pv_preference_weight  # PV preference weight

    def set_turbine_ratio(self, ratio):
        """Set the ratio of turbine number to sub-turbine number."""
        self.turbine_ratio = ratio

    def set_demand_met_target(self, target_demand_met: float, demand_met_tolerance: float = 0.5):
        """Set the target demand met percentage and tolerance."""
        self.target_demand_met = target_demand_met
        self.demand_met_tolerance = demand_met_tolerance

    def clear_demand_met_target(self):
        """Clear the demand met target constraint."""
        self.target_demand_met = None
        self.demand_met_tolerance = None

    def optimize_system(self, bounds: List[Tuple[float, float]],
                       initial_conditions: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Optimize system configuration.

        Args:
            bounds: List of (min, max) tuples for each parameter.
            initial_conditions: List of initial parameter sets to try.

        Returns:
            Dictionary containing optimal system configuration and metrics.
        """
        best_result = None
        best_lcoe = float('inf')
        found_constraint_satisfying_solution = False

        print(f"Starting optimization, target time_load_met: {self.target_demand_met}% ± {self.demand_met_tolerance}%")

        for i, x0 in enumerate(initial_conditions):
            print(f"Trying initial point {i+1}/{len(initial_conditions)}: {x0}")

            try:
                result = minimize(
                    self.penalized_objective,
                    x0,
                    # method='nelder-mead'('maxiter': 300, 'xatol': 1, 'fatol': 1e-3),
                    # method="L-BFGS-B"('maxiter': 300, 'ftol': 1e-3, 'gtol': 1e-3)

                    method='nelder-mead',
                    bounds=bounds,
                    options={'maxiter': 300, 'xatol': 1, 'fatol': 1e-2}
                )

                if result.success:
                    optimal_config = [
                        int(round(result.x[0])),
                        int(round(result.x[1])),
                        self.round_battery_capacity(result.x[2]),
                        int(round(result.x[3]))
                    ]
                    lcoe, optimal_results = self.objective_function(optimal_config)
                    time_load_met = optimal_results.get('Time Load Met (%)', 0)

                    print(f"  Initial point {i+1} result: LCOE = {lcoe:.4f}, Time Load Met = {time_load_met:.2f}%")

                    # Check if demand met constraint is set
                    if self.target_demand_met is not None and self.demand_met_tolerance is not None:
                        # Check if time_load_met is within target range
                        if abs(time_load_met - self.target_demand_met) <= self.demand_met_tolerance:
                            # Found a solution that meets the constraint
                            found_constraint_satisfying_solution = True
                            if lcoe < best_lcoe:
                                best_lcoe = lcoe
                                best_result = optimal_results
                                print(f"  ✓ Found better solution satisfying constraints: LCOE = {lcoe:.4f}, Time Load Met = {time_load_met:.2f}%")
                        else:
                            print(f"  ✗ Does not satisfy constraints: Time Load Met = {time_load_met:.2f}% (target: {self.target_demand_met}% ± {self.demand_met_tolerance}%)")
                            # Even if constraints are not satisfied, consider as alternative result
                            if lcoe < best_lcoe:
                                best_lcoe = lcoe
                                best_result = optimal_results
                    else:
                        # No demand met constraint, use original logic
                        if lcoe < best_lcoe:
                            best_lcoe = lcoe
                            best_result = optimal_results

            except Exception as e:
                print(f"Optimization failed, initial point {x0}: {str(e)}")
                continue

        if self.target_demand_met is not None and self.demand_met_tolerance is not None:
            if found_constraint_satisfying_solution:
                print(f"✓ Found optimal solution satisfying constraints: LCOE = {best_lcoe:.4f}")
            else:
                print(f"✗ No solution found satisfying constraints (target: {self.target_demand_met}% ± {self.demand_met_tolerance}%)")
                print("Try relaxing constraints or increasing number of initial points...")
                return None

        return best_result



    def optimize_system_de(self, bounds: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        """
            Use differential evolution algorithm for global optimization

        Args:
            bounds: Parameter boundary list [(min, max), ...]

        Returns:
            Optimal system configuration and metrics
        """
        print(f"Starting differential evolution global optimization, target time_load_met: {self.target_demand_met}% ± {self.demand_met_tolerance}%")

        try:
            # Use differential evolution algorithm
            result = differential_evolution(
                self.penalized_objective,
                bounds,
                strategy='best1bin',  # Mutation strategy
                maxiter=50,  # Maximum iterations
                popsize=10,  # Population size
                mutation=(0.5, 1.0),  # Mutation factor range
                recombination=0.7,  # Recombination probability
                tol=1e-2,  # Tolerance
                seed=42,  # Random seed for reproducibility
                disp=True,  # Display progress
                polish=True,  # Final local optimization
                workers=1,  # Parallel computation
                updating='immediate'
            )

            if result.success:
                # Get optimal solution
                optimal_config = [
                    int(round(result.x[0])),  # PV
                    int(round(result.x[1])),  # Wind turbines
                    self.round_battery_capacity(result.x[2]),  # Battery capacity kWh
                    int(round(result.x[3]))  # Battery capacity kW
                ]

                # Calculate final results
                lcoe, optimal_results = self.objective_function(optimal_config)
                time_load_met = optimal_results.get('Time Load Met (%)', 0)

                print(f"Differential evolution optimization completed:")
                print(f"  Optimal LCOE: {lcoe:.4f}")
                print(f"  Time Load Met: {time_load_met:.2f}%")
                print(f"  Function evaluations: {result.nfev}")

                # Check if constraints are satisfied
                if self.target_demand_met is not None and self.demand_met_tolerance is not None:
                    if abs(time_load_met - self.target_demand_met) <= self.demand_met_tolerance:
                        print("✓ Solution satisfies constraints")
                        return optimal_results
                    else:
                        print(f"✗ Solution does not satisfy constraints: target {self.target_demand_met}% ± {self.demand_met_tolerance}%")
                        return None
                else:
                    return optimal_results
            else:
                print(f"Optimization failed: {result.message}")
                return None

        except Exception as e:
            print(f"Error occurred during differential evolution optimization: {str(e)}")
            return None


    def optimize_system_ga(self, bounds: List[Tuple[float, float]],
                           pop_size: int = 50,
                           generations: int = 100) -> Optional[Dict[str, Any]]:
        """
        Use genetic algorithm for global optimization

        Args:
            bounds: Parameter boundary list [(min, max), ...]
            pop_size: Population size
            generations: Evolution generations

        Returns:
            Optimal system configuration and metrics
        """
        print(f"Starting genetic algorithm global optimization, target time_load_met: {self.target_demand_met}% ± {self.demand_met_tolerance}%")

        # Define fitness and individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Define gene ranges
        for i, (min_val, max_val) in enumerate(bounds):
            toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3, toolbox.attr_4], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            return (self.penalized_objective(individual),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run genetic algorithm
        population = toolbox.population(n=pop_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)

        # Get optimal individual
        best_ind = tools.selBest(population, 1)[0]
        optimal_config = [
            int(round(best_ind[0])),
            int(round(best_ind[1])),
            self.round_battery_capacity(best_ind[2]),
            int(round(best_ind[3])),
            int(round(best_ind[4]))
        ]

        lcoe, optimal_results = self.objective_function(optimal_config)
        time_load_met = optimal_results.get('Time Load Met (%)', 0)

        print(f"Genetic algorithm optimization completed:")
        print(f"  Optimal LCOE: {lcoe:.4f}")
        print(f"  Time Load Met: {time_load_met:.2f}%")

        return optimal_results

    def optimize_system_adaptive(self, bounds: List[Tuple[float, float]],
                                initial_conditions: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Adaptive optimization that gradually relaxes constraints if no solution is found.

        Args:
            bounds: List of (min, max) tuples for each parameter.
            initial_conditions: List of initial parameter sets to try.

        Returns:
            Dictionary containing optimal system configuration and metrics.
        """
        if self.target_demand_met is None or self.demand_met_tolerance is None:
            return self.optimize_system(bounds, initial_conditions)

        original_tolerance = self.demand_met_tolerance
        tolerance_steps = [0.5, 1.0, 2.0, 5.0, 10.0]  # Gradually relax constraints

        for step, tolerance in enumerate(tolerance_steps):
            print(f"\n=== Trying constraint tolerance {tolerance}% (step {step+1}/{len(tolerance_steps)}) ===")
            self.demand_met_tolerance = tolerance

            result = self.optimize_system(bounds, initial_conditions)

            if result is not None:
                time_load_met = result.get('Time Load Met (%)', 0)
                lcoe = result.get('System LCOE (cent$/kWh)', float('inf'))

                print(f"✓ Solution found: Time Load Met = {time_load_met:.2f}%, LCOE = {lcoe:.4f}")
                print(f"Constraint tolerance: {tolerance}% (original target: {self.target_demand_met}% ± {original_tolerance}%)")

                # Restore original tolerance
                self.demand_met_tolerance = original_tolerance
                return result

        # Restore original tolerance
        self.demand_met_tolerance = original_tolerance
        print(f"✗ No solution found even with constraints relaxed to {tolerance_steps[-1]}%")
        return None

    @staticmethod
    def round_battery_capacity(capacity: float) -> float:
        """Round battery capacity to nearest MWh."""
        return round(capacity / 1000) * 1000

    def penalized_objective(self, x: List[float]) -> float:
        """Calculate penalized objective function value."""
        x_rounded = [
            int(round(x[0])),  # PV
            int(round(x[1])),  # Wind turbines
            self.round_battery_capacity(x[2]),  # Battery capacity kWh
            int(round(x[3]))  # Battery capacity kW
        ]
        lcoe, results = self.objective_function(x_rounded)

        # Add penalty for constraint violation
        if self.target_demand_met is not None and self.demand_met_tolerance is not None:
            time_load_met = results.get('Time Load Met (%)', 0)
            constraint_violation = abs(time_load_met - self.target_demand_met)

            if constraint_violation > self.demand_met_tolerance:
                # Add penalty for constraint violation
                penalty_factor = 1.0 + (constraint_violation / self.demand_met_tolerance) * 0.5
                lcoe *= penalty_factor

        # Add PV preference logic
        if self.pv_preference_weight > 0:
            pv_capacity = x_rounded[0]
            wind_turbines = x_rounded[1]
            total_generation = results.get('Total System Generation (kWh)', 1)
            pv_generation = results.get('Total PV Generation (kWh)', 0)
            wind_generation = results.get('Total Wind Generation (kWh)', 0)

            # Calculate PV ratio in total renewable generation
            renewable_generation = pv_generation + wind_generation
            if renewable_generation > 0:
                pv_ratio = pv_generation / renewable_generation
                # Reward higher PV ratios (reduce LCOE for higher PV ratios)
                pv_bonus = self.pv_preference_weight * (1.0 - pv_ratio)
                lcoe *= (1.0 + pv_bonus)

        return lcoe

    def objective_function(self, x: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Calculate objective function value and system metrics."""
        pv_size, num_turbines, battery_capacity_kwh, battery_capacity_kw = x
        battery_capacity_kwh = self.round_battery_capacity(battery_capacity_kwh)

        # Update configuration
        config = self.config_manager.load_yaml_safely(self.yaml_file_path)
        config['technologies']['pv']['system_capacity_kw'] = float(pv_size)
        if self.turbine_ratio>1:
            config['technologies']['wind']['num_turbines'] = int(num_turbines)
            config['technologies']['wind']['sub_num_turbines'][0] = int(math.ceil(num_turbines * self.turbine_ratio))
        else:
            config['technologies']['wind']['num_turbines'] = int(math.ceil(num_turbines / self.turbine_ratio))
            config['technologies']['wind']['sub_num_turbines'][0] = int(num_turbines)

        config['technologies']['battery']['system_capacity_kwh'] = float(battery_capacity_kwh)
        config['technologies']['battery']['system_capacity_kw'] = float(battery_capacity_kw)
        self.config_manager.save_yaml_safely(config, self.yaml_file_path)

        # Run simulation
        try:
            Py_Microgrid = HoppInterface(self.yaml_file_path)
            Py_Microgrid.simulate(project_life=self.economic_calculator.project_lifetime)
        except Exception as e:
            print(f"Simulation failed: {e}")
            return 1e6, {}  # Return high LCOE for failed simulations

        hybrid_plant = Py_Microgrid.system

        # Calculate generation
        pv_total_generation = np.sum(hybrid_plant.generation_profile.pv)
        wind_total_generation = np.sum(hybrid_plant.generation_profile.wind)
        battery_total_generation = np.sum(hybrid_plant.generation_profile.battery)
        genset_total_generation = np.sum(np.tile(hybrid_plant.site.desired_schedule*1000,self.economic_calculator.project_lifetime)-hybrid_plant.generation_profile.grid)
        total_system_generation = pv_total_generation + wind_total_generation + genset_total_generation

        
        # Get time_load_met from grid
        time_load_met = hybrid_plant.grid.time_load_met if hasattr(hybrid_plant.grid, 'time_load_met') else 0
        
        # Calculate LCOE (hybrid_plant.lcoe_nom.hybrid unit is cents/kWh)
        lcoe = hybrid_plant.lcoe_nom.hybrid

        # Prepare results
        result = {
            "PV Capacity (kW)": pv_size,
            "Wind Turbine Capacity (kW)": hybrid_plant.wind.system_capacity_kw,
            "wind turbine numbers": int(num_turbines),
            "Battery Energy Capacity (kWh)": battery_capacity_kwh,
            "Battery Power Capacity (kW)": battery_capacity_kw,
            "Total System Generation (kWh)": total_system_generation,
            "Total PV Generation (kWh)": pv_total_generation,
            "Total Wind Generation (kWh)": wind_total_generation,
            "Total Genset Generation (kWh)": genset_total_generation,
            "Total Battery Generation (kWh)": battery_total_generation,
            "System LCOE (cents/kWh)": lcoe,  # Explicit unit as cents/kWh
            "Time Load Met (%)": time_load_met,  # Add time_load_met
        }

        return lcoe, result
