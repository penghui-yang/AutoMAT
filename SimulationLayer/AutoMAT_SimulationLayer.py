"""
AutoMAT Simulation Layer - AI-Guided CALPHAD Optimization

This module implements the Simulation Layer described in the AutoMAT paper, featuring:
- AI-driven iterative neighborhood search with coarse-to-fine perturbations
- Support for two case studies with different scoring functions
- Multi-threaded execution for high-throughput composition evaluation
- Adaptive step sizes and search ranges for optimization

Case 1: Ti-based alloys with score = yield_strength / exp(density)
Case 2: High-entropy alloys with score = yield_strength
"""

import itertools
import os
import numpy as np
import pandas as pd
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

from util.YS import get_total_yield_strength
from util.composition_unify import composition_unify
from util.phase_volume import phase_volume_compute_with_scheil
from util.tc_single_point import tc_single_point


class ScoreFunction(ABC):
    """Abstract base class for scoring functions"""
    
    @abstractmethod
    def calculate(self, result: 'CompositionResult') -> float:
        """
        Calculate score for a composition result
        
        Args:
            result: CompositionResult with calculated properties
            
        Returns:
            Score value (higher is better)
        """
        pass
    
    @abstractmethod
    def requires_density(self) -> bool:
        """Return True if this scoring function requires density calculation"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this scoring function"""
        pass


class YieldStrengthScore(ScoreFunction):
    """Simple yield strength maximization (Case 2 from paper)"""
    
    def calculate(self, result: 'CompositionResult') -> float:
        if result.yield_strength is None:
            raise ValueError("Yield strength required for scoring")
        return result.yield_strength
    
    def requires_density(self) -> bool:
        return False
    
    def get_name(self) -> str:
        return "Yield Strength"


class SpecificStrengthScore(ScoreFunction):
    """Specific strength optimization: YS/exp(density) (Case 1 from paper)"""
    
    def calculate(self, result: 'CompositionResult') -> float:
        if result.density is None or result.yield_strength is None:
            raise ValueError("Density and yield strength required for specific strength scoring")
        return result.yield_strength / math.exp(result.density)
    
    def requires_density(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "Specific Strength (YS/exp(ρ))"


class OwnScore(ScoreFunction):
    """
    TODO:
    Here is a simple strength-to-density ratio as an example.
    You can define your own scoring function here.
    """
    
    def calculate(self, result: 'CompositionResult') -> float:
        if result.density is None or result.yield_strength is None:
            raise ValueError("Density and yield strength required for strength-to-density ratio")
        return result.yield_strength / result.density
    
    def requires_density(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "Strength-to-Density Ratio"


class CustomScore(ScoreFunction):
    """Custom scoring function using user-provided callable"""
    
    def __init__(self, score_func: Callable[['CompositionResult'], float], 
                 requires_density: bool = False, name: str = "Custom"):
        """
        Initialize custom scoring function
        
        Args:
            score_func: Function that takes CompositionResult and returns float
            requires_density: Whether density calculation is needed
            name: Descriptive name for the scoring function
        """
        self.score_func = score_func
        self._requires_density = requires_density
        self.name = name
    
    def calculate(self, result: 'CompositionResult') -> float:
        return self.score_func(result)
    
    def requires_density(self) -> bool:
        return self._requires_density
    
    def get_name(self) -> str:
        return self.name


@dataclass
class SearchConfig:
    """Configuration for the AI-guided search"""
    seed_composition: Dict[str, float]
    score_function: ScoreFunction
    initial_step_size: float
    final_step_size: float
    initial_search_range: float
    final_search_range: float
    max_iterations: int
    max_workers: int
    target_temperature: float = 25 + 273.15  # Room temperature in Kelvin
    grain_size: float = 100.0  # Fixed grain size in micrometers
    

class CompositionResult:
    """Container for composition evaluation results"""
    def __init__(self, composition: Dict[str, float], dependent_element: str):
        self.composition = composition
        self.dependent_element = dependent_element
        self.phase_volumes: Optional[Dict[str, float]] = None
        self.yield_strength: Optional[float] = None
        self.density: Optional[float] = None
        self.score: Optional[float] = None
        self.error: Optional[str] = None
        
    def is_valid(self) -> bool:
        """Check if the result contains valid data"""
        return (self.phase_volumes is not None and 
                self.yield_strength is not None and 
                self.score is not None and 
                self.error is None)


class AutoMATSimulationLayer:
    """
    AI-guided CALPHAD optimization engine for alloy design
    
    Implements the methodology described in the AutoMAT paper with support
    for both Ti-based alloys (Case 1) and high-entropy alloys (Case 2).
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.results_history: List[CompositionResult] = []
        self.best_composition: Optional[CompositionResult] = None
        self.current_iteration = 0
        self.search_center: Dict[str, float] = config.seed_composition.copy()
        
        # Create results directory
        os.makedirs('./save', exist_ok=True)
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame(columns=[
            "Iteration", "Dependent Element", "Composition", "Phase Volumes", 
            "Yield Strength (MPa)", "Density (g/cm³)", "Score", "Score Function", "Step Size", "Search Range"
        ])
        
    def calculate_score(self, result: CompositionResult) -> float:
        """
        Calculate the score using the configured scoring function
        
        Args:
            result: CompositionResult with calculated properties
            
        Returns:
            Score value
        """
        return self.config.score_function.calculate(result)
    
    def generate_neighborhood(self, center: Dict[str, float], step_size: float, 
                            search_range: float) -> List[Dict[str, float]]:
        """
        Generate neighborhood compositions around the center point
        
        Args:
            center: Current best composition
            step_size: Step size for perturbations
            search_range: Maximum range for perturbations
            
        Returns:
            List of neighbor compositions
        """
        neighbors = []
        elements = list(center.keys())
        
        # Generate perturbation ranges for each element
        perturbation_ranges = {}
        for element in elements:
            current_value = center[element]
            min_val = max(step_size, current_value - search_range)  # Minimum 0.1% to avoid zero
            max_val = min(100.0 - step_size, current_value + search_range)  # Maximum 99% to leave room for others
            perturbation_ranges[element] = np.arange(min_val, max_val + step_size, step_size)
        
        # Generate all combinations
        for values in itertools.product(*perturbation_ranges.values()):
            composition = dict(zip(elements, values))
            
            # Check if composition sums to approximately 100%
            total = sum(composition.values())
            if 99.99 <= total <= 100.01:  # Allow some tolerance
                # Normalize to exactly 100%
                normalized_composition = {k: v * 100.0 / total for k, v in composition.items()}
                neighbors.append(normalized_composition)
        
        return neighbors
    
    def evaluate_composition(self, composition: Dict[str, float]) -> CompositionResult:
        """
        Evaluate a single composition using CALPHAD calculations
        
        Args:
            composition: Elemental composition in mol%
            
        Returns:
            CompositionResult with calculated properties
        """
        try:
            # Unify composition format
            dependent_element, unified_composition = composition_unify(composition)
            result = CompositionResult(unified_composition, dependent_element)
            
            # Calculate phase volumes using Scheil solidification
            phase_volumes = phase_volume_compute_with_scheil(
                unified_composition, dependent_element, modify_flag=True
            )
            
            if not isinstance(phase_volumes, dict):
                result.error = f"Phase volume calculation failed: {phase_volumes}"
                return result
                
            result.phase_volumes = phase_volumes
            
            # Calculate density if required by scoring function
            if self.config.score_function.requires_density():
                mass, volume = tc_single_point(
                    unified_composition, dependent_element, 
                    phases=list(phase_volumes.keys()),
                    property=["mass", "volume"], 
                    temperature=self.config.target_temperature
                )
                
                if mass == -1 or volume == -1:
                    result.error = "Density calculation failed"
                    return result
                    
                result.density = mass / volume / 1e6  # Convert to g/cm³
            
            # Calculate yield strength
            yield_strength = get_total_yield_strength(
                unified_composition, dependent_element, phase_volumes, 
                temperature=self.config.target_temperature
            )
            
            # Ensure yield_strength is a float
            if isinstance(yield_strength, (int, float)):
                result.yield_strength = float(yield_strength)
            else:
                result.error = f"Invalid yield strength calculation: {yield_strength}"
                return result
            
            # Calculate score
            result.score = self.calculate_score(result)
            
            return result
            
        except Exception as e:
            result = CompositionResult(composition, "")
            result.error = f"Evaluation error: {str(e)}"
            return result
    
    def evaluate_compositions_parallel(self, compositions: List[Dict[str, float]]) -> List[CompositionResult]:
        """
        Evaluate multiple compositions in parallel
        
        Args:
            compositions: List of compositions to evaluate
            
        Returns:
            List of CompositionResult objects
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_composition = {
                executor.submit(self.evaluate_composition, comp): comp 
                for comp in compositions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_composition):
                result = future.result()
                results.append(result)
                
                # Print progress
                if result.is_valid():
                    print(f"Evaluated composition with score: {result.score:.2f}")
                else:
                    print(f"Failed to evaluate composition: {result.error}")
        
        return results
    
    def update_search_parameters(self) -> Tuple[float, float]:
        """
        Update step size and search range based on current iteration
        Implements coarse-to-fine search strategy
        
        Returns:
            Tuple of (step_size, search_range)
        """
        progress = self.current_iteration / self.config.max_iterations
        
        # Linear interpolation from initial to final values
        step_size = (self.config.initial_step_size * (1 - progress) + 
                    self.config.final_step_size * progress)
        
        search_range = (self.config.initial_search_range * (1 - progress) + 
                       self.config.final_search_range * progress)
        
        return step_size, search_range
    
    def save_iteration_results(self, results: List[CompositionResult], 
                              step_size: float, search_range: float):
        """Save results from current iteration to DataFrame and CSV"""
        for result in results:
            if result.is_valid():
                new_row = {
                    "Iteration": self.current_iteration,
                    "Dependent Element": result.dependent_element,
                    "Composition": result.composition,
                    "Phase Volumes": result.phase_volumes,
                    "Yield Strength (MPa)": result.yield_strength,
                    "Density (g/cm³)": result.density,
                    "Score": result.score,
                    "Score Function": self.config.score_function.get_name(),
                    "Step Size": step_size,
                    "Search Range": search_range
                }
                
                self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], 
                                          ignore_index=True)
        
        # Save to CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        score_name = self.config.score_function.get_name().replace(" ", "_").replace("/", "_")
        filename = f"./save/AutoMAT_{score_name}_{timestamp}.csv"
        self.results_df.to_csv(filename, index=False)
    
    def run_optimization(self) -> CompositionResult:
        """
        Run the complete AI-guided optimization process
        
        Returns:
            Best composition found during optimization
        """
        print(f"Starting AutoMAT Simulation Layer optimization")
        print(f"Score function: {self.config.score_function.get_name()}")
        print(f"Seed composition: {self.config.seed_composition}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Parallel workers: {self.config.max_workers}")
        print("-" * 60)
        
        # Evaluate seed composition
        print("Evaluating seed composition...")
        seed_result = self.evaluate_composition(self.config.seed_composition)
        
        if not seed_result.is_valid():
            raise RuntimeError(f"Failed to evaluate seed composition: {seed_result.error}")
        
        self.best_composition = seed_result
        self.results_history.append(seed_result)
        
        print(f"Seed composition score: {seed_result.score:.4f}")
        print(f"Seed YS: {seed_result.yield_strength:.2f} MPa")
        if seed_result.density:
            print(f"Seed density: {seed_result.density:.3f} g/cm³")
        print()
        
        # Main optimization loop
        for iteration in range(1, self.config.max_iterations + 1):
            self.current_iteration = iteration
            step_size, search_range = self.update_search_parameters()
            
            print(f"Iteration {iteration}/{self.config.max_iterations}")
            print(f"Step size: {step_size:.2f}, Search range: ±{search_range:.2f}")
            print(f"Search center: {self.search_center}")
            
            # Generate neighborhood around current best composition
            neighbors = self.generate_neighborhood(self.search_center, step_size, search_range)
            print(f"Generated {len(neighbors)} neighbor compositions")
            
            # Evaluate all neighbors in parallel
            start_time = time.time()
            results = self.evaluate_compositions_parallel(neighbors)
            evaluation_time = time.time() - start_time
            
            # Filter valid results and find best
            valid_results = [r for r in results if r.is_valid()]
            
            if not valid_results:
                print("No valid compositions found in this iteration")
                continue
            
            # Find best composition in this iteration
            iteration_best = max(valid_results, key=lambda x: x.score)
            
            # Update global best if improved
            if (iteration_best.score is not None and 
                self.best_composition.score is not None and
                iteration_best.score > self.best_composition.score):
                self.best_composition = iteration_best
                self.search_center = self._composition_to_dict(iteration_best)
                print(f"* NEW BEST! Score: {iteration_best.score:.4f} "
                      f"(YS: {iteration_best.yield_strength:.2f} MPa)")
                if iteration_best.density:
                    print(f"   Density: {iteration_best.density:.3f} g/cm³")
            else:
                print(f"Best in iteration: {iteration_best.score:.4f} "
                      f"(Global best: {self.best_composition.score:.4f})")
            
            # Save results
            self.save_iteration_results(valid_results, step_size, search_range)
            self.results_history.extend(valid_results)
            
            print(f"Evaluated {len(valid_results)} valid compositions in {evaluation_time:.1f}s")
            print(f"Throughput: {len(valid_results)/evaluation_time*3600:.0f} compositions/hour")
            print("-" * 60)
        
        # Final summary
        print("OPTIMIZATION COMPLETE!")
        print(f"Best composition found: {self._composition_to_dict(self.best_composition)}")
        print(f"Dependent element: {self.best_composition.dependent_element}")
        print(f"Final score: {self.best_composition.score:.4f}")
        print(f"Yield strength: {self.best_composition.yield_strength:.2f} MPa")
        if self.best_composition.density:
            print(f"Density: {self.best_composition.density:.3f} g/cm³")
        
        return self.best_composition
    
    def _composition_to_dict(self, result: CompositionResult) -> Dict[str, float]:
        """Convert CompositionResult back to full composition dictionary"""
        full_composition = result.composition.copy()
        
        # Calculate dependent element percentage
        dependent_percentage = 100.0 - sum(full_composition.values())
        full_composition[result.dependent_element] = dependent_percentage
        
        return full_composition


def create_case1_config() -> SearchConfig:
    """Create configuration for Case 1: Ti-based alloys with specific strength optimization"""
    return SearchConfig(
        seed_composition={"Ti": 81.4, "Al": 13.6, "V": 2.8, "Fe": 0.2},  # Ti-185 from paper
        score_function=SpecificStrengthScore(),
        initial_step_size=0.5,
        final_step_size=0.2,
        initial_search_range=10.0,  # ±10 mol%
        final_search_range=2.0,     # ±2 mol%
        max_iterations=5,
        max_workers=4,
    )


def create_case2_config() -> SearchConfig:
    """Create configuration for Case 2: High-entropy alloys with yield strength optimization"""
    return SearchConfig(
        seed_composition={"Al": 0.5*20, "Co": 20, "Cr": 20, "Fe": 20, "Ni": 20},  # Al0.5-Co-Cr-Fe-Ni
        score_function=YieldStrengthScore(),
        initial_step_size=0.5,
        final_step_size=0.5,  # Stable step size for HEA case
        initial_search_range=5.0,
        final_search_range=2.0,
        max_iterations=5,
        max_workers=4,
    )


def create_custom_config(seed_composition: Dict[str, float], 
                        score_function: ScoreFunction,
                        initial_step_size: float = 0.5,
                        final_step_size: float = 0.2,
                        initial_search_range: float = 5.0,
                        final_search_range: float = 2.0,
                        max_iterations: int = 5,
                        max_workers: int = 4) -> SearchConfig:
    """Create a custom configuration with user-defined parameters"""
    return SearchConfig(
        seed_composition=seed_composition,
        score_function=score_function,
        initial_step_size=initial_step_size,
        final_step_size=final_step_size,
        initial_search_range=initial_search_range,
        final_search_range=final_search_range,
        max_iterations=max_iterations,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoMAT Simulation Layer")
    parser.add_argument("--case", choices=["case1", "case2", "own_case"], 
                       default="case1", help="Predefined case to run")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Maximum number of iterations")
    
    args = parser.parse_args()
    
    # Create configuration based on case study
    if args.case == "case1":
        config = create_case1_config()
    elif args.case == "case2":
        config = create_case2_config()
    elif args.case == "own_case":
        # Here is an example. You can define your own search parameters here.
        config = SearchConfig(
            seed_composition={"Ti": 81.4, "Al": 13.6, "V": 2.8, "Fe": 0.2},
            score_function=OwnScore(),
            initial_step_size=0.5,
            final_step_size=0.2,
            initial_search_range=10.0,
            final_search_range=2.0,
            max_iterations=5,
            max_workers=4,
        )
    
    # Override with command line arguments
    config.max_workers = args.workers
    config.max_iterations = args.iterations
    
    # Run optimization
    simulation_layer = AutoMATSimulationLayer(config)
    best_result = simulation_layer.run_optimization()
    
    print(f"\nOptimization completed! Check ./save/ for detailed results.")
