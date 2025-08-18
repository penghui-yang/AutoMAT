# AutoMAT Simulation Layer

An AI-guided CALPHAD optimization engine for high-throughput alloy design using iterative neighborhood search with coarse-to-fine perturbations.

## Overview

This system implements the Simulation Layer described in the AutoMAT paper, providing:
1. AI-driven iterative neighborhood search optimization
2. Support for multiple predefined case studies with different scoring functions
3. Multi-threaded execution for high-throughput composition evaluation
4. Adaptive step sizes and search ranges for efficient optimization
5. Extensible framework for custom scoring functions and search parameters

## File Structure

```
SimulationLayer/
├── AutoMAT_SimulationLayer.py         # Main optimization engine
├── util/                              # Utility modules
│   ├── YS.py                          # Yield strength calculations
│   ├── composition_unify.py           # Composition format utilities
│   ├── phase_volume.py                # Phase volume calculations (Scheil)
│   ├── tc_single_point.py             # Single-point thermodynamic calculations
│   └── logger.py                      # Logging utilities
├── save/                              # Output directory for results
│   └── *.csv                          # Optimization results and history
├── phase_volume.py_cache/             # TC-Python cache directory
└── README_SimulationLayer.md          # This file
```

## Predefined Case Studies

### Case 1: Ti-based Alloys (Specific Strength Optimization)
- **Objective**: Maximize specific strength = yield_strength / exp(density)
- **Seed Composition**: Ti-81.4, Al-13.6, V-2.8, Fe-0.2 (Ti-185 alloy)
- **Search Strategy**: Coarse-to-fine with adaptive step sizes (0.5→0.2)
- **Search Range**: ±10 mol% → ±2 mol%

### Case 2: High-Entropy Alloys (Yield Strength Optimization)
- **Objective**: Maximize yield strength
- **Seed Composition**: Al₀.₅CoCrFeNi (Al-10, Co-20, Cr-20, Fe-20, Ni-20)
- **Search Strategy**: Stable step size (0.5) for HEA exploration
- **Search Range**: ±5 mol% → ±2 mol%

### Own Case: Custom Scoring Function
- **Objective**: Maximize strength-to-density ratio = yield_strength / density (customizable)
- **Seed Composition**: Ti-81.4, Al-13.6, V-2.8, Fe-0.2 (customizable)
- **Parameters**: Fully customizable search parameters (customizable)

## Setup

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy tc-python
   ```

2. **TC-Python License:**
   Ensure you have a valid TC-Python license and database access (TCHEA7 by default).

3. **Database Configuration:**
   The system uses TCHEA7 database by default. Modify the `database` parameter in utility functions if using different databases.

## Usage

### Command Line Interface

Run predefined case studies:
```bash
# Case 1: Ti-based alloys with specific strength optimization
python AutoMAT_SimulationLayer.py --case case1 --iterations 5 --workers 5

# Case 2: High-entropy alloys with yield strength optimization
python AutoMAT_SimulationLayer.py --case case2 --iterations 5 --workers 5

# Own case: Custom scoring function example
python AutoMAT_SimulationLayer.py --case own_case --iterations 5 --workers 5
```

### Programmatic Usage

```python
from AutoMAT_SimulationLayer import (
    AutoMATSimulationLayer, 
    SearchConfig, 
    SpecificStrengthScore,
    YieldStrengthScore,
    CustomScore
)

# Create custom configuration
config = SearchConfig(
    seed_composition={"Ti": 85.0, "Al": 10.0, "V": 5.0},
    score_function=SpecificStrengthScore(),
    initial_step_size=0.5,
    final_step_size=0.2,
    initial_search_range=8.0,
    final_search_range=2.0,
    max_iterations=10,
    max_workers=6
)

# Run optimization
simulation_layer = AutoMATSimulationLayer(config)
best_result = simulation_layer.run_optimization()
```

### Custom Scoring Functions

Create your own scoring function by extending the `ScoreFunction` class:

```python
class MyCustomScore(ScoreFunction):
    def calculate(self, result):
        # Your custom scoring logic here
        return result.yield_strength * some_factor / result.density
    
    def requires_density(self):
        return True  # Set to False if density not needed
    
    def get_name(self):
        return "My Custom Score"

# Use in configuration
config = SearchConfig(
    seed_composition={"Al": 20, "Co": 20, "Cr": 20, "Fe": 20, "Ni": 20},
    score_function=MyCustomScore(),
    # ... other parameters
)
```

## Performance Optimization

### Recommended Settings
- **Workers**: 5 threads for most systems
- **Step Sizes**: Start with 0.5-1.0, end with 0.1-0.2
- **Search Ranges**: Start with 5-10 mol%, end with 1-3 mol%

### System Requirements
- **CPU**: Multi-core processor recommended (4+ cores)
- **Storage**: large SSD recommended for TC-Python cache performance

## Notes

- All intermediate calculations and results are preserved for analysis
- The system automatically handles composition normalization and dependent element selection
- Phase volume calculations use Scheil solidification model by default
- Yield strength calculations include solid solution and grain size strengthening effects
