# Autonomous, Physics-Grounded Pipeline for Multi-Objective Alloy Design

AutoMAT is a hierarchical and autonomous framework for alloy discovery that integrates large language models (LLMs), automated CALPHAD simulations, AI-driven search, and experimental validation into a unified design loop. To our knowledge, AutoMAT is the first system to span the entire alloy design pipeline, from ideation to simulation and optimization, offering a more scalable, interpretable, and efficient alternative to conventional methods.

📄 **Preprint**: [AutoMAT: A Hierarchical Framework for Autonomous Alloy Discovery](https://arxiv.org/abs/2507.16005)

## Overview

The framework comprises three modular layers that integrate the complementary strengths of different approaches:

1. **Ideation Layer**: Uses LLMs to extract and propose alloy candidates from literature and handbooks based on user-defined targets. By embedding physics priors from established models and scientific literature, it delivers structured, physically consistent suggestions within minutes.

2. **Simulation Layer**: Automates CALPHAD-based thermodynamic modeling for the first time, coupled with AI-guided search to optimize compositions using orders-of-magnitude fewer evaluations.

3. **Validation Layer**: (not included in this repo) Physically performs synthesis and characterization of top-ranked candidates, providing critical feedback on real-world performance.

AutoMAT integrates the complementary strengths of each method: LLMs for intelligent and data-efficient knowledge retrieval; CALPHAD for thermodynamic accuracy and interpretability; AI-driven search for high-throughput time efficiency and automation.

## Project Structure

```
AutoMAT/
├── IdeationLayer/                        # Alloy recommendation system
│   ├── alloy_recommendation_system.py    # Main program
│   ├── step1_llm_retrieval.py            # PDF information extraction
│   ├── step2_json_extractor_recommend.py # JSON data extraction
│   ├── step3_llm_filter_json.py          # Requirements-based filtering
│   ├── step4_llm_recommend_json.py       # Final recommendations
│   ├── prompt/                           # LLM prompt templates
│   ├── requirements_extractor.py         # Requirements processing
│   └── user_requirements.txt             # User input file
└── SimulationLayer/                      # CALPHAD optimization engine
    ├── AutoMAT_SimulationLayer.py        # Main optimization engine
    ├── util/                             # Calculation utilities
    │   ├── YS.py                         # Yield strength calculations
    │   ├── composition_unify.py          # Composition utilities
    │   ├── phase_volume.py               # Phase volume (Scheil)
    │   └── tc_single_point.py            # Thermodynamic calculations
    └── save/                             # Optimization results
```

## Quick Start

### Prerequisites

   - numpy, pandas, pymupdf, tqdm, openai
   - OpenAI API key (for Ideation Layer)
   - TC-Python license (for Simulation Layer)

### Ideation Layer Usage

1. Set your OpenAI API key in `IdeationLayer/alloy_recommendation_system.py`
2. Define requirements in `IdeationLayer/user_requirements.txt`
3. Place material handbooks (PDFs) in `IdeationLayer/paper/`
4. Run the system:
   ```bash
   cd IdeationLayer
   python alloy_recommendation_system.py
   ```

**Output:** Detailed alloy recommendations with supporting data from material handbooks.

### Simulation Layer Usage

Run predefined optimization cases:

```bash
cd SimulationLayer

# Ti-based alloys (yield strength / exp(density) optimization)
python AutoMAT_SimulationLayer.py --case case1 --iterations 5 --workers 5

# High-entropy alloys (yield strength optimization)
python AutoMAT_SimulationLayer.py --case case2 --iterations 5 --workers 5

# Custom scoring function
python AutoMAT_SimulationLayer.py --case own_case --iterations 5 --workers 5
```

**Output:** Optimized compositions with thermodynamic and mechanical properties.

## Key Achievements

AutoMAT has demonstrated significant improvements in alloy discovery:

- **Lightweight High-Strength Alloy**: Identified a titanium alloy with **8.1% lower density** and comparable yield strength relative to state-of-the-art reference, achieving the highest specific strength among all comparisons
- **High-Entropy Alloy Optimization**: Achieved **28.2% improvement in yield strength** over the base alloy
- **Timeline Reduction**: Reduces discovery timeline **from years to weeks**
- **High Efficiency**: Achieves results without manually curated large datasets

## Documentation

- [Ideation Layer README](IdeationLayer/README_IdeationLayer.md) - Detailed usage and configuration
- [Simulation Layer README](SimulationLayer/README_SimulationLayer.md) - Advanced optimization settings

## Contributing

AutoMAT is designed to be extensible. Key extension points:
- Custom scoring functions in Simulation Layer
- Additional LLM prompts in Ideation Layer
- New utility functions for property calculations

## License

Please ensure compliance with TC-Python licensing terms when using the Simulation Layer.
