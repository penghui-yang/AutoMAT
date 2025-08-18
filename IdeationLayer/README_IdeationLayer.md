# AutoMAT Ideation Layer

A comprehensive system for recommending alloys based on user requirements using LLM-powered analysis of material handbooks.

## Overview

This system provides a one-click solution for alloy recommendation by:
1. Analyzing user requirements to recommend appropriate alloy families
2. Extracting relevant information from material handbooks
3. Filtering alloys based on specific criteria
4. Generating detailed recommendations

## File Structure

```
IdeationLayer/
├── alloy_recommendation_system.py      # Main orchestrator (one-click solution)
├── step1_llm_retrieval.py              # Step 1: Extract info from PDFs
├── step2_json_extractor_recommend.py   # Step 2: Extract JSON data
├── step3_llm_filter_json.py            # Step 3: Filter based on requirements
├── step4_llm_recommend_json.py         # Step 4: Generate final recommendations
├── prompt/                             # Prompt templates
├── paper/                              # PDF handbooks
├── requirements_extractor.py           # Requirements extraction tools
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key:**
   The system uses a pre-configured API key. If you need to change it, update the `OPENAI_API_KEY` variable in the file `./alloy_recommendation_system.py`.

3. **Prepare PDF Files:**
   Place your material handbook PDFs in the `paper/` folder.

## Usage

Save the user requirements in the file `./user_requirements.txt`. Please strictly write the requirements that are not related to the value in the primary requirements, and the numerical requirements (such as density, YS) in the secondary requirements. An example is given in the file `./user_requirements.txt`.

Then run the main system:
```bash
python alloy_recommendation_system.py
```

The system will:
1. Prompt you for your alloy requirements
2. Recommend appropriate alloy families
3. Process relevant PDF files
4. Generate detailed recommendations

## Output Files

- `alloy_families_recommendation.txt`: Initial alloy family recommendations
- `final_recommendations.txt`: Detailed final recommendations
- `filtered_output.json`: Filtered alloy data
- `output.json`: Extracted alloy data

## Notes

- All intermediate files are preserved for debugging and analysis