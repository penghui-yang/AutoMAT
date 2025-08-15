import os
import json
import openai
from step1_llm_retrieval import run_retrieval
from step2_json_extractor_recommend import run_json_extraction
from step3_llm_filter_json import run_filter_json
from step4_llm_recommend_json import run_recommend_json
from requirements_extractor import split_requirements
import re

# TODO: Change to your own OpenAI API key
OPENAI_API_KEY = "**YOUR_OPENAI_API_KEY**"

def ask_gpt(requirements, prompt):
    """Ask GPT for alloy family recommendations based on requirements."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-4o"
    
    all_prompt = f"{prompt}\n{requirements}\n"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a material scientist expert in titanium alloys."},
            {"role": "user", "content": all_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_prompt(file_path):
    """Read prompt from file."""
    with open(file_path, "r") as file:
        return file.read()

def run_alloy_recommendation_system(user_requirements):
    """
    Main function to run the complete alloy recommendation system.
    
    Args:
        user_requirements (str): User's requirements for alloy selection
    """
    print("=" * 60)
    print("ALLOY RECOMMENDATION SYSTEM")
    print("=" * 60)
    print(f"User Requirements: {user_requirements}")
    print("=" * 60)

    requirements_split = split_requirements(user_requirements)
    
    # Recommend alloy families based on requirements
    print("\nStep 1: Recommending alloy families based on requirements...")
    # New prompt for general alloy family recommendation
    family_prompt = (
        "Given the following user requirements, recommend the most suitable main alloy family (such as Ti, Al, Fe, Ni, Mg, HEA, etc.). "
        "Only provide the name of the main alloy family (element symbol or name) and a one-sentence justification. Do not recommend sub-families or specific alloys.\n"
    )
    alloy_families_response = ask_gpt(requirements_split['primary'] + requirements_split['secondary'], family_prompt)
    
    # Save the initial recommendation
    print("Alloy families recommendation response:")
    print(alloy_families_response)
    with open("./alloy_families_recommendation.txt", "w", encoding="utf-8") as f:
        f.write(alloy_families_response)
    print("Alloy families recommendation saved to ./alloy_families_recommendation.txt")
    
    # Step 2: Extract main alloy family (element symbol or name) from the response
    alloy_families = []
    # Try to extract element symbol (e.g., Ti, Al, Fe, Ni, Cu, Mg) from the response
    match = re.search(r"\b(Ti|Al|Fe|Ni|Mg|Zr|Mo|Co|Cr|Sn|V|Nb|Ta|W|Mn|Si|Zn|Li|Ag|Au|Pd|Pt|HEA)\b", alloy_families_response, re.IGNORECASE)
    if match:
        alloy_families.append(match.group(1))
    else:
        # Fallback: take the first word (in case the model outputs e.g. 'Titanium: ...')
        first_word = alloy_families_response.strip().split()[0]
        alloy_families.append(first_word)

    print(f"Extracted alloy families: {alloy_families}")
    
    # Check if subfolder exists under ./paper for the alloy family
    paper_folder = os.path.join("./paper", alloy_families[0])
    while not os.path.isdir(paper_folder):
        print(f"\nNo subfolder named '{alloy_families[0]}' found under ./paper.")
        print(f"Please create the folder './paper/{alloy_families[0]}' and put the relevant handbook or papers inside.")
        input("Once done, press ENTER to retry...")
        # Re-check if the folder now exists

    # Step 3: Run retrieval on relevant papers
    print("\n" + "=" * 40)
    run_retrieval(pdf_folder_path=f"./paper/{alloy_families[0]}", answer_folder_path="./answer")
    
    # Step 4: Extract JSON from answers
    print("\n" + "=" * 40)
    run_json_extraction(answer_folder_path="./answer", output_path="./output/output.json")
    
    # Step 5: Filter JSON based on requirements
    print("\n" + "=" * 40)
    run_filter_json(input_path="./output/output.json", 
                   answer_folder_path="./answer_json",
                   filtered_output_path="./output/filtered_output.json",
                   requirements=requirements_split['primary'])
    
    # Step 6: Generate final recommendations
    print("\n" + "=" * 40)
    run_recommend_json(filtered_input_path="./output/filtered_output.json",
                      output_path="./output/final_recommendations.txt",
                      requirements=requirements_split['secondary'])
    
    print("\n" + "=" * 60)
    print("SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Output files generated:")
    print("- ./output/alloy_families_recommendation.txt: Initial alloy family recommendations")
    print("- ./output/final_recommendations.txt: Final detailed recommendations")
    print("- ./output/filtered_output.json: Filtered alloy data")
    print("- ./output/output.json: Extracted alloy data")
    print("=" * 60)

def main():
    """Main function to get user input and run the system."""
    print("Welcome to the Alloy Recommendation System!")
    print("Please enter your requirements for alloy selection:")
    print("- If your requirements fit on one line, type them and press Enter.")
    print("- If you want to provide multi-line or detailed requirements, create a file named 'user_requirements.txt' in the current directory, write your requirements there, and then just press Enter without typing anything.")
    print("(e.g., 'I need a titanium alloy with high strength, low density, and good corrosion resistance')")
    
    user_input = input("\nEnter your requirements (or press Enter to use user_requirements.txt): ")
    
    if user_input.strip():
        user_requirements = user_input.strip()
    else:
        if os.path.exists("user_requirements.txt"):
            with open("user_requirements.txt", "r", encoding="utf-8") as f:
                user_requirements = f.read().strip()
            if not user_requirements:
                print("user_requirements.txt is empty. Using default requirements...")
                user_requirements = "I need a titanium alloy with high strength (>1000 MPa), low density (<3.8 g/cm³), and good corrosion resistance. The alloy should be cost-effective with at least 4 elements and avoid expensive elements like Zr and Mo."
        else:
            print("No requirements provided and user_requirements.txt not found. Using default requirements...")
            user_requirements = "I need a titanium alloy with high strength (>1000 MPa), low density (<3.8 g/cm³), and good corrosion resistance. The alloy should be cost-effective with at least 4 elements and avoid expensive elements like Zr and Mo."
    
    run_alloy_recommendation_system(user_requirements)

if __name__ == "__main__":
    main() 
