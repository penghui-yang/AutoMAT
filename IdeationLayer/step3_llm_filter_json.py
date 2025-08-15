from copy import deepcopy
import json
import os
import openai
from tqdm import tqdm

from step2_json_extractor_recommend import get_json

OPENAI_API_KEY = "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"

def ask_gpt(retrieval_content, prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-4o"

    all_prompt = f"Here is the retrieved contents:\n\n{retrieval_content}\n\n{prompt}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "There some retrieved contents of a material handbook. You are a material scientist and help me analyze the retrieved contents."},
            {"role": "user", "content": all_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_prompt(file_path):
    with open(file_path, "r") as file:
        template = file.read()
    return template

def run_filter_json(input_path="./output.json", answer_folder_path="./answer_json", 
                   filtered_output_path="./filtered_output.json", 
                   prompt_path="./prompt/filter_json.prompt",
                   requirements=""):
    """
    Filter JSON data based on requirements and save filtered results.
    
    Args:
        input_path (str): Path to input JSON file
        answer_folder_path (str): Path to folder where filtered answers will be saved
        filtered_output_path (str): Path where filtered output will be saved
        prompt_path (str): Path to the filter prompt file
        requirements (str): User requirements
    """
    print("Step 4: Running LLM Filter JSON...")
    
    # Create answer folder if it doesn't exist
    os.makedirs(answer_folder_path, exist_ok=True)
    
    filter_prompt = get_prompt(prompt_path).format(requirements=requirements)
    
    with open(input_path, "r", encoding="utf-8") as file:  
        retrieval_contents = json.load(file)
    
    print(f"Processing {len(retrieval_contents)} items for filtering")
    
    filtered_contents = []
    for idx, retrieval_content in tqdm(enumerate(retrieval_contents), total=len(retrieval_contents), desc="Filtering items"):
        filtered_answer = ask_gpt(retrieval_content, filter_prompt)
        with open(f"./{answer_folder_path}/{idx}.answer", "w", encoding="utf-8") as file:
            file.write(filtered_answer)
        if '"Meets Requirements": "Yes"' in filtered_answer:
            filtered_answer = get_json(filtered_answer)
            filter_content = deepcopy(retrieval_content)
            if filtered_answer is not None:
                filter_content["Number of elements"] = filtered_answer["Properties"]["Number of elements"]
                filter_content["Has expensive elements or not"] = filtered_answer["Properties"]["Has expensive elements or not"]
            filtered_contents.append(filter_content)
    
    with open(filtered_output_path, "w") as file:
        json.dump(filtered_contents, file, indent=4)
    
    print(f"Filtering completed. {len(filtered_contents)} items passed filtering. Output saved to {filtered_output_path}")
    return filtered_output_path

if __name__ == "__main__":
    run_filter_json() 