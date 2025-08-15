import json
import os
import re
import pandas as pd

def get_json(text):
    json_match = re.search(r'```json\n.*\n```', text, re.DOTALL)
    start = 8
    if json_match:
        try:
            # Load the JSON data from the matched string
            extracted_json = json.loads(json_match.group(0)[start:-4])
            return extracted_json
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return None

def run_json_extraction(answer_folder_path="./answer", output_path="./output.json"):
    """
    Extract JSON data from answer files and save to output file.
    
    Args:
        answer_folder_path (str): Path to folder containing answer files
        output_path (str): Path where the extracted JSON will be saved
    """
    print("Step 3: Running JSON Extraction...")
    
    extracted_jsons = []
    answer_files = [f for f in os.listdir(answer_folder_path) if f.endswith(".answer")]
    print(f"Found {len(answer_files)} answer files to process")
    
    for filename in answer_files:
        answer_path = os.path.join(answer_folder_path, filename)
        with open(answer_path, "r", encoding="utf-8") as file:
            text = file.read()
        extracted_json = get_json(text)
        if extracted_json is not None:
            if isinstance(extracted_json, list):
                extracted_jsons.extend(extracted_json)
            elif isinstance(extracted_json, dict):
                if any("alloy" in key.lower() for key in extracted_json):
                    for key in extracted_json:
                        if "alloy" in key.lower():
                            if isinstance(extracted_json[key], dict):
                                extracted_jsons.append(extracted_json[key])
                            elif isinstance(extracted_json[key], list):
                                extracted_jsons.extend(extracted_json[key])
                extracted_jsons.append(extracted_json)
    
    extracted_jsons_new = []
    key_set = ['Component', 'Density', 'Yield']
    for extracted_json in extracted_jsons:
        print(extracted_json)
        if all(any(s.lower() in key.lower() for key in extracted_json) for s in key_set):
            temp_dict = {}
            for new_key in key_set:
                for key in extracted_json:
                    if new_key.lower() in key.lower():
                        temp_dict[new_key] = extracted_json[key]
            if None not in temp_dict.values():
                extracted_jsons_new.append(temp_dict)
    
    with open(output_path, "w") as file:
        json.dump(extracted_jsons_new, file, indent=4)
    
    print(f"JSON extraction completed. Output saved to {output_path}")
    return output_path

if __name__ == "__main__":
    run_json_extraction() 