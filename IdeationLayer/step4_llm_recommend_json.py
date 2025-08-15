import json
import openai

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

def run_recommend_json(filtered_input_path="./filtered_output.json", 
                      output_path="./recommend_json.answer",
                      prompt_path="./prompt/recommend_json.prompt",
                      requirements=""):
    """
    Generate final recommendations based on filtered data.
    
    Args:
        filtered_input_path (str): Path to filtered input JSON file
        output_path (str): Path where the recommendation answer will be saved
        prompt_path (str): Path to the recommendation prompt file
        requirements (str): User requirements
    """
    print("Step 5: Running LLM Recommend JSON...")
    
    with open(filtered_input_path, "r") as file:
        filtered_contents = json.load(file)
    
    recommend_prompt = get_prompt(prompt_path).format(requirements=requirements)
    recommend_answer = ask_gpt(filtered_contents, recommend_prompt)
    
    print(recommend_answer)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(recommend_answer)
    
    print(f"Recommendation completed. Output saved to {output_path}")
    return output_path

if __name__ == "__main__":
    run_recommend_json() 