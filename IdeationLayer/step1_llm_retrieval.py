import os
import openai
import pymupdf
from tqdm import tqdm

OPENAI_API_KEY = "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"

def read_pdf(file_path):
    text = ""
    doc = pymupdf.open(file_path) # open a document
    for page in doc: # iterate the document pages
        text += page.get_text() # get plain text encoded as UTF-8
    return text

def ask_gpt(pdf_content, prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-4o-mini"

    all_prompt = f"Here is the PDF content:\n\n{pdf_content}\n\n{prompt}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an assistant that reads and answers questions based on PDF content."},
            {"role": "user", "content": all_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_prompt(file_path):
    with open(file_path, "r") as file:
        template = file.read()
    return template

def run_retrieval(pdf_folder_path="./paper", answer_folder_path="./answer", prompt_path="./prompt/retrieval.prompt"):
    """
    Run the retrieval process to extract information from PDF files.
    
    Args:
        pdf_folder_path (str): Path to folder containing PDF files
        answer_folder_path (str): Path to folder where answers will be saved
        prompt_path (str): Path to the retrieval prompt file
    """
    print("Step 2: Running LLM Retrieval...")
    
    # Create answer folder if it doesn't exist
    os.makedirs(answer_folder_path, exist_ok=True)
    
    retrieval_prompt = get_prompt(prompt_path)
    
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder_path, filename)
        pdf_content = read_pdf(pdf_path)
        answer = ask_gpt(pdf_content, retrieval_prompt)
        answer_path = os.path.join(answer_folder_path, filename[:-4] + ".answer")
        with open(answer_path, "w", encoding="utf-8") as file:
            file.write(answer)
    
    print(f"Retrieval completed. Answers saved to {answer_folder_path}")
    return answer_folder_path

if __name__ == "__main__":
    run_retrieval() 