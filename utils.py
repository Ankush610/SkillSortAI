import os
import torch 
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
from accelerate import Accelerator
from docx import Document
import pymupdf

# Load environment variables
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

# Authenticate with Hugging Face Hub
login(token=HUGGINGFACE_TOKEN)

# Define model ID
model_id = "akjindal53244/Llama-3.1-Storm-8B"

# Initialize Accelerator to handle multi-GPU setup
accelerator = Accelerator(mixed_precision=None)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map="auto"
)

# Move model to the proper device
model = accelerator.prepare(model)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)


def extract_resume(file_path):

    """Extracts text from a resume file (.pdf using PyMuPDF, .docx, or .txt)."""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = pymupdf.open(file_path)  # open a document
        text = ""
        for page in doc:  # iterate the document pages
            text += page.get_text()  # get plain text encoded as UTF-8
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file format: {ext}")
    

def askAI(text, requirements):

    """Main function which calls LLM and generates output for user query"""

    outputs = pipe(
        [{"role": "system", "content": f"""
                                        You are an Applicant Tracking System (ATS) designed to extract key candidate information from resumes. When provided with resume text, you will analyze it and extract specific fields in a consistent format.

                                        Your task is to:
                                        1. Extract the candidate's information accurately
                                        2. Compare the candidate's qualifications against provided job requirements
                                        3. Calculate specific scoring metrics
                                        4. Output the results in a specific format

                                        For each resume analysis, extract the following fields:
                                        - Name: The candidate's full name
                                        - Email: Valid email address
                                        - Phone: Phone number (may include country code)
                                        - Address: Physical location information
                                        - Professional Experience: Format as "Company:Years, Company:Years" (e.g., "Google:3, Meta:2")
                                        - Highest Education: Most advanced degree or educational qualification
                                        - Skills: All relevant skills mentioned in the resume
                                        - Skills Score: A number from 0-10 reflecting how well the candidate's skills match the job requirements
                                        - Experience Score: A number from 0-10 based on how closely the candidate's total experience matches the required experience
                                        - Behavior Index: A number from 0-10 evaluating employment stability (higher scores for longer tenures, lower scores for job changes under 2 years)


                                        SCORING GUIDELINES:
                                        1. Skills Score: Compare candidate skills with required skills, rating compatibility from 0-10
                                        2. Experience Score: Calculate based on match between total experience and required experience even higher experience than required is better, score from 0-10
                                        3. Behavior Index: Evaluate job stability - lower scores for frequent job changes (<2 years), higher scores for longer tenures, score from 0-10


                                        FORMAT RULES:
                                        1. Use "NA" for any field where information is not found in the resume
                                        2. Output ONLY the formatted fields with no explanations or additional text
                                        3. Follow this exact format:
                                    
                                        Name: [Value]
                                        Email: [Value]
                                        Phone: [Value]
                                        Address: [Value]
                                        Professional Experience: [Value]
                                        Highest Education: [Value]
                                        Skills: [Value]
                                        Skills Score: [Value]
                                        Experience Score: [Value]
                                        Behavior Index: [Value]
          """},
         {"role": "user", "content": f"Job Requirements: {requirements} \nResume Text: {text}\n"}],
        max_new_tokens=1000,
        temperature=0.01,  
        do_sample=False,   
        top_p=1.0,        
        top_k=0,           
    )
    
    assistant_response = outputs[0]["generated_text"][-1]["content"]

    return assistant_response


def process_resumes(requirements, resume_dir):

    """Process the Output Generated by LLM and Sotred it in List of Dictonaries"""

    # Initialize list to store all the extracted data
    extracted_data_list = []

    # Process all resumes in the directory
    for filename in os.listdir(resume_dir):
        resume_path = os.path.join(resume_dir, filename)  
        resume_text = extract_resume(resume_path)
        extracted_data = askAI(resume_text, requirements)

        # Parse the extracted data and convert it into a structured dictionary
        resume_info = {
            "filename":filename,
            "Name": "NA",
            "Skills Score": "NA",
            "Experience Score": "NA",
            "Behavior Index": "NA",
            "Overall Score": "NA",
            "Email": "NA",
            "Phone": "NA",
            "Address": "NA",
            "Professional Experience": "NA",
            "Highest Education": "NA",
            "Skills": "NA",
        }

        # Extract the values from the generated response
        lines = extracted_data.split("\n")
        for line in lines:
            for key in resume_info.keys():
                if line.startswith(f"{key}:"):
                    resume_info[key] = line.split(":", 1)[1].strip()
                    break

        # Calculate the Overall Score based on the three score fields
        try:
            skills_score = float(resume_info["Skills Score"])
            experience_score = float(resume_info["Experience Score"])
            behavior_index = float(resume_info["Behavior Index"])
            
            # Calculate overall score (sum of scores multiplied by 3.33333 to get equivalent to 100)
            overall_score = (skills_score + experience_score + behavior_index) * (100/30)
            
            # Round to 2 decimal places
            resume_info["Overall Score"] = f"{overall_score:.2f}"
        except (ValueError, TypeError):
            # Handle case where scores are not valid numbers
            resume_info["Overall Score"] = "NA"
        
        # Append the structured data to the list
        extracted_data_list.append(resume_info)
    
    return extracted_data_list

def save_resumes_to_excel(extracted_data_list, output_file="resume_analysis.xlsx"):
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(extracted_data_list)
    
    # Convert score columns to numeric for proper sorting
    numeric_columns = ["Skills Score", "Experience Score", "Behavior Index", "Overall Score"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by Overall Score in descending order
    df_sorted = df.sort_values(by="Overall Score", ascending=False)
    
    # Save the sorted DataFrame to an Excel file
    df_sorted.to_excel(output_file, index=False)
    
    print(f"Resume data saved to {output_file} sorted by Overall Score (highest to lowest)")
    
    return df_sorted


if __name__=="__main__":

    requirements = input("Enter job requirements (skills, experience, etc.): ")
    resume_dir = input("Enter the path to resume dir : ").strip(" ")
    output = process_resumes(requirements, resume_dir)
    save_resumes_to_excel(output)
