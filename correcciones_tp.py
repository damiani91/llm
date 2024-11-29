import os
from dotenv import load_dotenv
from openai import OpenAI
import json

def process_notebook(file_path):
    """
    Processes a .ipynb file and extracts its content in a readable format.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    result = []
    
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            result.append("### Markdown:\n" + "\n".join(cell['source']))
        elif cell['cell_type'] == 'code':
            result.append("### Code:\n" + "\n".join(cell['source']))
    
    return "\n\n".join(result)

def tp_correction(openai, model, system_prompt, user_prompt):
    """
    Sends a correction request to OpenAI and retrieves the result.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

def process_all_notebooks(path, openai, model, system_prompt, output_file):
    """
    Iterates through all .ipynb files in a directory, processes each one, 
    and accumulates the results in a final JSON file.
    
    :param path: Directory path containing .ipynb files.
    :param openai: OpenAI API object for processing.
    :param model: Model to use with OpenAI (e.g., "gpt-4").
    :param system_prompt: System prompt for processing.
    :param output_file: Name of the JSON file to store all results.
    """
    final_results = []  # List to store results

    # Iterate through all files in the directory
    for filename in os.listdir(path):
        if filename.endswith('.ipynb'):  # Filter only .ipynb files
            file_path = os.path.join(path, filename)
            
            # Process the .ipynb file
            user_prompt = process_notebook(file_path)
            corrected_tp = tp_correction(openai, model, system_prompt, user_prompt)
            
            # Add the result to the accumulator
            final_results.append(corrected_tp)
            print(f"Processed: {filename}")

    # Save all results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"Results saved in: {output_file}")

def main():
    # Create the OpenAI object
    load_dotenv()
    openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # API key
    model = 'gpt-4o-mini'  # Selected model
    tps_path = 'path'  # Path where TPs are stored
    output_file = "corrections_name.json"  # Name of the final JSON file

    tp_correction = process_notebook('/Users/damianilkow/Desktop/Dami/Docencia/MiM/SIyT/correciones_tps/correcto.ipynb')  # Test file

    # Define our system prompt
    system_prompt = f"""
    
    You are an expert computer science and analytics professor evaluating student work. Your task is to provide a comprehensive, objective assessment that generates a structured JSON response.
        
        Evaluation Criteria:
        1. Assess the technical solution's accuracy and depth
        2. Compare the approach with existing methodologies and to this correct work {tp_correction}
        3. Determine if the work meets academic standards
        
        Grading Guidelines:
        - Approved: Solution demonstrates comprehensive understanding
        - Disapproved: Significant gaps in knowledge or execution
        Feedback Requirements:
        - Identify specific strong points in the submission
        - Highlight areas needing improvement (if applicable)

        Response Instructions:
        - Always respond in valid JSON format and all the text in the JSON in Latin Spanish
        - Ensure the response includes all specified fields
        - Be precise and academically rigorous
        - Not always be 2 students, and not always need to be two strong points and two improvements points, it could be two, three, four, or five, depending on the situation.
        - Recheck your response

        JSON Output Structure:
        {{
            "team_members": ["Student Name 1", "Student Name 2"],
            "overall_grade": "Approved or Disapproved",
            "strong_points": [
                "Specific technical strength",
                "Notable analytical approach"
            ],
            "improvement_points": [
                "Area requiring further development",
                "Potential enhancement"
            ]
        }}

        Key Evaluation Focus:
        - Technical accuracy
        - Problem-solving approach
        - Depth of understanding
        - Innovative thinking
        """
    
    process_all_notebooks(tps_path, openai, model, system_prompt, output_file)

if __name__ == "__main__":
    main()