from dotenv import load_dotenv
load_dotenv()
import os
from datetime import datetime
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Annotated, Dict, Any
from medical_models import alzheimer_model, chest_model, brain_mri_model
from langchain.prompts import ChatPromptTemplate

# Clinical Report Templates
CLINICAL_REPORT_SYSTEM_PROMPT = """You are an expert medical professional specializing in medical imaging and diagnostics.
Your task is to generate a detailed clinical report based on AI model analysis results.

Follow this structured report format:
===============================
CLINICAL DIAGNOSTIC REPORT
===============================
Date: [Current Date]
Examination: [Exam Type]
AI Model: [Model Used]

CLINICAL INDICATION
------------------
[Brief description of the examination purpose]

TECHNIQUE
---------
[Description of imaging technique and AI analysis method]

FINDINGS
--------
[Detailed observations in medical terminology]
- Primary diagnosis
- Confidence level
- Key observations
- Notable patterns

IMPRESSION
----------
[Clinical interpretation and recommendations]
1. Primary diagnosis with confidence level
2. Clinical significance
3. Recommended follow-up if needed

NOTES
-----
[Additional relevant information or cautionary notes]
===============================

Use professional medical terminology while maintaining clarity.
Be precise and clinical in your language.
Include relevant statistical confidence levels.
Make specific follow-up recommendations when warranted."""

def create_clinical_prompt(exam_type: str, model_result: Dict[str, Any]) -> str:
    """Create a specific clinical prompt based on exam type and results"""
    
    exam_prompts = {
        "brain": """Focus on neurological findings:
- Tumor characteristics if present
- Mass effect or midline shift
- Regional brain anatomy involvement
- Contrast enhancement patterns
Include differential diagnosis if applicable.""",
        
        "alzheimer": """Focus on cognitive impairment indicators:
- Degree of atrophy if present
- Hippocampal volume assessment
- White matter changes
- Vascular components
Compare with age-appropriate norms.""",
        
        "chest": """Focus on thoracic findings:
- Lung field characteristics
- Mediastinal assessment
- Cardiac silhouette
- Pleural spaces
Note any suspicious masses or infiltrates."""
    }
    
    return f"""Based on the following AI analysis results, generate a detailed clinical report:

ANALYSIS RESULTS:
----------------
Prediction: {model_result['prediction']}
Confidence: {model_result['confidence']*100:.2f}%

Detailed Probabilities:
{chr(10).join([f'- {k}: {v*100:.2f}%' for k, v in model_result['probabilities'].items()])}

{exam_prompts.get(exam_type, "")}"""


def generate_clinical_report(exam_type: str, image_path: str) -> Dict[str, Any]:
    """
    Generate a clinical-style report for a specific type of medical examination
    
    Args:
        exam_type: One of 'brain', 'alzheimer', or 'chest'
        image_path: Path to the medical image
        
    Returns:
        Dictionary containing the analysis results and clinical report
    """
    try:
        # Select appropriate model based on exam type
        model_map = {
            'brain': brain_mri_model,
            'alzheimer': alzheimer_model,
            'chest': chest_model
        }
        
        if exam_type not in model_map:
            raise ValueError(f"Unsupported examination type: {exam_type}")
            
        # Get model prediction
        model_result = model_map[exam_type](image_path)
        
        # Create the clinical prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_REPORT_SYSTEM_PROMPT),
            ("human", create_clinical_prompt(exam_type, model_result))
        ])
        
        # Generate clinical report
        llm = ChatNVIDIA(model="openai/gpt-oss-120b")
        messages = prompt.format_messages()
        response = llm.invoke(messages)
        
        # Determine result color based on prediction confidence
        confidence = model_result['confidence']
        if confidence >= 0.9:
            result_color = "green"
        elif confidence >= 0.7:
            result_color = "blue"
        else:
            result_color = "yellow"
        
        return {
            "status": "success",
            "examination_type": exam_type,
            "model_result": model_result,
            "clinical_report": response.content,
            "analysis_points": [
                {
                    "color": result_color,
                    "text": f"{exam_type.title()} Analysis: {model_result['prediction']} ({model_result['confidence']*100:.1f}%)"
                }
            ],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "examination_type": exam_type,
            "error": str(e),
            "analysis_points": [
                {
                    "color": "red",
                    "text": f"Error in {exam_type} analysis: {str(e)}"
                }
            ],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Test functionality
if __name__ == "__main__":
    test_image = r"E:\virtualHospital\MODELS\Brain\image3.png"
    
    # Test each type of analysis
    exam_types = ['brain', 'alzheimer', 'chest']

    result = generate_clinical_report('brain', test_image)
    
    print("Clinical Report Result:",result)




