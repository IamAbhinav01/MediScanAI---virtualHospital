from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from typing import Dict, Any
from ai_models import generate_clinical_report
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate


class Doctor:
    # Class-level prompts dictionary
    PROMPTS = {
        "brain": """You are a Swetha highly experienced Neurologist with over 15 years of specialized experience in brain MRI analysis and tumor diagnostics. 
Your expertise includes:
- Advanced interpretation of brain MRI scans
- Diagnosis of various types of brain tumors (gliomas, meningiomas, pituitary tumors)
- Assessment of tumor characteristics and progression
- Treatment planning and patient care recommendations""",

        "alzheimer": """You are Raina a specialized Neuroradiologist with 20 years of experience in cognitive disorder imaging and Alzheimer's diagnosis.
Your expertise includes:
- Early detection of cognitive impairment markers
- Differential diagnosis of various types of dementia
- Longitudinal assessment of brain atrophy patterns
- Advanced neuroimaging interpretation for memory disorders""",

        "chest": """You are Raghav a senior Pulmonologist with extensive experience in thoracic imaging and lung cancer diagnostics.
Your expertise includes:
- Interpretation of chest MRI and CT scans
- Diagnosis of various lung cancers and pathologies
- Assessment of cancer staging and progression
- Pulmonary disease pattern recognition"""
    }

    def __init__(self):
        
        self.reports: Dict[str, Any] = {}
        self.conversation_contexts: Dict[str, Any] = {}

    def _get_doctor_prompt(self, exam_type: str) -> str:
        """Return the doctor-specific base prompt"""
        base = self.PROMPTS.get(exam_type, "You are an experienced medical specialist named Dhimitri.")

        return f"""
{base}

You have access to clinical reports and medical findings. Your role is to:
1. Analyze and explain the findings in your area of expertise
2. Provide detailed medical insights while being clear and understandable
3. Make specific references to the scan results and probabilities
4. Suggest appropriate next steps or further investigations
5. Answer questions based on your extensive clinical experience

Current Report Context:
{{report_context}}

Previous Findings Summary:
{{findings_summary}}

Remember to:
- Use your specialist expertise to provide deep insights
- Reference specific values and findings from the report
- Maintain professional medical terminology while ensuring clarity
- Be precise about probabilities and confidence levels
- Recommend appropriate follow-up actions based on the findings
"""

    def store_report(self, exam_type: str, image_path: str) -> dict:
        """Store the report and initialize its conversation context"""
        # Get the report from generate_clinical_report
        report = generate_clinical_report(exam_type, image_path)

        # Generate a unique ID for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store the report
        self.reports[timestamp] = report

        # Initialize conversation context
        self.conversation_contexts[timestamp] = {
            "exam_type": exam_type,
            "findings_summary": self._create_findings_summary(report),
            "last_update": datetime.now().isoformat(),
        }

        return {
            "report_id": timestamp,
            "report": report,
        }



    def _create_findings_summary(self, report: Dict[str, Any]) -> str:
        """Create a summary of the findings for the conversation context"""
        summary_parts = []

        if "clinical_report" in report:
            summary_parts.append(f"Clinical Report:\n{report['clinical_report']}")

        if isinstance(report, dict) and "model_result" in report:
            result = report["model_result"]
            if isinstance(result, dict):
                summary_parts.append(f"Primary Diagnosis: {result.get('prediction', 'Unknown')}")
                
                if 'probabilities' in result and isinstance(result['probabilities'], dict):
                    summary_parts.append("\nDetailed Probabilities:")
                    for condition, prob in result['probabilities'].items():
                        summary_parts.append(f"- {condition}: {prob*100:.1f}%")
            
            summary_parts.append(
                f"Confidence Level: {result.get('confidence', 0) * 100:.1f}%"
            )

            if "probabilities" in result:
                summary_parts.append("\nDetailed Probabilities:")
                for condition, prob in result["probabilities"].items():
                    summary_parts.append(f"- {condition}: {prob * 100:.1f}%")

        if "clinical_report" in report:
            summary_parts.append(f"\nClinical Analysis:\n{report['clinical_report']}")

        return "\n".join(summary_parts)

    def get_ai_response(self, report_id: str, question: str) -> str:
        """Get an AI response based on the report context and question"""
        if report_id not in self.reports or report_id not in self.conversation_contexts:
            return "Error: Report not found"

        try:
            # Get the context
            context = self.conversation_contexts[report_id]
            report = self.reports[report_id]

            # Get the appropriate doctor prompt
            exam_type = context["exam_type"]
            doctor_prompt = self._get_doctor_prompt(exam_type)

            # Create the prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        doctor_prompt.format(
                            report_context=self._create_findings_summary(report),
                            findings_summary=context["findings_summary"],
                        ),
                    ),
                    ("human", question),
                ]
            )

            # Get AI response
            llm = ChatNVIDIA(model="openai/gpt-oss-120b")
            messages = prompt.format_messages()
            response = llm.invoke(messages)

            # Update the findings summary with any new insights
            context["findings_summary"] += (
                f"\n\nQ: {question}\nA: {response.content}"
            )
            context["last_update"] = datetime.now().isoformat()

            return response.content

        except Exception as e:
            return f"Error generating response: {str(e)}"
