from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import tempfile
from ai_models import generate_clinical_report
from doctor import Doctor

app = FastAPI()
doctor = Doctor()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-scan")
async def upload_scan(
    image: UploadFile = File(...),
    filename: str = Form(...),
    exam_type: str = Form(...),
    question: str = Form(None)  # Optional question for the doctor
):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        shutil.copyfileobj(image.file, temp_file)
        temp_path = temp_file.name

    try:
        # Store the report using Doctor class's store_report method
        store_result = doctor.store_report(exam_type, temp_path)
        result = store_result["report"]
        session_id = store_result["report_id"]

        # Get doctor's analysis if a question was provided
        doctor_response = None
        if question:
            doctor_response = doctor.get_ai_response(
                report_id=session_id,
                question=question
            )

        if result["status"] == "success":
            # Format the complete clinical report
            response = {
                "status": "success",
                "session_id": session_id,
                "clinical_report": result["clinical_report"],
                "detailed_findings": result.get("model_result", {}).get("prediction", ""),
                "probabilities": result.get("model_result", {}).get("probabilities", {}),
                "recommendations": result.get("recommendations", ""),
                "doctor_response": doctor_response,  # Include doctor's response if available
                "analysis_points": [
                    {
                        "color": "blue",
                        "text": f"Primary Diagnosis: {result.get('model_result', {}).get('prediction', 'Unknown')}"
                    },
                    {
                        "color": "green",
                        "text": f"Confidence: {result.get('model_result', {}).get('confidence', 0)*100:.1f}%"
                    },
                    *[
                        {
                            "color": "gray",
                            "text": f"{class_name}: {prob*100:.1f}%"
                        }
                        for class_name, prob in result.get("model_result", {}).get("probabilities", {}).items()
                    ]
                ]
            }
            return response
        else:
            return {
                "status": "error",
                "error": result["error"]
            }
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/doctor-consultation")
async def doctor_consultation(
    session_id: str = Form(...),
    user_message: str = Form(...)
):
    try:
        print(f"Received consultation request: session={session_id}")
        print(f"Message: {user_message}")
        
        # Get doctor's response
        response = doctor.get_ai_response(
            report_id=session_id,
            question=user_message
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "doctor_response": response
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}