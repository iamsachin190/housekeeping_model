import time
import json
import os
from datetime import datetime
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.models import InspectionResult
from app.services import image_service, rag_service, llm_service
from app.config import settings

router = APIRouter()

def save_dataset_record(image_path: str, result: dict):
    """Background task to save data for fine-tuning."""
    record_id = os.path.basename(image_path).replace(".jpg", "")
    json_path = os.path.join(settings.DATASET_DIR, f"{record_id}_label.json")
    
    record = {
        "image_path": image_path,
        "timestamp": datetime.utcnow().isoformat(),
        "ai_output": result,
        "verified": False
    }
    
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

@router.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@router.post("/evaluate", response_model=InspectionResult)
async def evaluate_cleanliness(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    start_time = time.time()
    
    # 1. Stitch
    try:
        image_bytes_list = [await file.read() for file in files]
        stitched_image = image_service.stitch_images(image_bytes_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    # 2. Save Locally
    saved_path = image_service.save_image_locally(stitched_image, prefix="eval")
    
    # 3. RAG Context
    rag_context = rag_service.retrieve_similar_context(stitched_image)
    
    # 4. AI Analysis
    try:
        result_json = await llm_service.analyze_image_with_failover(saved_path, rag_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")
    
    # 5. Background Save
    background_tasks.add_task(save_dataset_record, saved_path, result_json)
    
    return result_json
