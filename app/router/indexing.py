import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from app.models import CleanlinessStatus
from app.services import image_service, rag_service

router = APIRouter()

@router.post("/index")
async def index_reference_image(
    file: UploadFile = File(...),
    status: CleanlinessStatus = Form(...),
    description: str = Form(...)
):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Save locally
        saved_path = image_service.save_image_locally(image, prefix=f"ref_{status.value}")
        
        # Add to ChromaDB
        rag_service.add_to_index(saved_path, status.value, description)
        
        return {"message": "Image indexed successfully", "id": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
