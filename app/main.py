from fastapi import FastAPI
from app.router import evaluation
import os

# Initialize FastAPI app
# root_path helps FastAPI generate correct docs URLs when behind a proxy like /axis_AI
app = FastAPI(
    title="BIMS AI Inspector",
    version="1.0.0",
    root_path=os.getenv("ROOT_PATH", "") 
)

# Include the evaluation router
# This makes /evaluate and /health available
app.include_router(evaluation.router)