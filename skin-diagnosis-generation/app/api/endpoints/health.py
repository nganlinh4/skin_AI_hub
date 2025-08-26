"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
import torch
import time
from datetime import datetime

from app.models.response import HealthResponse, StatusResponse
from app.services.diagnosis_service import DiagnosisService
from app.api.dependencies import get_diagnosis_service
from app.config import settings

router = APIRouter()

# Service start time for uptime calculation
start_time = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    try:
        # Try to get the diagnosis service to check if it's initialized
        diagnosis_service = get_diagnosis_service()
        model_loaded = diagnosis_service.is_initialized
        status = "healthy" if model_loaded else "initializing"
    except Exception:
        # Service not available yet
        model_loaded = False
        status = "initializing"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        cuda_available=torch.cuda.is_available(),
        timestamp=datetime.now().isoformat()
    )

@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check - ensures service is ready to handle requests"""
    try:
        # Check if the diagnosis service is properly initialized
        diagnosis_service = get_diagnosis_service()
        model_loaded = diagnosis_service.is_initialized
        status = "ready" if model_loaded else "not_ready"
    except Exception:
        # Service not available yet
        model_loaded = False
        status = "not_ready"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        cuda_available=torch.cuda.is_available(),
        timestamp=datetime.now().isoformat()
    )

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Detailed system status"""
    
    # GPU information
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory
        }
    
    # Memory usage (simplified)
    memory_usage = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory_used": torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    }
    
    return StatusResponse(
        service="MedGemma 4B Skin Diagnosis API",
        version="1.0.0",
        model_type=settings.MODEL_TYPE,
        cuda_available=torch.cuda.is_available(),
        gpu_info=gpu_info,
        memory_usage=memory_usage,
        uptime=time.time() - start_time
    )
