"""
Response models for the VLM server API.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    cuda_available: bool
    timestamp: str


class StatusResponse(BaseModel):
    """Detailed system status response model"""
    service: str
    version: str
    model_type: str
    cuda_available: bool
    gpu_info: Optional[Dict[str, Any]] = None
    memory_usage: Dict[str, Any]
    uptime: float


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: Optional[str] = None
