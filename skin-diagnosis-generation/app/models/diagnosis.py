"""
Diagnosis models for the VLM server API.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class DiagnosisRequest(BaseModel):
    """Diagnosis request model"""
    patient_name: str
    patient_age: int
    patient_gender: str
    classification: Optional[str] = None
    history: Optional[str] = None
    language: str = "en"


class AnalysisResult(BaseModel):
    """Analysis result model"""
    concise_analysis: str
    comprehensive_report: str
    generation_time: float


class DiagnosisResponse(BaseModel):
    """Diagnosis response model"""
    success: bool
    patient_name: str
    analysis: AnalysisResult
    files: Dict[str, str]
