"""
API dependencies
"""

from fastapi import HTTPException
from app.services.diagnosis_service import DiagnosisService

# Global diagnosis service instance (will be set by main.py)
_diagnosis_service: DiagnosisService = None

def set_diagnosis_service(service: DiagnosisService):
    """Set the global diagnosis service instance"""
    global _diagnosis_service
    _diagnosis_service = service

def get_diagnosis_service() -> DiagnosisService:
    """Get the global diagnosis service instance"""
    if _diagnosis_service is None:
        raise HTTPException(status_code=503, detail="Diagnosis service not available")
    return _diagnosis_service
