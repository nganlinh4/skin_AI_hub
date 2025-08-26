"""
FastAPI application for MedGemma 4B Skin Diagnosis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from contextlib import asynccontextmanager

from app.api.endpoints import diagnosis, health
from app.api.dependencies import set_diagnosis_service
from app.services.diagnosis_service import DiagnosisService
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global diagnosis service instance
diagnosis_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global diagnosis_service
    
    # Startup
    logger.info("Starting MedGemma Diagnosis API...")
    try:
        diagnosis_service = DiagnosisService()
        await diagnosis_service.initialize()
        set_diagnosis_service(diagnosis_service)
        logger.info("Diagnosis service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize diagnosis service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MedGemma Diagnosis API...")
    if diagnosis_service:
        await diagnosis_service.cleanup()

# Create FastAPI application
app = FastAPI(
    title="MedGemma 4B Skin Diagnosis API",
    description="AI-powered medical image analysis and diagnosis report generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(diagnosis.router, prefix="/api/v1", tags=["diagnosis"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MedGemma 4B Skin Diagnosis API",
        "version": "1.0.0",
        "docs": "/docs"
    }


