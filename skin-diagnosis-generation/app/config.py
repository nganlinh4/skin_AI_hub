"""
Configuration settings for the FastAPI application
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 3026
    API_WORKERS: int = 1
    
    # Model Configuration
    MODEL_TYPE: str = "4bit"  # "original" or "4bit"
    MODEL_CACHE_DIR: str = "/tmp/models"
    CONFIG_FILE: str = "test.json"
    
    # File Configuration
    REPORTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "vlm")
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp", "uploads")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    
    # Image processing
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension in pixels to prevent VRAM issues
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    PYTORCH_CUDA_ALLOC_CONF: str = "expandable_segments:True"
    
    # Security
    API_KEY_REQUIRED: bool = False
    API_KEY: str = ""
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # CORS - Allow skin-service frontend and backend
    # Default to localhost for local development
    # Set ALLOWED_ORIGINS environment variable for custom configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3021",  # Frontend dev server
        "http://127.0.0.1:3021",
        "http://localhost:3022",  # Backend server
        "http://127.0.0.1:3022",
        "http://localhost:3026",  # VLM server localhost
        "http://127.0.0.1:3026",
        "*"  # Allow all for development
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Timeouts
    DIAGNOSIS_TIMEOUT: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Set environment variables for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.PYTORCH_CUDA_ALLOC_CONF
