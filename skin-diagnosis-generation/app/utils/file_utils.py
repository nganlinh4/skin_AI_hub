"""
File handling utilities
"""

import os
import aiofiles
from fastapi import UploadFile, HTTPException
from pathlib import Path
from typing import List
import mimetypes

from app.config import settings

def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If file is invalid
    """
    
    # Check file size
    if hasattr(file, 'size') and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Check file extension
    if file.filename:
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
    else:
        raise HTTPException(status_code=400, detail="No filename provided")

async def save_uploaded_file(file: UploadFile, target_filename: str) -> str:
    """
    Save uploaded file to disk
    
    Args:
        file: Uploaded file object
        target_filename: Target filename to save as
        
    Returns:
        Path to saved file
    """
    
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Create full file path
    file_path = os.path.join(settings.UPLOAD_DIR, target_filename)
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return file_path
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def cleanup_file(file_path: str) -> None:
    """
    Safely remove a file
    
    Args:
        file_path: Path to file to remove
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        # Ignore cleanup errors
        pass

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "extension": Path(file_path).suffix.lower()
    }
