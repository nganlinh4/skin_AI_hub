"""
Diagnosis endpoints
"""

import os
import time
import logging
import json
import uuid
import tempfile
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from app.models.diagnosis import DiagnosisRequest, DiagnosisResponse, AnalysisResult
from app.models.response import ErrorResponse
from app.services.diagnosis_service import DiagnosisService
from app.utils.file_utils import validate_image_file, save_uploaded_file
from app.utils.report_generator import ReportGenerator
from app.api.dependencies import get_diagnosis_service
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=dict)
async def analyze_image_concise(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    classification: str = Form(None, description="Medical classification"),
    history: str = Form(None, description="Medical history"),
    language: str = Form("en", description="Language for analysis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Quick concise analysis of medical image (fast)

    Returns only a single-sentence analysis without generating full reports.
    """

    try:
        # Validate image file
        validate_image_file(image)

        # Save uploaded image
        image_path = await save_uploaded_file(image, "test.jpg")

        # Prepare patient information
        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "classification": classification or "Skin Condition",
            "history": history or "No medical history provided"
        }

        # Start timing
        start_time = time.time()

        # Perform ONLY concise analysis (fast)
        logger.info(f"Starting concise analysis for patient: {patient_name} in language: {language}")
        concise_analysis = await diagnosis_service._generate_concise_analysis(
            await diagnosis_service._load_image(image_path), patient_info, language
        )

        # Calculate generation time
        generation_time = time.time() - start_time

        # Generate analysis text file for database storage
        from app.utils.report_generator import ReportGenerator
        import uuid
        report_generator = ReportGenerator()

        # Generate unique filename for this analysis
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for uniqueness
        analysis_filename = f"analysis_{patient_name.replace(' ', '_')}_{unique_id}.txt"
        analysis_path = os.path.join(report_generator.reports_dir, analysis_filename)
        await report_generator._generate_analysis_file(analysis_path, concise_analysis)

        logger.info(f"Concise analysis completed for {patient_name} in {generation_time:.2f}s")

        return {
            "success": True,
            "patient_name": patient_name,
            "concise_analysis": concise_analysis,
            "generation_time": generation_time,
            "timestamp": time.time(),
            "files": {
                "analysis_text": f"/api/v1/reports/{analysis_filename}"
            }
        }

    except Exception as e:
        logger.error(f"Error during concise analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup uploaded file
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_skin_condition(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    classification: str = Form(None, description="Medical classification"),
    history: str = Form(None, description="Medical history"),
    language: str = Form("en", description="Language for diagnosis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Analyze medical image and generate diagnosis report
    
    This endpoint accepts a medical image and patient information,
    then generates both concise analysis and comprehensive diagnosis reports.
    """
    
    try:
        # Validate image file
        validate_image_file(image)
        
        # Save uploaded image
        image_path = await save_uploaded_file(image, "test.jpg")
        
        # Prepare patient information
        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "classification": classification or "Skin Condition",
            "history": history or "No medical history provided"
        }
        
        # Start timing
        start_time = time.time()
        
        # Perform analysis
        logger.info(f"Starting diagnosis for patient: {patient_name} in language: {language}")
        concise_analysis, comprehensive_report = await diagnosis_service.analyze_image(
            image_path, patient_info, language
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Generate reports (only markdown and PDF for Full Diagnosis)
        report_generator = ReportGenerator()
        file_paths = await report_generator.generate_diagnosis_reports(
            patient_info, concise_analysis, comprehensive_report, image_path, language
        )

        # Create response
        analysis_result = AnalysisResult(
            concise_analysis=concise_analysis,
            comprehensive_report=comprehensive_report,
            generation_time=generation_time
        )

        # Use actual generated filenames in API endpoints
        api_files = {
            "markdown_report": f"/api/v1/reports/{file_paths['markdown_filename']}",
            "pdf_report": f"/api/v1/reports/{file_paths['pdf_filename']}"
        }
        
        logger.info(f"Diagnosis completed for {patient_name} in {generation_time:.2f}s")
        
        return DiagnosisResponse(
            success=True,
            patient_name=patient_name,
            analysis=analysis_result,
            files=api_files
        )
        
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded file
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

@router.get("/reports/{filename}")
async def download_report(filename: str):
    """Download generated report files and images"""

    file_path = os.path.join(settings.REPORTS_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    # Determine media type based on file extension
    if filename.endswith('.pdf'):
        media_type = 'application/pdf'
    elif filename.endswith('.md'):
        media_type = 'text/markdown'
    elif filename.endswith('.txt'):
        media_type = 'text/plain'
    elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Handle image files for markdown embedding
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            media_type = 'image/jpeg'
        elif filename.lower().endswith('.png'):
            media_type = 'image/png'
        elif filename.lower().endswith('.gif'):
            media_type = 'image/gif'
        elif filename.lower().endswith('.bmp'):
            media_type = 'image/bmp'
        elif filename.lower().endswith('.webp'):
            media_type = 'image/webp'
        else:
            media_type = 'image/jpeg'  # Default for unknown image types
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

@router.post("/generate-pdf")
async def generate_pdf_on_demand(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    language: str = Form("en", description="Language for diagnosis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Generate PDF report on-demand (for download button)
    """
    
    try:
        # Validate image file
        validate_image_file(image)
        
        # Save uploaded image
        image_path = await save_uploaded_file(image, "on_demand.jpg")
        
        # Prepare patient information
        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "classification": "Skin Condition",
            "history": "No medical history provided"
        }
        
        # Start timing
        start_time = time.time()
        
        # Perform analysis
        logger.info(f"Starting on-demand PDF generation for patient: {patient_name}")
        concise_analysis, comprehensive_report = await diagnosis_service.analyze_image(
            image_path, patient_info, language
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Generate reports
        report_generator = ReportGenerator()
        file_paths = await report_generator.generate_diagnosis_reports(
            patient_info, concise_analysis, comprehensive_report, image_path, language
        )

        # Create response with actual file URLs
        api_files = {
            "markdown_report": f"/api/v1/reports/{file_paths['markdown_filename']}",
            "pdf_report": f"/api/v1/reports/{file_paths['pdf_filename']}"
        }
        
        logger.info(f"On-demand PDF generation completed for {patient_name} in {generation_time:.2f}s")
        
        return {
            "success": True,
            "patient_name": patient_name,
            "files": api_files,
            "generation_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"Error during on-demand PDF generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded file
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

@router.post("/generate-pdf-fast")
async def generate_pdf_fast(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    language: str = Form("en", description="Language for diagnosis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Generate PDF report quickly using existing analysis (fast)
    """
    
    try:
        # Validate image file
        validate_image_file(image)
        
        # Save uploaded image
        image_path = await save_uploaded_file(image, "fast_pdf.jpg")
        
        # Prepare patient information
        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "classification": "Skin Condition",
            "history": "No medical history provided"
        }
        
        # Start timing
        start_time = time.time()
        
        # Perform quick analysis (reuse existing analysis if possible)
        logger.info(f"Starting fast PDF generation for patient: {patient_name}")
        concise_analysis, comprehensive_report = await diagnosis_service.analyze_image(
            image_path, patient_info, language
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Generate reports quickly
        report_generator = ReportGenerator()
        file_paths = await report_generator.generate_diagnosis_reports(
            patient_info, concise_analysis, comprehensive_report, image_path, language
        )

        # Create response with actual file URLs
        api_files = {
            "markdown_report": f"/api/v1/reports/{file_paths['markdown_filename']}",
            "pdf_report": f"/api/v1/reports/{file_paths['pdf_filename']}"
        }
        
        logger.info(f"Fast PDF generation completed for {patient_name} in {generation_time:.2f}s")
        
        return {
            "success": True,
            "patient_name": patient_name,
            "files": api_files,
            "generation_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"Error during fast PDF generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded file
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": "original",
                "description": "Full precision MedGemma 4B",
                "vram_usage": "12GB+",
                "speed": "Baseline",
                "quality": "Highest"
            },
            {
                "name": "4bit",
                "description": "4-bit quantized MedGemma 4B",
                "vram_usage": "6-8GB",
                "speed": "1.4x faster",
                "quality": "Very High"
            }
        ],
        "current_model": settings.MODEL_TYPE
    }

@router.post("/analyze-stream")
async def analyze_image_stream(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    classification: str = Form(None, description="Medical classification"),
    history: str = Form(None, description="Medical history"),
    language: str = Form("en", description="Language for analysis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Stream quick concise analysis of medical image in real-time

    Returns Server-Sent Events with streaming text generation.
    """

    async def generate_stream():
        image_path = None
        try:
            logger.info(f"Starting streaming analysis for patient: {patient_name}")

            # Simple approach: Create a temporary file directly without using save_uploaded_file
            # since the file might be closed when coming through the proxy

            # Basic validation
            if not image.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            # Create temporary file path
            temp_filename = f"stream_analysis_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(settings.UPLOAD_DIR, temp_filename)

            # Ensure upload directory exists
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

            # Read file content properly
            try:
                content = await image.read()
                logger.info(f"Successfully read {len(content)} bytes from uploaded file")

                # Write to temporary file
                with open(image_path, 'wb') as f:
                    f.write(content)
                logger.info(f"Image saved successfully to: {image_path}")

            except Exception as file_error:
                logger.error(f"Could not read uploaded file: {file_error}")
                raise HTTPException(status_code=500, detail=f"Could not process uploaded file: {file_error}")

            # Prepare patient information
            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "classification": classification or "Skin Condition",
                "history": history or "No medical history provided"
            }

            # Load image using the working method
            loaded_image = await diagnosis_service._load_image(image_path)
            logger.info(f"Image loaded successfully, size: {loaded_image.size}")

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting analysis...'})}\n\n"

            # Stream analysis
            logger.info("Starting streaming text generation...")
            full_text = ""
            token_count = 0
            async for token in diagnosis_service._generate_concise_analysis_stream(loaded_image, patient_info, language):
                full_text += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token, 'full_text': full_text})}\n\n"

            logger.info(f"Streaming completed. Generated {token_count} tokens, total length: {len(full_text)}")

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text, 'patient_name': patient_name})}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming analysis: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        finally:
            # Cleanup uploaded file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/diagnose-stream")
async def diagnose_skin_condition_stream(
    image: UploadFile = File(..., description="Medical image file"),
    patient_name: str = Form(..., description="Patient name"),
    patient_age: int = Form(..., description="Patient age"),
    patient_gender: str = Form(..., description="Patient gender"),
    classification: str = Form(None, description="Medical classification"),
    history: str = Form(None, description="Medical history"),
    language: str = Form("en", description="Language for analysis (en, ko, vi)"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Stream comprehensive diagnosis of medical image in real-time

    Returns Server-Sent Events with streaming text generation.
    """

    async def generate_stream():
        image_path = None
        try:
            logger.info(f"Starting streaming diagnosis for patient: {patient_name}")

            # Use the same file handling approach as the working /diagnose endpoint
            # Validate image file first
            validate_image_file(image)

            # Save uploaded image using the working method
            image_path = await save_uploaded_file(image, "stream_diagnosis.jpg")
            logger.info(f"Image saved successfully to: {image_path}")

            # Prepare patient information
            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "classification": classification or "Skin Condition",
                "history": history or "No medical history provided"
            }

            # Load image using the working method
            loaded_image = await diagnosis_service._load_image(image_path)
            logger.info(f"Image loaded successfully, size: {loaded_image.size}")

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting comprehensive diagnosis...'})}\n\n"

            # Stream diagnosis
            full_text = ""
            async for token in diagnosis_service._generate_diagnosis_report_stream(loaded_image, patient_info, language):
                full_text += token
                yield f"data: {json.dumps({'type': 'token', 'token': token, 'full_text': full_text})}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text, 'patient_name': patient_name})}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming diagnosis: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        finally:
            # Cleanup uploaded file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/test-stream")
async def test_streaming_without_file(
    patient_name: str = Form(..., description="Patient name"),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Test streaming text generation without file upload to isolate the streaming functionality
    """

    async def generate_test_stream():
        try:
            logger.info(f"Starting test streaming for patient: {patient_name}")

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting test streaming...'})}\n\n"

            # Create a simple test prompt (no image required)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Please provide a brief medical analysis for patient {patient_name}. This is a test of the streaming functionality."}]
                }
            ]

            # Stream text generation
            full_text = ""
            token_count = 0
            logger.info("Starting test streaming text generation...")

            async for token in diagnosis_service._generate_response_stream(messages, max_tokens=100):
                full_text += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token, 'full_text': full_text})}\n\n"

            logger.info(f"Test streaming completed. Generated {token_count} tokens, total length: {len(full_text)}")

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text, 'patient_name': patient_name})}\n\n"

        except Exception as e:
            logger.error(f"Error during test streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_test_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/analyze-stream-simple")
async def analyze_image_stream_simple(
    request: dict,
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Simple streaming analysis with base64 image data to avoid multipart issues
    """

    async def generate_stream():
        image_path = None
        start_time = time.time()  # Track generation time
        try:
            logger.info(f"Starting simple streaming analysis for patient: {request.get('patient_name', 'Unknown')}")

            # Extract data from request
            image_base64 = request.get('image_data')
            filename = request.get('filename', 'image.jpg')
            patient_name = request.get('patient_name', 'Unknown')
            patient_age = request.get('patient_age', 30)
            patient_gender = request.get('patient_gender', 'unknown')
            classification = request.get('classification', 'Skin Condition')
            history = request.get('history', 'No history provided')
            language = request.get('language', 'en')

            if not image_base64:
                raise HTTPException(status_code=400, detail="No image data provided")

            # Decode base64 image
            import base64
            try:
                image_data = base64.b64decode(image_base64)
                logger.info(f"Decoded image data: {len(image_data)} bytes")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

            # Save to temporary file
            temp_filename = f"simple_stream_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

            with open(image_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"Image saved to: {image_path}")

            # Prepare patient information
            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "classification": classification,
                "history": history
            }

            # Load and process image
            loaded_image = await diagnosis_service._load_image(image_path)
            logger.info(f"Image loaded successfully, size: {loaded_image.size}")

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting analysis...'})}\n\n"

            # Stream analysis
            full_text = ""
            token_count = 0
            async for token in diagnosis_service._generate_concise_analysis_stream(loaded_image, patient_info, language):
                full_text += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token, 'full_text': full_text})}\n\n"

            logger.info(f"Simple streaming completed. Generated {token_count} tokens")

            # Calculate generation time
            generation_time = time.time() - start_time

            # Generate and save analysis text file for database integration
            try:
                # Create analysis text file
                analysis_filename = f"analysis_{patient_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.txt"
                analysis_file_path = os.path.join(settings.REPORTS_DIR, analysis_filename)

                # Ensure reports directory exists
                os.makedirs(settings.REPORTS_DIR, exist_ok=True)

                # Write analysis to file
                with open(analysis_file_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)

                logger.info(f"Analysis text file saved: {analysis_file_path}")

                # Send completion event with file info for database integration
                completion_data = {
                    'type': 'complete',
                    'full_text': full_text,
                    'patient_name': patient_name,
                    'files': {
                        'analysis_text': f'/api/v1/reports/{analysis_filename}'
                    },
                    'success': True,
                    'concise_analysis': full_text,
                    'generation_time': generation_time
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as file_error:
                logger.error(f"Error saving analysis file: {file_error}")
                # Send completion without file info
                yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text, 'patient_name': patient_name})}\n\n"

        except Exception as e:
            logger.error(f"Error during simple streaming analysis: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        finally:
            # Cleanup
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/diagnose-stream-simple")
async def diagnose_stream_simple(
    request: dict,
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service)
):
    """
    Simple streaming diagnosis with base64 image data to avoid multipart issues
    """

    async def generate_stream():
        image_path = None
        start_time = time.time()  # Track generation time
        try:
            logger.info(f"Starting simple streaming diagnosis for patient: {request.get('patient_name', 'Unknown')}")

            # Extract data from request
            image_base64 = request.get('image_data')
            filename = request.get('filename', 'image.jpg')
            patient_name = request.get('patient_name', 'Unknown')
            patient_age = request.get('patient_age', 30)
            patient_gender = request.get('patient_gender', 'unknown')
            classification = request.get('classification', 'Skin Condition')
            history = request.get('history', 'No history provided')
            language = request.get('language', 'en')

            if not image_base64:
                raise HTTPException(status_code=400, detail="No image data provided")

            # Decode base64 image
            import base64
            try:
                image_data = base64.b64decode(image_base64)
                logger.info(f"Decoded image data: {len(image_data)} bytes")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

            # Save to temporary file
            temp_filename = f"simple_diagnosis_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

            with open(image_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"Image saved to: {image_path}")

            # Prepare patient information
            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "classification": classification,
                "history": history
            }

            # Load and process image
            loaded_image = await diagnosis_service._load_image(image_path)
            logger.info(f"Image loaded successfully, size: {loaded_image.size}")

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting comprehensive diagnosis...'})}\n\n"

            # Stream diagnosis
            full_text = ""
            token_count = 0
            
            async for token in diagnosis_service._generate_diagnosis_report_stream(loaded_image, patient_info, language):
                full_text += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token, 'full_text': full_text})}\n\n"

            logger.info(f"Simple diagnosis streaming completed. Generated {token_count} tokens")

            # Calculate generation time
            generation_time = time.time() - start_time

            # Generate and save report files for database integration
            try:
                # Create markdown and PDF reports
                markdown_filename = f"diagnosis_{patient_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.md"
                pdf_filename = f"diagnosis_{patient_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.pdf"

                # Ensure reports directory exists (use absolute path)
                reports_dir = os.path.abspath(settings.REPORTS_DIR)
                os.makedirs(reports_dir, exist_ok=True)

                # Update file paths to use absolute paths
                markdown_file_path = os.path.join(reports_dir, markdown_filename)
                pdf_file_path = os.path.join(reports_dir, pdf_filename)

                # Copy image to reports directory for embedding
                image_name = f"medical_image_{os.path.basename(image_path)}"
                image_dest_path = os.path.join(reports_dir, image_name)

                # Copy the image file to reports directory
                import shutil
                if os.path.exists(image_path):
                    shutil.copy2(image_path, image_dest_path)
                    logger.info(f"Image copied to reports directory: {image_dest_path}")

                # Write markdown report using proper template
                # Write markdown report using proper template with language support
                from app.utils.report_generator import get_report_templates
                templates = get_report_templates(language)
                
                with open(markdown_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {templates['title']}\n\n")

                    f.write(f"## {templates['patient_info']}\n\n")
                    f.write(f"- **{templates['name']}:** {patient_name}\n")
                    f.write(f"- **{templates['age']}:** {patient_age} {templates['years']}\n")
                    f.write(f"- **{templates['gender']}:** {patient_gender}\n")
                    f.write(f"- **{templates['classification']}:** {classification}\n")
                    f.write(f"- **{templates['medical_history']}:** {history}\n\n")

                    f.write(f"## {templates['medical_image']}\n\n")
                    if os.path.exists(image_dest_path):
                        f.write(f"![{templates['medical_image']}]({image_name})\n\n")
                        f.write(f"*Image: {image_name}*\n\n")
                    else:
                        f.write(f"*{templates['image_not_available']}*\n\n")

                    f.write(f"## {templates['comprehensive_report']}\n\n")
                    f.write(full_text)
                # Generate PDF from markdown using the same method as regular endpoint
                try:
                    import subprocess
                    import sys

                    # Get the path to md_to_pdf.py script (it's in the vlm-server root directory)
                    # __file__ is app/api/endpoints/diagnosis.py, so we need to go up 3 levels to vlm-server root
                    vlm_server_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    md_to_pdf_script = os.path.join(vlm_server_root, 'md_to_pdf.py')

                    logger.info(f"Looking for md_to_pdf.py at: {md_to_pdf_script}")
                    logger.info(f"Script exists: {os.path.exists(md_to_pdf_script)}")

                    if os.path.exists(md_to_pdf_script):
                        # Run md_to_pdf.py script to convert markdown to PDF
                        cmd = [sys.executable, md_to_pdf_script, markdown_file_path, '-o', pdf_file_path]
                        logger.info(f"Running PDF generation command: {' '.join(cmd)}")

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                        logger.info(f"PDF generation return code: {result.returncode}")
                        if result.stdout:
                            logger.info(f"PDF generation stdout: {result.stdout}")
                        if result.stderr:
                            logger.info(f"PDF generation stderr: {result.stderr}")

                        if result.returncode == 0 and os.path.exists(pdf_file_path):
                            logger.info(f"PDF report generated successfully: {pdf_file_path}")
                        else:
                            logger.error(f"PDF generation failed or file not created")
                            pdf_file_path = None
                    else:
                        logger.warning(f"md_to_pdf.py not found at {md_to_pdf_script}, skipping PDF generation")
                        pdf_file_path = None

                except subprocess.TimeoutExpired:
                    logger.error("PDF generation timed out")
                    pdf_file_path = None
                except Exception as pdf_error:
                    logger.error(f"Error generating PDF: {pdf_error}")
                    pdf_file_path = None

                logger.info(f"Diagnosis reports saved: {markdown_file_path}")

                # Send completion event with file info for database integration
                files_info = {
                    'markdown_report': f'/api/v1/reports/{markdown_filename}'
                }

                # Wait for PDF generation to complete (up to 10 seconds)
                pdf_ready = False
                if pdf_file_path:
                    for attempt in range(20):  # 20 attempts * 0.5s = 10 seconds max
                        if os.path.exists(pdf_file_path):
                            # Additional check: ensure file is not empty and fully written
                            try:
                                file_size = os.path.getsize(pdf_file_path)
                                if file_size > 1000:  # PDF should be at least 1KB
                                    files_info['pdf_report'] = f'/api/v1/reports/{pdf_filename}'
                                    logger.info(f"PDF file ready and included in response: {pdf_file_path} ({file_size} bytes)")
                                    pdf_ready = True
                                    break
                            except OSError:
                                pass  # File might still be being written

                        await asyncio.sleep(0.5)  # Wait 500ms before next check

                    if not pdf_ready:
                        logger.warning(f"PDF file not ready after 10 seconds: {pdf_file_path}")
                completion_data = {
                    'type': 'complete',
                    'full_text': full_text,
                    'patient_name': patient_name,
                    'files': files_info,
                    'success': True,
                    'analysis': {
                        'comprehensive_report': full_text,
                        'generation_time': generation_time,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as file_error:
                logger.error(f"Error saving diagnosis files: {file_error}")
                # Send completion without file info
                yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text, 'patient_name': patient_name})}\n\n"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error during simple streaming diagnosis: {e}\nTraceback:\n{error_details}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        finally:
            # Cleanup
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.get("/test-simple-stream")
async def test_simple_stream():
    """
    Ultra simple streaming test - just send characters one by one
    """

    async def generate_simple_stream():
        import asyncio

        test_message = "Hello, this is a streaming test. Each word should appear one by one."
        words = test_message.split()

        # Send start
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting simple test...'})}\n\n"

        accumulated = ""
        for i, word in enumerate(words):
            accumulated += word + " "
            yield f"data: {json.dumps({'type': 'token', 'token': word + ' ', 'full_text': accumulated})}\n\n"

            # Wait 500ms between words to make it very obvious
            await asyncio.sleep(0.5)

        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'full_text': accumulated})}\n\n"

    return StreamingResponse(
        generate_simple_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

