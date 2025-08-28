"""
VLM Client for interacting with the diagnosis server
Provides async methods for streaming diagnosis generation
"""

import httpx
import json
import base64
import os
from typing import AsyncGenerator, Dict, Any, Optional
from pathlib import Path

# Configuration
VLM_HOST = os.getenv("VLM_HOST", "http://localhost:3026")
VLM_API_BASE = f"{VLM_HOST}/api/v1"

class VLMClient:
    """Client for VLM diagnosis server"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        
    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def stream_concise_analysis(
        self, 
        image_base64: str,
        patient_info: Dict[str, Any],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """
        Stream concise analysis from VLM server
        
        Args:
            image_base64: Base64 encoded image
            patient_info: Patient information dictionary
            language: Language for analysis
            
        Yields:
            Text tokens from the analysis
        """
        payload = {
            "image_data": image_base64,
            "filename": "image.jpg",
            "patient_name": patient_info.get("name", "Unknown"),
            "patient_age": patient_info.get("age", 30),
            "patient_gender": patient_info.get("gender", "unknown"),
            "classification": patient_info.get("classification", "Skin Condition"),
            "history": patient_info.get("history", "No history provided"),
            "language": language
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{VLM_API_BASE}/analyze-stream-simple",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            if data.get("type") == "token":
                                yield data.get("token", "")
                            elif data.get("type") == "complete":
                                # Extract files info if available
                                if "files" in data:
                                    yield f"\n\n[FILES]{json.dumps(data['files'])}[/FILES]"
                            elif data.get("type") == "error":
                                yield f"\n\nError: {data.get('error', 'Unknown error')}"
                                
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            yield f"\n\nError: Server returned {e.response.status_code}"
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    async def stream_diagnosis(
        self,
        image_base64: str,
        patient_info: Dict[str, Any],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """
        Stream comprehensive diagnosis from VLM server
        
        Args:
            image_base64: Base64 encoded image
            patient_info: Patient information dictionary
            language: Language for diagnosis
            
        Yields:
            Text tokens from the diagnosis
        """
        payload = {
            "image_data": image_base64,
            "filename": "image.jpg",
            "patient_name": patient_info.get("name", "Unknown"),
            "patient_age": patient_info.get("age", 30),
            "patient_gender": patient_info.get("gender", "unknown"),
            "classification": patient_info.get("classification", "Skin Condition"),
            "history": patient_info.get("history", "No history provided"),
            "language": language
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{VLM_API_BASE}/diagnose-stream-simple",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            if data.get("type") == "token":
                                yield data.get("token", "")
                            elif data.get("type") == "complete":
                                # Extract files info if available
                                if "files" in data:
                                    yield f"\n\n[FILES]{json.dumps(data['files'])}[/FILES]"
                            elif data.get("type") == "error":
                                yield f"\n\nError: {data.get('error', 'Unknown error')}"
                                
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            yield f"\n\nError: Server returned {e.response.status_code}"
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    async def generate_pdf_fast(
        self,
        image_base64: str,
        patient_info: Dict[str, Any],
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate PDF report quickly
        
        Args:
            image_base64: Base64 encoded image
            patient_info: Patient information dictionary
            language: Language for report
            
        Returns:
            Response with file paths or None if failed
        """
        payload = {
            "image": image_base64,  # Note: different field name for this endpoint
            "patient_name": patient_info.get("name", "Unknown"),
            "patient_age": patient_info.get("age", 30),
            "patient_gender": patient_info.get("gender", "unknown"),
            "language": language
        }
        
        try:
            response = await self.client.post(
                f"{VLM_API_BASE}/generate-pdf-fast",
                data=payload  # Form data for this endpoint
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
    
    async def download_report(self, filename: str) -> Optional[bytes]:
        """
        Download a report file from VLM server
        
        Args:
            filename: Name of the report file
            
        Returns:
            File bytes or None if failed
        """
        try:
            response = await self.client.get(
                f"{VLM_API_BASE}/reports/{filename}"
            )
            response.raise_for_status()
            return response.content
            
        except Exception as e:
            print(f"Error downloading report: {e}")
            return None

# Global client instance
vlm_client = VLMClient()

async def stream_analysis(
    image_path: str,
    patient_info: Dict[str, Any],
    language: str = "en",
    comprehensive: bool = False
) -> AsyncGenerator[str, None]:
    """
    Convenience function to stream analysis
    
    Args:
        image_path: Path to image file
        patient_info: Patient information
        language: Language for analysis
        comprehensive: If True, generate comprehensive diagnosis
        
    Yields:
        Text tokens from analysis
    """
    image_base64 = vlm_client.encode_image(image_path)
    
    if comprehensive:
        async for token in vlm_client.stream_diagnosis(image_base64, patient_info, language):
            yield token
    else:
        async for token in vlm_client.stream_concise_analysis(image_base64, patient_info, language):
            yield token
