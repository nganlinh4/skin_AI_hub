#!/usr/bin/env python3
"""
Gemini API Segmentation Mask Generator
Generates segmentation masks for skin condition images using Google Gemini API
"""

import os
import json
import base64
import requests
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

class GeminiSegmentation:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.last_error = None
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def encode_image(self, image_path, max_size=640):
        """Encode image to base64 and resize if needed"""
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
    
    def create_segmentation_prompt(self, condition_name):
        """Create segmentation prompt based on condition name"""
        prompts = {
            'Acne': 'Segment all acne lesions, pimples, blackheads, and inflamed areas',
            'Carcinoma': 'Segment all carcinoma lesions, suspicious growths, and abnormal tissue areas',
            'Eczema': 'Segment all eczema patches, inflamed skin areas, and affected regions',
            'Keratosis': 'Segment all keratosis lesions, rough patches, and keratinized areas',
            'Milia': 'Segment all milia bumps, small white cysts, and raised areas',
            'Rosacea': 'Segment all rosacea affected areas, redness, and inflamed regions'
        }
        
        base_prompt = prompts.get(condition_name, f'Segment all {condition_name.lower()} related areas')
        
        return f"""Segment {base_prompt.lower()}. Output JSON list with "box_2d" [ymin,xmin,ymax,xmax] 0-1000, "mask" data, "label". Focus on visible {condition_name.lower()} areas only."""

    def generate_masks(self, image_path, condition_name):
        """Generate segmentation masks using Gemini API"""
        try:
            # Encode image
            image_b64 = self.encode_image(image_path)
            
            # Create prompt
            prompt = self.create_segmentation_prompt(condition_name)
            
            # Prepare request
            headers = {
                'Content-Type': 'application/json',
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 4096,
                }
            }
            
            # Make API request
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                error_details = {
                    'status_code': response.status_code,
                    'response_text': response.text,
                    'error_type': 'API_ERROR'
                }
                print(f"API Error: {response.status_code} - {response.text}")
                # Store error details for the calling function to handle
                self.last_error = error_details
                return None
            
            result = response.json()
            
            # Extract text response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text_response = candidate['content']['parts'][0]['text']

                    # Parse JSON from response
                    if '```json' in text_response:
                        json_text = text_response.split('```json')[1].split('```')[0]
                    else:
                        json_text = text_response

                    try:
                        masks_data = json.loads(json_text.strip())
                        return masks_data
                    except json.JSONDecodeError as e:
                        error_details = {
                            'error_type': 'JSON_PARSE_ERROR',
                            'json_error': str(e),
                            'response_text': text_response[:1000],
                            'json_text': json_text[:500]
                        }
                        print(f"JSON Parse Error: {e}")
                        print(f"Response text: {text_response[:500]}...")
                        self.last_error = error_details
                        return None
                else:
                    error_details = {
                        'error_type': 'UNEXPECTED_RESPONSE_STRUCTURE',
                        'candidate_structure': str(candidate)
                    }
                    print(f"Unexpected response structure: {candidate}")
                    self.last_error = error_details
                    return None
            
            return None
            
        except Exception as e:
            print(f"Error generating masks: {e}")
            return None
    
    def save_mask_visualization(self, image_path, masks_data, output_path):
        """Save visualization of segmentation masks on original image"""
        if not masks_data:
            return

        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return

        # Create overlay
        overlay = image.copy()

        # Colors for different masks
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for i, mask_info in enumerate(masks_data):
            color = colors[i % len(colors)]
            label = mask_info.get('label', f'Segment {i+1}')

            # Draw bounding box if available
            if 'box_2d' in mask_info:
                h, w = image.shape[:2]
                ymin, xmin, ymax, xmax = mask_info['box_2d']

                # Convert from 0-1000 to pixel coordinates
                x1 = int(xmin * w / 1000)
                y1 = int(ymin * h / 1000)
                x2 = int(xmax * w / 1000)
                y2 = int(ymax * h / 1000)

                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Add label
                cv2.putText(overlay, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Note about mask data
                if 'mask' in mask_info:
                    cv2.putText(overlay, "✓ Has Mask", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save result
        cv2.imwrite(str(output_path), overlay)
        print(f"Saved mask visualization: {output_path}")

def generate_masks_for_dataset(data_dir="data", output_dir="segmentation_masks"):
    """Generate masks for all images in the dataset"""
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Initialize segmentation
    segmenter = GeminiSegmentation(api_key)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory {data_dir} not found!")
        return
    
    # Process each condition folder
    for condition_dir in data_path.iterdir():
        if not condition_dir.is_dir():
            continue
        
        condition_name = condition_dir.name
        print(f"\nProcessing {condition_name}...")
        
        # Create output subdirectory
        condition_output = output_path / condition_name
        condition_output.mkdir(exist_ok=True)
        
        # Get image files
        image_files = list(condition_dir.glob("*.jpg")) + list(condition_dir.glob("*.jpeg")) + list(condition_dir.glob("*.png"))
        
        # Process first 5 images as examples
        for i, image_file in enumerate(image_files[:5]):
            print(f"  Processing {image_file.name}...")
            
            # Generate masks
            masks_data = segmenter.generate_masks(image_file, condition_name)
            
            if masks_data:
                # Save mask data as JSON
                json_output = condition_output / f"{image_file.stem}_masks.json"
                with open(json_output, 'w') as f:
                    json.dump(masks_data, f, indent=2)
                
                # Save visualization
                viz_output = condition_output / f"{image_file.stem}_visualization.png"
                segmenter.save_mask_visualization(image_file, masks_data, viz_output)
                
                print(f"    Generated {len(masks_data)} masks")
            else:
                print(f"    Failed to generate masks")
    
    print(f"\nSegmentation masks saved to: {output_dir}")

if __name__ == "__main__":
    generate_masks_for_dataset()
