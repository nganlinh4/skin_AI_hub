"""
Image utility functions for drawing bounding boxes
"""

import os
import logging
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def draw_bounding_box(
    input_image_path: str, 
    output_image_path: str, 
    bbox: List[float],
    box_color: str = 'red',
    box_width: int = 3,
    label: Optional[str] = None
) -> None:
    """
    Draw a bounding box on an image based on normalized coordinates
    
    Args:
        input_image_path: Path to the input image
        output_image_path: Path where the annotated image will be saved
        bbox: Normalized bounding box coordinates [x_min, y_min, x_max, y_max] (0-1)
        box_color: Color of the bounding box
        box_width: Width of the bounding box lines
        label: Optional label to display near the box
    """
    try:
        # Open the image
        image = Image.open(input_image_path).convert('RGB')
        width, height = image.size
        
        # Convert normalized coordinates to pixel coordinates
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Draw the bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=box_color,
            width=box_width
        )
        
        # Add label if provided
        if label:
            try:
                # Try to use a better font if available
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                # Use default font if truetype font not available
                font = ImageFont.load_default()
            
            # Draw label background
            text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)
            draw.rectangle(text_bbox, fill=box_color)
            
            # Draw label text
            draw.text((x_min, y_min - 20), label, fill='white', font=font)
        
        # Save the annotated image
        image.save(output_image_path, 'JPEG', quality=95)
        logger.info(f"Annotated image saved to: {output_image_path}")
        
    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}")
        raise

def extract_bbox_from_text(text: str) -> Optional[List[float]]:
    """
    Extract bounding box coordinates from diagnosis text
    
    Args:
        text: Diagnosis text containing BOX_2D coordinates
        
    Returns:
        List of normalized coordinates [x_min, y_min, x_max, y_max] or None
    """
    import re
    
    # Pattern to match BOX_2D: [x, y, x, y]
    pattern = r'BOX_2D:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
    
    match = re.search(pattern, text)
    if match:
        try:
            bbox = [
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4))
            ]
            return bbox
        except ValueError:
            logger.error(f"Could not parse bbox values: {match.groups()}")
            return None
    
    return None

def remove_bbox_from_text(text: str) -> str:
    """
    Remove BOX_2D line from diagnosis text for clean output
    
    Args:
        text: Diagnosis text containing BOX_2D coordinates
        
    Returns:
        Cleaned text without BOX_2D line
    """
    import re
    
    # Pattern to match and remove BOX_2D line
    pattern = r'BOX_2D:\s*\[.*?\]\s*\n?'
    
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()
