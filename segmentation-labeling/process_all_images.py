#!/usr/bin/env python3
"""
Process all skin condition images with segmentation masks
Resumes from where it left off if interrupted
"""

import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import time
from gemini_segmentation import GeminiSegmentation

def decode_mask_from_base64(mask_data_str):
    """Decode the base64 mask image from Gemini"""
    try:
        if mask_data_str.startswith('data:image'):
            mask_data_str = mask_data_str.split(',')[1]
        
        missing_padding = len(mask_data_str) % 4
        if missing_padding:
            mask_data_str += '=' * (4 - missing_padding)
        
        image_bytes = base64.b64decode(mask_data_str)
        image = Image.open(io.BytesIO(image_bytes))
        mask_array = np.array(image)
        
        if len(mask_array.shape) == 3:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        return mask_array
    except Exception as e:
        print(f"    Error decoding mask: {e}")
        return None

def draw_mask_contours(image, masks_data):
    """Draw yellow borders around actual mask contours"""
    result_image = image.copy()
    yellow = (255, 255, 0)  # RGB yellow
    
    contour_count = 0
    for i, mask_info in enumerate(masks_data):
        if 'mask' in mask_info and 'box_2d' in mask_info:
            mask_array = decode_mask_from_base64(mask_info['mask'])
            
            if mask_array is not None:
                h, w = image.shape[:2]
                ymin, xmin, ymax, xmax = mask_info['box_2d']
                
                x1 = int(xmin * w / 1000)
                y1 = int(ymin * h / 1000)
                x2 = int(xmax * w / 1000)
                y2 = int(ymax * h / 1000)
                
                box_width = x2 - x1
                box_height = y2 - y1
                
                if box_width > 0 and box_height > 0:
                    resized_mask = cv2.resize(mask_array, (box_width, box_height))
                    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        contour[:, :, 0] += x1
                        contour[:, :, 1] += y1
                        cv2.drawContours(result_image, [contour], -1, yellow, 3)
                        contour_count += 1
    
    return result_image, contour_count

def get_processed_images(output_dir):
    """Get list of already processed images"""
    processed = set()

    if os.path.exists(output_dir):
        for condition_dir in os.listdir(output_dir):
            condition_path = os.path.join(output_dir, condition_dir)
            if os.path.isdir(condition_path):
                for file in os.listdir(condition_path):
                    if file.endswith('_masks.json'):
                        # Extract original image name
                        image_name = file.replace('_masks.json', '') + '.jpg'
                        processed.add(f"{condition_dir}/{image_name}")

    return processed

def clean_resolved_errors(output_dir, processed_images):
    """Remove errors for images that now have successful results"""
    error_log_file = output_dir / "error_log.json"

    if not error_log_file.exists():
        print(f"  📋 No error log found to clean")
        return

    try:
        with open(error_log_file, 'r') as f:
            error_data = json.load(f)

        original_error_count = len(error_data.get('errors', []))
        print(f"  📋 Checking {original_error_count} errors for cleanup...")

        # Filter out errors for images that now have results
        remaining_errors = []
        resolved_errors = []
        for error in error_data.get('errors', []):
            error_image = error.get('image', '')
            if error_image not in processed_images:
                remaining_errors.append(error)
            else:
                resolved_errors.append(error_image)

        # Update error log if any errors were resolved
        if len(remaining_errors) < original_error_count:
            resolved_count = original_error_count - len(remaining_errors)
            print(f"  🧹 Cleaned {resolved_count} resolved errors from log")
            print(f"    Resolved: {resolved_errors[:5]}{'...' if len(resolved_errors) > 5 else ''}")

            error_data['errors'] = remaining_errors
            error_data['total_errors'] = len(remaining_errors)
            error_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

            with open(error_log_file, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
        else:
            print(f"  📋 No resolved errors found to clean")

    except Exception as e:
        print(f"  ⚠️  Could not clean error log: {e}")

def write_error_log(output_dir, errors):
    """Write error log immediately"""
    if errors:
        error_log_file = output_dir / "error_log.json"
        with open(error_log_file, 'w') as f:
            json.dump({
                'total_errors': len(errors),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'errors': errors
            }, f, indent=2, default=str)

def get_next_api_key():
    """Get next available API key from the rotation list"""
    api_keys = [
        'AIzaSyAQg-6-56wZ784wFRioXFhm1Ic6WjOpab4',
        'AIzaSyCzzFd43_7w16ER0QwdTagnHJUpfP9qv-o',
        'AIzaSyA3kBy3wxFgpYr02ZeoYmkw0XdV80verCA',
        'AIzaSyC8e6up60L0Yqwk88niw4mLTzvwkr7v0jQ',
        'AIzaSyCzRHkl8YI6pl8-Okb7YpVMwsnrA2rM3yQ',
        'AIzaSyCsBj3g7GwirPObMfsaSnFSK5DYLHDdqDQ',
        'AIzaSyDT3Ja6FbTdRR3GwOXFNZ_4ysC0uMdzZ80',
        'AIzaSyBXfrZJR430XYu0SeTWVXpLd0HduAFB7o0',
        'AIzaSyB1kiX9r3t__ok-FUphvbgNiDkednJYNPw',
        'AIzaSyDsZ-bKuMilVkyJS9Le0guhgdNfRSwGwSk',
        'AIzaSyCS5AieRrjYGRchC8Z_WZgcMRdEe4iw-fA',
        'AIzaSyCOLewIMbZiNN_-h8s7kUGPED2V9AQRcSw'
    ]

    # Get current key index from environment or start at 0
    current_index = int(os.environ.get('CURRENT_KEY_INDEX', '0'))

    # Move to next key
    next_index = (current_index + 1) % len(api_keys)
    os.environ['CURRENT_KEY_INDEX'] = str(next_index)

    return api_keys[next_index], next_index

def rotate_api_key():
    """Rotate to next API key when current one hits rate limit"""
    api_key, key_index = get_next_api_key()
    os.environ['GEMINI_API_KEY'] = api_key
    print(f"    🔄 Rotated to API key #{key_index + 1} due to rate limit")
    return api_key

def process_all_images():
    """Process all skin condition images with resume capability"""

    # Set initial API key
    api_key, key_index = get_next_api_key()
    os.environ['GEMINI_API_KEY'] = api_key
    print(f"🔑 Starting with API key #{key_index + 1}")

    print("🎭 PROCESSING ALL SKIN CONDITION IMAGES")
    print("=" * 60)

    # Setup directories
    data_dir = Path("data")
    output_dir = Path("segmentation_results")
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    # Get all conditions
    conditions = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found conditions: {conditions}")
    
    # Get already processed images
    processed_images = get_processed_images(output_dir)
    print(f"Already processed: {len(processed_images)} images")

    # Debug: show a few processed images
    if processed_images:
        sample_processed = list(processed_images)[:3]
        print(f"  Sample processed: {sample_processed}")

    # Clean up resolved errors from previous runs
    clean_resolved_errors(output_dir, processed_images)
    
    # Initialize segmenter
    segmenter = GeminiSegmentation()
    
    # Statistics
    total_processed = 0
    total_masks = 0
    total_contours = 0
    errors = []
    
    # Process each condition
    for condition in conditions:
        print(f"\n📋 Processing {condition}...")
        
        condition_data_dir = data_dir / condition
        condition_output_dir = output_dir / condition
        condition_output_dir.mkdir(exist_ok=True)
        
        # Get all images in condition
        image_files = list(condition_data_dir.glob("*.jpg")) + \
                     list(condition_data_dir.glob("*.jpeg")) + \
                     list(condition_data_dir.glob("*.png"))
        
        print(f"  Found {len(image_files)} images")
        
        # Filter out already processed images
        remaining_images = []
        for img_file in image_files:
            relative_path = f"{condition}/{img_file.name}"
            if relative_path not in processed_images:
                remaining_images.append(img_file)
        
        print(f"  Need to process: {len(remaining_images)} images")
        
        # Process remaining images
        for i, image_file in enumerate(remaining_images):
            print(f"  🖼️  [{i+1}/{len(remaining_images)}] {image_file.name}")
            
            try:
                # Generate segmentation masks
                masks = segmenter.generate_masks(str(image_file), condition)

                # Check if we got a rate limit error and need to rotate API key
                if not masks and hasattr(segmenter, 'last_error') and segmenter.last_error:
                    error_details = segmenter.last_error
                    if (error_details.get('error_type') == 'API_ERROR' and
                        error_details.get('status_code') == 429):

                        # Rotate API key and retry
                        rotate_api_key()
                        segmenter = GeminiSegmentation()  # Reinitialize with new key

                        print(f"    🔄 Retrying with new API key...")
                        masks = segmenter.generate_masks(str(image_file), condition)

                if masks:
                    # Load original image for visualization
                    original = cv2.imread(str(image_file))
                    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                    # Draw mask contours
                    bordered_image, contour_count = draw_mask_contours(original_rgb, masks)

                    # Save results
                    base_name = image_file.stem

                    # Save mask data
                    mask_file = condition_output_dir / f"{base_name}_masks.json"
                    with open(mask_file, 'w') as f:
                        json.dump({
                            'image': image_file.name,
                            'condition': condition,
                            'masks': masks,
                            'mask_count': len(masks),
                            'contour_count': contour_count
                        }, f, indent=2, default=str)

                    # Save visualization
                    viz_file = condition_output_dir / f"{base_name}_contours.png"
                    bordered_bgr = cv2.cvtColor(bordered_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(viz_file), bordered_bgr)

                    print(f"    ✓ {len(masks)} masks, {contour_count} contours")
                    total_masks += len(masks)
                    total_contours += contour_count
                else:
                    # Log the failure with details from API
                    error_info = {
                        'image': f"{condition}/{image_file.name}",
                        'error_type': 'NO_MASKS_DETECTED',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'details': 'API returned empty or invalid response'
                    }

                    # Add API-specific error details if available
                    if hasattr(segmenter, 'last_error') and segmenter.last_error:
                        error_info['api_error'] = segmenter.last_error
                        segmenter.last_error = None  # Clear for next call

                    errors.append(error_info)
                    print(f"    ⚠️  No masks detected - logged for investigation")

                    # Write error log immediately
                    write_error_log(output_dir, errors)

                total_processed += 1

                # Small delay to be nice to API
                time.sleep(1)

            except Exception as e:
                # Log detailed error information
                error_info = {
                    'image': f"{condition}/{image_file.name}",
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'details': 'Exception during processing'
                }
                errors.append(error_info)
                print(f"    ❌ Error: {e} - logged for investigation")

                # Write error log immediately
                write_error_log(output_dir, errors)

                # Continue processing other images
                continue
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"  • Images processed: {total_processed}")
    print(f"  • Total masks detected: {total_masks}")
    print(f"  • Total contours drawn: {total_contours}")
    print(f"  • Average masks per image: {total_masks/total_processed:.1f}" if total_processed > 0 else "  • No images processed")
    print(f"  • Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            if isinstance(error, dict):
                print(f"  • {error['image']}: {error['error_type']}")
            else:
                print(f"  • {error}")
        if len(errors) > 5:
            print(f"  • ... and {len(errors)-5} more errors")
        print(f"  📋 See detailed error log: {output_dir}/error_log.json")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("  • JSON files with mask data")
    print("  • PNG files with yellow contour borders")
    
    # Save processing log
    log_file = output_dir / "processing_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'total_processed': total_processed,
            'total_masks': total_masks,
            'total_contours': total_contours,
            'error_count': len(errors),
            'conditions': conditions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    # Save detailed error log
    if errors:
        error_log_file = output_dir / "error_log.json"
        with open(error_log_file, 'w') as f:
            json.dump({
                'total_errors': len(errors),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'errors': errors
            }, f, indent=2, default=str)
        print(f"📋 Detailed error log saved to: {error_log_file}")

    print(f"📋 Processing summary saved to: {log_file}")

if __name__ == "__main__":
    process_all_images()
