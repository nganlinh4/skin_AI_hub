#!/usr/bin/env python3
"""
Run segmentation with API key set
"""

import os
from gemini_segmentation import generate_masks_for_dataset

def main():
    # Set the API key
    os.environ['GEMINI_API_KEY'] = 'AIzaSyCywPm48ip322uxGy89JjENkzSFoVO6t0I'
    
    print("=" * 60)
    print("🚀 STARTING GEMINI SEGMENTATION")
    print("=" * 60)
    print("API Key set successfully!")
    print("Processing skin condition images...")
    print()
    
    # Run the segmentation
    generate_masks_for_dataset(data_dir="../data", output_dir="../segmentation_masks")

if __name__ == "__main__":
    main()
