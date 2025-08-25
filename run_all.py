#!/usr/bin/env python3
"""
Simple skin condition classifier with attention maps
Run this script to train the model and generate attention visualizations
"""

import os
import sys

def main():
    print("=" * 60)
    print("SKIN CONDITION CLASSIFIER WITH ATTENTION MAPS")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("ERROR: 'data' directory not found!")
        print("Please make sure your skin condition images are in the 'data' folder")
        return
    
    # Check data structure
    classes = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
    if not classes:
        print("ERROR: No class directories found in 'data' folder!")
        return
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Count images
    total_images = 0
    for class_name in classes:
        class_dir = os.path.join('data', class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        print(f"  {class_name}: {len(images)} images")
    
    print(f"Total images: {total_images}")
    print()
    
    # Step 1: Train the model
    print("STEP 1: Training the classifier...")
    print("-" * 40)
    
    try:
        exec(open('skin_classifier.py').read())
        print("✓ Model training completed successfully!")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return
    
    print()
    
    # Step 2: Generate attention maps
    print("STEP 2: Generating attention maps...")
    print("-" * 40)
    
    try:
        exec(open('attention_maps.py').read())
        print("✓ Attention maps generated successfully!")
    except Exception as e:
        print(f"✗ Error generating attention maps: {e}")
        return
    
    print()
    print("=" * 60)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated files:")
    print("  - skin_classifier_model.pth (trained model)")
    print("  - training_history.png (training curves)")
    print("  - confusion_matrix.png (evaluation results)")
    print("  - attention_maps/ (folder with attention visualizations)")
    print()
    print("You can now use these results for your competition report!")

if __name__ == "__main__":
    main()
