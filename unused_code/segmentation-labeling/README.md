# Gemini Segmentation Mask Generator

Simple Python script to generate segmentation masks for skin condition images using Google Gemini API.

## Setup

1. Get a Gemini API key from Google AI Studio
2. Set environment variable:
   ```bash
   export GEMINI_API_KEY='your_api_key_here'
   ```

## Usage

```bash
python gemini_segmentation.py
```

The script will:
- Process images from each condition folder (Acne, Carcinoma, etc.)
- Use the folder name as the prompt (e.g., "Segment all acne lesions...")
- Generate segmentation masks using Gemini API
- Save results as JSON and visualization images

## Output

- `segmentation_masks/[condition]/[image]_masks.json` - Mask data
- `segmentation_masks/[condition]/[image]_visualization.png` - Visual overlay

## Features

- Automatic prompt generation based on folder names
- Processes first 5 images per condition as examples
- Creates bounding boxes and labels for detected areas
- Saves both raw data and visualizations
