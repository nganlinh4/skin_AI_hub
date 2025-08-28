# Skin AI Hub - Web Demo

A minimal FastAPI + Bootstrap demo that:
- Uploads a skin image and patient metadata
- Runs on-device classification using ResNet50 weights saved at C:\work\skin_AI_hub\skin_classifier_model.pth
- Streams comprehensive diagnosis text from your existing VLM server
- Provides a PDF download link when the VLM finishes rendering the report

## Prerequisites
- Python 3.9+
- The VLM server running at http://localhost:3026
- The classifier weights file at C:\work\skin_AI_hub\skin_classifier_model.pth (already present)

## Setup

```
# From this folder
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt

# Optionally configure environment
copy .env.example .env  # then edit if needed
```

## Run

```
cd skin-diagnosis-generation ; venv\Scripts\Activate.ps1 ; cd ..\web_demo
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Notes
- Classes are fixed to: Acne, Carcinoma, Eczema, Keratosis, Milia, Rosacea based on your data/ folders.
- If you move the weights file, set CLASSIFIER_MODEL_PATH in .env and adjust utils/classifier.py accordingly.
- The streaming proxy converts the VLM's streaming output to SSE for the frontend fetch stream parser.
- If cross-origin access to the VLM reports is blocked, this app proxies PDF downloads via /download_pdf.

