# MedGemma 4B Skin Diagnosis API

AI-powered medical image analysis using Google's MedGemma 4B model with FastAPI service.

## 🚀 Quick Start

**Start the API server:**
```bash
source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 3026
```

**Quick concise analysis (fast ~2s):**
```bash
curl -X POST "http://localhost:3026/api/v1/analyze" \
  -F "image=@test.jpg" \
  -F "patient_name=Susan" \
  -F "patient_age=38" \
  -F "patient_gender=Female"
```

**Full diagnosis with reports (~18s):**
```bash
curl -X POST "http://localhost:3026/api/v1/diagnose" \
  -F "image=@test.jpg" \
  -F "patient_name=Susan" \
  -F "patient_age=38" \
  -F "patient_gender=Female"
```

**API Documentation:** http://localhost:3026/docs

## 📋 Output

### Quick Analysis (`/api/v1/analyze`)
- **Fast response** (~2 seconds)
- **JSON with concise analysis** only
- **No files generated**

### Full Diagnosis (`/api/v1/diagnose`)
- **Complete response** (~18 seconds)
- **JSON with analysis + file links**
- **Generated Files**:
  - `diagnosis_report.md` - Comprehensive markdown report
  - `diagnosis_report.pdf` - Professional PDF report
  - `image_analysis.txt` - Concise analysis text
- **Download Endpoints**:
  - `/api/v1/reports/diagnosis_report.md`
  - `/api/v1/reports/diagnosis_report.pdf`
  - `/api/v1/reports/image_analysis.txt`

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow bitsandbytes markdown reportlab beautifulsoup4
pip install fastapi uvicorn python-multipart aiofiles pydantic-settings
```

### 2. Hugging Face Authentication
```bash
huggingface-cli login
```
Accept license at: https://huggingface.co/google/medgemma-4b-it

### 3. Configuration
Edit `test.json` for patient information and settings.

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up --build
```

## 🔧 System Requirements

- **GPU**: NVIDIA with 6GB+ VRAM (4-bit model)
- **Python**: 3.8+
- **Input**: Medical image as `test.jpg`

## 📁 Key Files

- `app/main.py` - FastAPI application entry point
- `app/api/endpoints/diagnosis.py` - Main diagnosis endpoints
- `docker/docker-compose.yml` - Docker deployment
- `test.json` - Configuration file
- `test.jpg` - Input medical image
- `reports/` - Output directory with consistent naming:
  - `diagnosis_report.md` - Markdown report
  - `diagnosis_report.pdf` - PDF report
  - `image_analysis.txt` - Concise analysis

## ⚠️ Disclaimer

For research and educational purposes only. Not for clinical diagnosis.
