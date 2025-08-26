@echo off
REM Setup script for offline MedGemma model usage on Windows
REM This script will help you authenticate with Hugging Face and download the model locally

echo 🚀 Setting up MedGemma for offline usage
echo ========================================

REM Check if we're in the right directory
if not exist "test.json" (
    echo ❌ Please run this script from the vlm-server directory
    exit /b 1
)

REM Check if Python virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run 'npm install' first.
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if huggingface-cli is installed
huggingface-cli --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing Hugging Face CLI...
    pip install huggingface_hub[cli]
)

REM Check authentication status
echo 🔐 Checking Hugging Face authentication...
huggingface-cli whoami >nul 2>&1
if errorlevel 1 (
    echo ❌ Not authenticated with Hugging Face
    echo.
    echo Please follow these steps:
    echo 1. Go to https://huggingface.co/settings/tokens
    echo 2. Create a new token (read access is sufficient)
    echo 3. Run: huggingface-cli login
    echo 4. Accept the license at: https://huggingface.co/google/medgemma-4b-it
    echo.
    pause
    
    REM Check again
    huggingface-cli whoami >nul 2>&1
    if errorlevel 1 (
        echo ❌ Still not authenticated. Please complete authentication first.
        exit /b 1
    )
) else (
    echo ✅ Already authenticated with Hugging Face
)

REM Download the model
echo 📥 Downloading MedGemma model for offline use...
python scripts\download_model.py

if errorlevel 1 (
    echo ❌ Setup failed. Please check the error messages above.
    exit /b 1
) else (
    echo.
    echo 🎉 Setup complete!
    echo ✅ Model downloaded and configured for offline use
    echo ✅ Configuration updated to use local model
    echo.
    echo You can now run the VLM server with: npm run dev
    echo The model will load from the local directory without internet access.
)
