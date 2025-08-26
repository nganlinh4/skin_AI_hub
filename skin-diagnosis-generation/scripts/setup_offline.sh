#!/bin/bash

# Setup script for offline MedGemma model usage
# This script will help you authenticate with Hugging Face and download the model locally

echo "🚀 Setting up MedGemma for offline usage"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "test.json" ]; then
    echo "❌ Please run this script from the vlm-server directory"
    exit 1
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'npm install' first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing Hugging Face CLI..."
    pip install huggingface_hub[cli]
fi

# Check authentication status
echo "🔐 Checking Hugging Face authentication..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "✅ Already authenticated with Hugging Face"
else
    echo "❌ Not authenticated with Hugging Face"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token (read access is sufficient)"
    echo "3. Run: huggingface-cli login"
    echo "4. Accept the license at: https://huggingface.co/google/medgemma-4b-it"
    echo ""
    read -p "Press Enter after you've completed authentication..."
    
    # Check again
    if ! huggingface-cli whoami > /dev/null 2>&1; then
        echo "❌ Still not authenticated. Please complete authentication first."
        exit 1
    fi
fi

# Download the model
echo "📥 Downloading MedGemma model for offline use..."
python scripts/download_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete!"
    echo "✅ Model downloaded and configured for offline use"
    echo "✅ Configuration updated to use local model"
    echo ""
    echo "You can now run the VLM server with: npm run dev"
    echo "The model will load from the local directory without internet access."
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
