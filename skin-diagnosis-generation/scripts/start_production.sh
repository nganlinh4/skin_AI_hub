#!/bin/bash

# Production startup script for MedGemma FastAPI service

set -e

echo "🚀 Starting MedGemma 4B FastAPI Service (Production Mode)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required dependencies are installed
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "❌ FastAPI dependencies not found. Installing..."
    pip install fastapi uvicorn python-multipart aiofiles pydantic-settings
}

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if test.json exists
if [ ! -f "test.json" ]; then
    echo "⚠️  test.json not found. Using default configuration."
fi

# Create reports directory if it doesn't exist
mkdir -p reports

# Set production environment variables
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-3026}
export API_WORKERS=${API_WORKERS:-1}
export MODEL_TYPE=${MODEL_TYPE:-4bit}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "📋 Configuration:"
echo "   Host: $API_HOST"
echo "   Port: $API_PORT"
echo "   Workers: $API_WORKERS"
echo "   Model: $MODEL_TYPE"
echo "   Log Level: $LOG_LEVEL"

echo ""
echo "🌐 Starting FastAPI server..."
echo "   API Documentation: http://localhost:$API_PORT/docs"
echo "   Health Check: http://localhost:$API_PORT/health/"
echo ""

# Start the FastAPI application
exec uvicorn app.main:app \
    --host $API_HOST \
    --port $API_PORT \
    --workers $API_WORKERS \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --access-log \
    --no-use-colors
