# MedGemma Offline Setup Guide

This guide will help you set up the MedGemma model for offline usage, eliminating the need for internet access during runtime.

## 🎉 Status: COMPLETED ✅

The MedGemma model has been successfully configured for offline usage with GPU support!

- ✅ Model downloaded locally to `./models/medgemma-4b-it/`
- ✅ GPU support enabled (CUDA 12.1 + RTX 4070 SUPER)
- ✅ 4-bit quantization configured
- ✅ Essential model files tracked in Git
- ✅ Large model weights properly ignored

## Quick Setup (For New Installations)

### Windows
```bash
cd vlm-server
scripts\setup_offline.bat
```

### Linux/Mac
```bash
cd vlm-server
./scripts/setup_offline.sh
```

## Manual Setup

If the automated scripts don't work, follow these manual steps:

### 1. Authenticate with Hugging Face

First, you need to authenticate with Hugging Face to download the model:

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face
huggingface-cli login
```

You'll need to:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Enter the token when prompted
4. Accept the license at: https://huggingface.co/google/medgemma-4b-it

### 2. Download the Model

Run the download script:

```bash
cd vlm-server
python scripts/download_model.py
```

This will:
- Download the MedGemma model to `./models/medgemma-4b-it/`
- Update your `test.json` configuration to use the local model

### 3. Verify Configuration

Check that your `test.json` file has been updated:

```json
{
  "model": {
    "model_id": "google/medgemma-4b-it",
    "local_model_path": "./models/medgemma-4b-it",
    "use_local": true,
    ...
  }
}
```

### 4. Test Offline Usage

Now you can run the VLM server without internet access:

```bash
npm run dev
```

The model will load from the local directory instead of downloading from Hugging Face.

## Configuration Options

You can control the offline behavior by editing `test.json`:

- `use_local: true` - Use local model instead of downloading
- `use_local: false` - Download from Hugging Face (requires internet)
- `local_model_path` - Path to the local model directory

## Troubleshooting

### Authentication Issues
- Make sure you have a valid Hugging Face token
- Ensure you've accepted the MedGemma license
- Try logging out and back in: `huggingface-cli logout && huggingface-cli login`

### Download Issues
- Check your internet connection
- Ensure you have enough disk space (model is ~8GB)
- Try downloading to a different directory

### Model Loading Issues
- Verify the local model path exists and contains the model files
- Check that `use_local` is set to `true` in your configuration
- Ensure your virtual environment has the required dependencies

## Model Size and Requirements

- **Model Size**: ~8GB
- **RAM Requirements**: 8GB+ recommended
- **GPU**: Optional but recommended for faster inference
- **Disk Space**: 10GB+ free space recommended

## 📁 What's Included in Git

The following essential files are tracked in Git for offline operation:

### Model Configuration Files (Small, Essential)
- `tokenizer.model` - **Critical tokenizer model file**
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings
- `config.json` - Model configuration
- `generation_config.json` - Text generation settings
- `preprocessor_config.json` - Image preprocessing config
- `processor_config.json` - Processor configuration
- `special_tokens_map.json` - Special token mappings
- `added_tokens.json` - Additional tokens
- `model.safetensors.index.json` - Model file index
- `chat_template.jinja` - Chat template
- `README.md` - Model documentation
- `.gitattributes` - Git attributes

### Model Weights (Ignored, Too Large)
- `model-00001-of-00002.safetensors` - Model weights part 1 (~4GB)
- `model-00002-of-00002.safetensors` - Model weights part 2 (~4GB)

## 🚀 Current Setup Status

Your VLM server is now running with:
- **Local Model Path**: `./models/medgemma-4b-it/`
- **GPU Acceleration**: NVIDIA RTX 4070 SUPER
- **CUDA Version**: 12.1
- **Quantization**: 4-bit (BitsAndBytes)
- **Memory Usage**: 90% GPU memory for model, 10% buffer
- **Status**: ✅ Fully operational offline

## Benefits of Offline Setup

1. **No Internet Required**: Run the model without internet access
2. **Faster Loading**: No download time on subsequent runs
3. **Reliability**: No dependency on Hugging Face servers
4. **Privacy**: Model runs entirely locally
5. **Consistent Performance**: No network latency issues
6. **GPU Acceleration**: Optimized for your RTX 4070 SUPER
