# Manual Setup for Offline MedGemma

Since the automated setup requires interactive authentication, here's a step-by-step manual process:

## Step 1: Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "medgemma-access" 
4. Select "Read" access level
5. Copy the token

## Step 2: Accept Model License

1. Go to https://huggingface.co/google/medgemma-4b-it
2. Click "Agree and access repository"

## Step 3: Authenticate

Open a terminal in the vlm-server directory and run:

```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Login to Hugging Face
python -c "from huggingface_hub import login; login()"
```

When prompted:
- Enter your token (it won't be visible as you type)
- Press Enter
- Type 'y' when asked about git credentials

## Step 4: Download Model

After authentication, run:

```bash
python scripts\download_model.py
```

This will:
- Download the model to `./models/medgemma-4b-it/` (~8GB)
- Update your configuration to use the local model

## Step 5: Test Offline Usage

```bash
npm run dev
```

The server should now start and load the model from the local directory.

## Alternative: Quick Test

If you want to test without downloading the full model first, you can:

1. Set `"use_local": false` in `test.json` 
2. Run `npm run dev` with internet connection
3. The model will be cached automatically after first download
4. Then set `"use_local": true` for future offline usage

## Verification

Check that your `test.json` has:

```json
{
  "model": {
    "model_id": "google/medgemma-4b-it",
    "local_model_path": "./models/medgemma-4b-it",
    "use_local": true
  }
}
```

## Troubleshooting

- **Authentication fails**: Make sure you have the correct token and accepted the license
- **Download fails**: Check internet connection and disk space (need ~10GB free)
- **Model loading fails**: Verify the local path exists and contains model files
