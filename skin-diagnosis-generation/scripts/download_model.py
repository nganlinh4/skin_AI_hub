#!/usr/bin/env python3
"""
Script to download and cache the MedGemma model locally for offline use.
This script will download the model from Hugging Face and save it locally.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional

def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated with Hugging Face: {e}")
        print("Please run: huggingface-cli login")
        print("And accept the license at: https://huggingface.co/google/medgemma-4b-it")
        return False

def download_model(model_id: str = "google/medgemma-4b-it", local_path: str = "./models/medgemma-4b-it"):
    """Download the model and processor to Hugging Face cache, then copy to local directory"""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from huggingface_hub import snapshot_download
        import shutil

        print(f"📥 Downloading model: {model_id}")
        print(f"📁 Target local path: {local_path}")

        # Download to Hugging Face cache first
        print("📥 Downloading to Hugging Face cache...")
        cache_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=None,  # Use default cache
            local_files_only=False
        )
        print(f"✅ Model cached at: {cache_dir}")

        # Create local directory
        os.makedirs(local_path, exist_ok=True)

        # Copy from cache to local directory
        print("📁 Copying to local directory...")
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        shutil.copytree(cache_dir, local_path)
        print("✅ Model copied to local directory successfully")

        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def update_config(config_path: str = "test.json", local_model_path: str = "./models/medgemma-4b-it"):
    """Update the configuration to use local model"""
    try:
        # Read current config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Update model configuration
        config['model']['use_local'] = True
        config['model']['local_model_path'] = local_model_path
        
        # Write updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Updated configuration file: {config_path}")
        print(f"   - use_local: true")
        print(f"   - local_model_path: {local_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def main():
    """Main function"""
    print("🚀 MedGemma Model Download Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("test.json"):
        print("❌ Please run this script from the vlm-server directory")
        sys.exit(1)
    
    # Check authentication
    if not check_huggingface_auth():
        sys.exit(1)
    
    # Download model
    model_id = "google/medgemma-4b-it"
    local_path = "./models/medgemma-4b-it"
    
    if download_model(model_id, local_path):
        print("\n🎉 Model downloaded successfully!")
        
        # Update configuration
        if update_config("test.json", local_path):
            print("\n✅ Setup complete!")
            print("You can now run the VLM server offline.")
            print("The model will be loaded from the local directory.")
        else:
            print("\n⚠️  Model downloaded but config update failed.")
            print("Please manually set 'use_local': true in test.json")
    else:
        print("\n❌ Model download failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
