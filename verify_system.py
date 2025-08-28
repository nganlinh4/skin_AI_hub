"""
Script to verify system capabilities and actual performance metrics
"""

import torch
import os
import json
import time
from pathlib import Path

print("="*60)
print("SYSTEM VERIFICATION REPORT")
print("="*60)

# 1. PyTorch and CUDA Information
print("\n1. PyTorch and CUDA Information:")
print("-" * 40)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. Dataset Information
print("\n2. Dataset Information:")
print("-" * 40)
data_dir = Path("C:/work/skin_AI_hub/data")
if data_dir.exists():
    total_images = 0
    for category in sorted(data_dir.iterdir()):
        if category.is_dir():
            images = list(category.glob("*.jpg")) + list(category.glob("*.png"))
            print(f"{category.name}: {len(images)} images")
            total_images += len(images)
    print(f"Total images: {total_images}")

# 3. Model Information
print("\n3. Model Information:")
print("-" * 40)

# Check skin classifier model
model_path = Path("C:/work/skin_AI_hub/skin_classifier_model.pth")
if model_path.exists():
    model_size = model_path.stat().st_size / (1024**2)
    print(f"Skin classifier model exists: {model_size:.2f} MB")
    
    # Try to load and get info
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        num_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
        print(f"Model parameters: {num_params:,}")
    except:
        print("Could not load model checkpoint for parameter count")

# 4. Check configuration files
print("\n4. Configuration Files:")
print("-" * 40)

# Check test.json
test_json = Path("C:/work/skin_AI_hub/skin-diagnosis-generation/test.json")
if test_json.exists():
    with open(test_json, 'r', encoding='utf-8') as f:
        config = json.load(f)
        print("test.json found:")
        if 'model' in config:
            print(f"  - Model ID: {config['model'].get('model_id', 'N/A')}")
        if 'environment' in config:
            print(f"  - Float32 precision: {config['environment'].get('float32_matmul_precision', 'N/A')}")

# 5. Language Support
print("\n5. Language Support:")
print("-" * 40)
lang_file = Path("C:/work/skin_AI_hub/skin-diagnosis-generation/app/services/language_prompts.py")
if lang_file.exists():
    content = lang_file.read_text(encoding='utf-8')
    if '"en"' in content:
        print("✓ English support found")
    if '"ko"' in content:
        print("✓ Korean support found")
    if '"vi"' in content:
        print("✓ Vietnamese support found")

# 6. API Endpoints
print("\n6. API Endpoints:")
print("-" * 40)
diagnosis_file = Path("C:/work/skin_AI_hub/skin-diagnosis-generation/app/api/endpoints/diagnosis.py")
if diagnosis_file.exists():
    content = diagnosis_file.read_text(encoding='utf-8')
    endpoints = []
    for line in content.split('\n'):
        if '@router.post' in line or '@router.get' in line:
            if '@router.post("/analyze"' in line:
                endpoints.append("POST /api/v1/analyze - Quick analysis")
            elif '@router.post("/diagnose"' in line:
                endpoints.append("POST /api/v1/diagnose - Full diagnosis")
            elif '@router.post("/diagnose-stream"' in line:
                endpoints.append("POST /api/v1/diagnose-stream - Streaming diagnosis")
            elif '@router.get("/reports' in line:
                endpoints.append("GET /api/v1/reports/{filename} - Download reports")
    
    for ep in endpoints:
        print(f"  - {ep}")

# 7. Quick Performance Test
print("\n7. Quick Performance Test:")
print("-" * 40)

# Test tensor operations speed
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Testing on GPU...")
else:
    device = torch.device('cpu')
    print("Testing on CPU...")

# Simple benchmark
size = 1024
a = torch.randn(size, size).to(device)
b = torch.randn(size, size).to(device)

# Warmup
for _ in range(3):
    c = torch.matmul(a, b)
    
# Actual timing
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
for _ in range(100):
    c = torch.matmul(a, b)
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start

print(f"Matrix multiplication (1024x1024) x100: {elapsed:.3f} seconds")
print(f"Average per operation: {elapsed/100*1000:.2f} ms")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
