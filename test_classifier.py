"""
Test script to measure classification model performance
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import time
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import json

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model setup
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)  # 6 classes
model.load_state_dict(torch.load('skin_classifier_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Data transform (same as training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# Test on sample images from each category
print("\n" + "="*60)
print("TESTING CLASSIFICATION PERFORMANCE")
print("="*60)

# Measure inference time on single images
print("\n1. Single Image Inference Time Test:")
print("-" * 40)

test_times = []
for class_name in classes:
    class_dir = os.path.join('data', class_name)
    # Get first image from each class
    images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
    if images:
        img_path = os.path.join(class_dir, images[0])
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(img_tensor)
        
        # Time measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(1).item()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        test_times.append(elapsed)
        print(f"{class_name}: {elapsed:.4f} seconds - Predicted: {classes[pred]}")

avg_time = np.mean(test_times)
print(f"\nAverage inference time per image: {avg_time:.4f} seconds")

# Test on batch of images for accuracy
print("\n2. Batch Accuracy Test (using 20% of data):")
print("-" * 40)

from skin_classifier import SkinDataset

# Load test dataset
dataset = SkinDataset('data', transform=test_transform)
test_size = int(0.2 * len(dataset))  # Use 20% for testing
test_indices = torch.randperm(len(dataset))[:test_size].tolist()
test_subset = torch.utils.data.Subset(dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"Testing on {len(test_subset)} images...")

# Evaluate
all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

total_time = time.time() - start_time

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Total processing time for {len(test_subset)} images: {total_time:.2f} seconds")
print(f"Average time per image (batch): {total_time/len(test_subset):.4f} seconds")

# Detailed classification report
print("\n3. Detailed Classification Report:")
print("-" * 40)
report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

# Print formatted report
print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 52)
for class_name in classes:
    if class_name.lower() in report:
        metrics = report[class_name.lower()]
    elif class_name in report:
        metrics = report[class_name]
    else:
        continue
    print(f"{class_name:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")

print("-" * 52)
print(f"{'Accuracy':<12} {'':<10} {'':<10} {accuracy:<10.3f}")
print(f"{'Macro avg':<12} {report['macro avg']['precision']:<10.3f} {report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f}")

# Save results
results = {
    "accuracy": accuracy,
    "avg_inference_time": avg_time,
    "batch_inference_time": total_time/len(test_subset),
    "total_test_images": len(test_subset),
    "classification_report": report
}

with open('classification_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to classification_results.json")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
