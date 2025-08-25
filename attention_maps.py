import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model():
    classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('skin_classifier_model.pth'))
    model = model.to(device)
    model.eval()
    return model, classes

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam
        return cam.detach().cpu().numpy()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize((224, 224)))
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    return input_tensor, original_image

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Create overlay with more visible background and segment borders"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Create overlay with more visible background (lower alpha)
    overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)

    # Add segment borders for high attention areas
    # Threshold to find high attention regions
    threshold = 0.7  # Focus on top 30% attention areas
    high_attention = (heatmap > threshold * 255).astype(np.uint8)

    # Find contours of high attention areas
    contours, _ = cv2.findContours(high_attention, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw borders around high attention areas
    for contour in contours:
        # Only draw contours for reasonably sized areas
        if cv2.contourArea(contour) > 100:  # Minimum area threshold
            cv2.drawContours(overlayed, [contour], -1, (255, 255, 255), 2)  # White border
            cv2.drawContours(overlayed, [contour], -1, (0, 0, 0), 1)        # Black inner border

    return overlayed

def create_segmented_view(image, heatmap, threshold=0.7):
    """Create clean segmentation borders on original image without heatmap"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Create mask for high attention areas
    high_attention_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Start with original image (no dimming, no overlay)
    segmented = image.copy()

    # Find contours of high attention areas
    contours, _ = cv2.findContours(high_attention_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw clean borders around segments on original image
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Only significant areas
            # Draw thick white border for visibility
            cv2.drawContours(segmented, [contour], -1, (255, 255, 255), 4)
            # Draw thinner colored border inside
            cv2.drawContours(segmented, [contour], -1, (0, 255, 0), 2)  # Green border

    return segmented.astype(np.uint8)

def create_pure_segmentation(image, heatmap, threshold=0.4):
    """Create pure segmentation borders covering wider attention range (including yellow areas)"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Create mask for wider attention areas (includes yellow areas from heatmap)
    # Lower threshold to capture more areas: 0.4 instead of 0.7
    attention_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Apply morphological operations to connect nearby areas and smooth borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    attention_mask = cv2.morphologyEx(attention_mask, cv2.MORPH_CLOSE, kernel)
    attention_mask = cv2.morphologyEx(attention_mask, cv2.MORPH_OPEN, kernel)

    # Start with original image
    segmented = image.copy()

    # Find contours
    contours, _ = cv2.findContours(attention_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw clean borders covering bigger range
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Lower area threshold for more coverage
            # Draw thick border for visibility
            cv2.drawContours(segmented, [contour], -1, (255, 0, 0), 4)  # Red border, thicker

    return segmented.astype(np.uint8)

def generate_attention_maps():
    # Load model
    model, classes = load_model()
    
    # Initialize GradCAM
    gradcam = GradCAM(model, model.layer4[-1])  # Last layer of ResNet
    
    # Create output directory
    os.makedirs('attention_maps', exist_ok=True)
    
    # Generate attention maps for each class
    for class_name in classes:
        class_dir = os.path.join('data', class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Get random images from each class
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected_images = random.sample(image_files, min(5, len(image_files)))
        
        fig, axes = plt.subplots(4, 5, figsize=(25, 16))
        fig.suptitle(f'Enhanced Attention Analysis for {class_name}', fontsize=18, fontweight='bold')

        for i, img_file in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_file)

            # Preprocess image
            input_tensor, original_image = preprocess_image(img_path)

            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = output.argmax(1).item()
                confidence = torch.softmax(output, 1)[0, predicted_class].item()

            # Generate attention map
            cam = gradcam.generate_cam(input_tensor, predicted_class)

            # Create different visualizations
            overlay_with_borders = overlay_heatmap(original_image, cam, alpha=0.4)

            # Create pure segmentation (no heatmap, just borders on original image)
            pure_segmentation = create_pure_segmentation(original_image, cam)

            # Plot original image
            axes[0, i].imshow(original_image)
            axes[0, i].set_title(f'Original Image\n{img_file[:15]}...', fontsize=10, fontweight='bold')
            axes[0, i].axis('off')

            # Plot attention heatmap
            im1 = axes[1, i].imshow(cam, cmap='jet')
            axes[1, i].set_title(f'Attention Heatmap\nConf: {confidence:.3f}', fontsize=10, fontweight='bold')
            axes[1, i].axis('off')

            # Plot overlay with heatmap
            axes[2, i].imshow(overlay_with_borders)
            axes[2, i].set_title(f'Heatmap Overlay\nPred: {classes[predicted_class]}', fontsize=10, fontweight='bold')
            axes[2, i].axis('off')

            # Plot pure segmentation (clean borders on original image)
            axes[3, i].imshow(pure_segmentation)
            axes[3, i].set_title(f'Wide Segmentation\nIncludes Yellow Areas', fontsize=10, fontweight='bold')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'attention_maps/{class_name}_attention_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Generated attention maps for {class_name}")

def compare_attention_patterns():
    """Compare attention patterns across different conditions"""
    model, classes = load_model()
    gradcam = GradCAM(model, model.layer4[-1])
    
    # Get one representative image from each class
    representative_images = []
    for class_name in classes:
        class_dir = os.path.join('data', class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                img_path = os.path.join(class_dir, random.choice(image_files))
                representative_images.append((img_path, class_name))
    
    fig, axes = plt.subplots(3, len(classes), figsize=(24, 12))
    fig.suptitle('Enhanced Attention Pattern Comparison Across Skin Conditions', fontsize=18, fontweight='bold')

    for i, (img_path, class_name) in enumerate(representative_images):
        # Preprocess image
        input_tensor, original_image = preprocess_image(img_path)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(1).item()
            confidence = torch.softmax(output, 1)[0, predicted_class].item()

        # Generate attention map
        cam = gradcam.generate_cam(input_tensor, predicted_class)

        # Create enhanced visualizations
        overlay_with_borders = overlay_heatmap(original_image, cam, alpha=0.4)
        pure_segmentation = create_pure_segmentation(original_image, cam)

        # Plot original image
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f'{class_name}\nConf: {confidence:.3f}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Plot heatmap overlay
        axes[1, i].imshow(overlay_with_borders)
        axes[1, i].set_title('Heatmap Overlay', fontsize=10)
        axes[1, i].axis('off')

        # Plot pure segmentation (clean borders only)
        axes[2, i].imshow(pure_segmentation)
        axes[2, i].set_title('Wide Segmentation', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_maps/attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Generated attention pattern comparison")

def analyze_misclassifications():
    """Analyze attention maps for misclassified images"""
    model, classes = load_model()
    gradcam = GradCAM(model, model.layer4[-1])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    misclassified = []
    
    # Find misclassified images
    for true_class_idx, class_name in enumerate(classes):
        class_dir = os.path.join('data', class_name)
        if not os.path.exists(class_dir):
            continue
            
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in random.sample(image_files, min(10, len(image_files))):
            img_path = os.path.join(class_dir, img_file)
            
            # Preprocess and predict
            input_tensor, original_image = preprocess_image(img_path)
            
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = output.argmax(1).item()
                confidence = torch.softmax(output, 1)[0, predicted_class].item()
            
            if predicted_class != true_class_idx:
                misclassified.append({
                    'path': img_path,
                    'true_class': class_name,
                    'pred_class': classes[predicted_class],
                    'confidence': confidence,
                    'input_tensor': input_tensor,
                    'original_image': original_image
                })
    
    if misclassified and len(misclassified) > 0:
        # Show top misclassifications
        misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:6]

        # Handle case where we have fewer than expected misclassifications
        num_cols = len(misclassified)
        if num_cols == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            axes = axes.reshape(2, 1)  # Ensure 2D array
        else:
            fig, axes = plt.subplots(2, num_cols, figsize=(4*num_cols, 8))
            if num_cols == 1:
                axes = axes.reshape(2, 1)

        fig.suptitle('Attention Maps for Misclassified Images', fontsize=16)

        for i, item in enumerate(misclassified):
            # Generate attention map
            cam = gradcam.generate_cam(item['input_tensor'],
                                     classes.index(item['pred_class']))

            # Plot original image
            axes[0, i].imshow(item['original_image'])
            axes[0, i].set_title(f"True: {item['true_class']}\nPred: {item['pred_class']}")
            axes[0, i].axis('off')

            # Plot attention map
            axes[1, i].imshow(cam, cmap='jet')
            axes[1, i].set_title(f"Conf: {item['confidence']:.3f}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('attention_maps/misclassified_attention.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Analyzed {len(misclassified)} misclassified images")
    else:
        print("No misclassifications found in sample - model performs excellently!")

if __name__ == "__main__":
    print("Generating attention maps...")
    
    # Generate attention maps for each class
    generate_attention_maps()
    
    # Compare attention patterns
    compare_attention_patterns()
    
    # Analyze misclassifications
    analyze_misclassifications()
    
    print("All attention maps generated successfully!")
