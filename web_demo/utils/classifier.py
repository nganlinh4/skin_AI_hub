"""
Skin disease classifier utility
Loads the pre-trained ResNet50 model for skin disease classification
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Dict, List

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names from the training data
CLASSES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# Image preprocessing pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinClassifier:
    """Singleton classifier for skin disease detection"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SkinClassifier, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load the pre-trained model"""
        try:
            # Get model path (ENV override supported)
            model_path = Path(os.getenv('CLASSIFIER_MODEL_PATH', 'C:/work/skin_AI_hub/skin_classifier_model.pth'))
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Initialize ResNet50 architecture
            self._model = models.resnet50(pretrained=False)
            self._model.fc = nn.Linear(self._model.fc.in_features, len(CLASSES))
            
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(device)
            self._model.eval()
            
            print(f"Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def classify(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a skin disease image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self._model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get predicted class and confidence
            predicted_class = CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = {}
            for idx, class_name in enumerate(CLASSES):
                all_probs[class_name] = float(probabilities[0, idx].cpu().numpy())
            
            return predicted_class, confidence_score, all_probs
            
        except Exception as e:
            print(f"Error during classification: {e}")
            raise
    
    def classify_from_pil(self, pil_image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a PIL Image directly
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        try:
            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess image
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self._model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get predicted class and confidence
            predicted_class = CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = {}
            for idx, class_name in enumerate(CLASSES):
                all_probs[class_name] = float(probabilities[0, idx].cpu().numpy())
            
            return predicted_class, confidence_score, all_probs
            
        except Exception as e:
            print(f"Error during classification: {e}")
            raise

# Create global classifier instance
classifier = SkinClassifier()

def classify_image(image_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Convenience function to classify an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    return classifier.classify(image_path)

def get_classes() -> List[str]:
    """Get list of possible classes"""
    return CLASSES.copy()
