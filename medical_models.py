import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Union, Optional
from pathlib import Path

class MRINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=False)  # pretrained=False since we're loading weights
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)  # 4 classes

    def forward(self, x):
        return self.base(x)

class BrainMRIClassifier:
    def __init__(self, model_path: Union[str, Path]):
        self.classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: Union[str, Path]) -> MRINet:
        model = MRINet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # same preprocessing as training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def predict(self, image_path: Union[str, Path]) -> dict:
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Return prediction with confidence
            return {
                "prediction": self.classes[predicted_class],
                "confidence": float(confidence),
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.classes, probabilities[0].tolist())
                }
            }
        except Exception as e:
            return {"error": str(e)}

def brain_mri_model(image_path: Union[str, Path]) -> dict:
    """
    Wrapper function for brain MRI classification
    
    Args:
        image_path: Path to the MRI image
        
    Returns:
        dict: Prediction results including class, confidence, and all class probabilities
    """
    try:
        classifier = BrainMRIClassifier(r"E:\virtualHospital\MODELS\Brain\mri_classifier.pth")
        return classifier.predict(image_path)
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

# Alzheimer's Model
class Alzheimer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=False)  # pretrained=False since we're loading weights
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)  # 4 classes

    def forward(self, x):
        return self.base(x)

class AlzheimerClassifier:
    def __init__(self, model_path: Union[str, Path]):
        self.classes = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: Union[str, Path]) -> Alzheimer:
        model = Alzheimer()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def predict(self, image_path: Union[str, Path]) -> dict:
        try:
            img_tensor = self.preprocess_image(image_path)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "prediction": self.classes[predicted_class],
                "confidence": float(confidence),
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.classes, probabilities[0].tolist())
                }
            }
        except Exception as e:
            return {"error": str(e)}

# Chest Model
class Chest(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=False)  # pretrained=False since we're loading weights
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)  # 4 classes

    def forward(self, x):
        return self.base(x)

class ChestClassifier:
    def __init__(self, model_path: Union[str, Path]):
        self.classes = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: Union[str, Path]) -> Chest:
        model = Chest()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def predict(self, image_path: Union[str, Path]) -> dict:
        try:
            img_tensor = self.preprocess_image(image_path)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "prediction": self.classes[predicted_class],
                "confidence": float(confidence),
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.classes, probabilities[0].tolist())
                }
            }
        except Exception as e:
            return {"error": str(e)}

def alzheimer_model(image_path: Union[str, Path]) -> dict:
    """
    Wrapper function for Alzheimer's classification
    
    Args:
        image_path: Path to the brain scan image
        
    Returns:
        dict: Prediction results including class, confidence, and all class probabilities
    """
    try:
        classifier = AlzheimerClassifier(r"E:\virtualHospital\MODELS\Alzeimer\Alzehimer.pth")
        return classifier.predict(image_path)
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

def chest_model(image_path: Union[str, Path]) -> dict:
    """
    Wrapper function for chest MRI classification
    
    Args:
        image_path: Path to the chest MRI image
        
    Returns:
        dict: Prediction results including class, confidence, and all class probabilities
    """
    try:
        classifier = ChestClassifier(r"E:\virtualHospital\MODELS\chest\chestmri.pth")
        return classifier.predict(image_path)
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Test brain MRI model
    brain_result = brain_mri_model(r"E:\virtualHospital\MODELS\Brain\image3.png")
    print("Brain MRI Result:", brain_result)

    # Test Alzheimer's model
    alzheimer_result = alzheimer_model(r"E:\virtualHospital\MODELS\Alzeimer\images.jpg")
    print("Alzheimer's Result:", alzheimer_result)

    # Test chest model
    chest_result = chest_model(r"E:\virtualHospital\MODELS\chest\images.jpg")
    print("Chest MRI Result:", chest_result)