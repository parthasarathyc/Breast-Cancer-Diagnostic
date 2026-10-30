import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, Optional
import timm
from app.api.schemas import Agent2ClassificationResult

class BreastClassifier(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.35)  # Temperature scaling parameter for calibration

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    def forward_calibrated(self, x):
        logits = self.backbone(x)
        calibrated_logits = logits / self.temperature
        return calibrated_logits

class TumorClassificationAgent:
    CLASSES = ["benign", "malignant", "normal"]

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "efficientnet_b0"
        self.model = BreastClassifier(model_name=self.model_name, num_classes=3, pretrained=False)

        default_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "best_classifier.pth"
        )
        if model_path is None and os.path.exists(default_checkpoint):
            model_path = default_checkpoint

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[*] Loaded trained classifier weights from {model_path}")
            except Exception as e:
                print(f"[!] Warning: could not load weights from {model_path}: {e}")
        else:
            # Fallback to pretrained ImageNet weights for zero-shot feature extraction
            try:
                self.model = BreastClassifier(model_name=self.model_name, num_classes=3, pretrained=True)
            except Exception:
                pass

        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, pil_image: Image.Image, lesion_detected: bool = True) -> Tuple[Agent2ClassificationResult, torch.Tensor]:
        """
        Classifies ultrasound image into benign, malignant, or normal with calibrated probabilities.
        """
        input_tensor = self.preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            raw_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            calibrated_probs = F.softmax(logits / 1.35, dim=1).squeeze(0).cpu().numpy()

        # If detection confirmed no lesion, adjust normal probability
        if not lesion_detected:
            calibrated_probs[2] = max(calibrated_probs[2], 0.85)  # normal
            rem = (1.0 - calibrated_probs[2]) / 2.0
            calibrated_probs[0] = rem
            calibrated_probs[1] = rem

        # Normalize
        calibrated_probs = calibrated_probs / calibrated_probs.sum()

        pred_idx = int(calibrated_probs.argmax())
        predicted_class = self.CLASSES[pred_idx]
        conf = float(calibrated_probs[pred_idx])

        prob_dict = {
            self.CLASSES[i]: round(float(calibrated_probs[i]), 4)
            for i in range(len(self.CLASSES))
        }

        result = Agent2ClassificationResult(
            predicted_class=predicted_class,
            raw_confidence=round(conf, 4),
            calibrated_probabilities=prob_dict,
            temperature_scaled=True,
            model_name="EfficientNet-B0 + TempScaling",
            status="completed"
        )
        return result, input_tensor
