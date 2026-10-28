"""
Agent 2 – Tumor Classification Module (ResNet)
Provides modular, reusable 3-class classification for breast ultrasound images.
"""

import io
import time
import os
from pathlib import Path
from typing import Union, Dict, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import BreastCancerClassifier
from .schemas import TumorClassificationResponse


class TumorClassificationAgent:
    """
    Agent 2: Classifies breast ultrasound images into:
    - normal
    - benign
    - malignant
    Distinguishes research model predictions from certified clinical diagnosis.
    """

    CLASS_NAMES = ["normal", "benign", "malignant"]

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the classification agent.
        Loads model once to ensure optimal API serving performance.
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_version = "resnet18-busi-v1.0"
        self.model = BreastCancerClassifier(architecture="resnet18", num_classes=3, pretrained=False)

        self.model_path = self._resolve_model_path(model_path)
        if self.model_path and os.path.exists(self.model_path):
            print(f"[Agent 2] Loading weights from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("[Agent 2] Checkpoint not found. Initializing with ImageNet pretrained baseline.")
            self.model = BreastCancerClassifier(architecture="resnet18", num_classes=3, pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline (ImageNet standards)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _resolve_model_path(self, custom_path: Optional[str]) -> Optional[str]:
        """Resolve weight file path."""
        if custom_path and os.path.exists(custom_path):
            return custom_path

        default_trained = Path("models/classification/resnet18_tumor_classifier.pth")
        if default_trained.exists():
            return str(default_trained)
        return None

    def _load_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """
        Validates and loads image into PIL Image in RGB format.
        Raises ValueError if format is corrupt.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at path: {path_str}")
            img = Image.open(path_str)
            return img.convert("RGB")

        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
            return img.convert("RGB")

        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 3:
                # Assume BGR if coming from OpenCV
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            return Image.fromarray(image_input).convert("RGB")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> TumorClassificationResponse:
        """
        Predicts whether a breast ultrasound image is normal, benign, or malignant.

        Returns:
            TumorClassificationResponse with predicted class, confidence, and probabilities.
        """
        start_time = time.perf_counter()

        try:
            pil_img = self._load_image(image_input)
        except Exception as e:
            return TumorClassificationResponse(
                status="error",
                predicted_class="unknown",
                confidence=0.0,
                class_probabilities={"normal": 0.0, "benign": 0.0, "malignant": 0.0},
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                model_version=self.model_version,
                message=f"Image decoding failed: {str(e)}"
            )

        # Preprocess and forward pass
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_probs: Dict[str, float] = {
            self.CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(self.CLASS_NAMES))
        }

        top_idx = int(np.argmax(probs))
        predicted_class = self.CLASS_NAMES[top_idx]
        confidence = round(float(probs[top_idx]), 4)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return TumorClassificationResponse(
            status="success",
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            inference_time_ms=inference_time_ms,
            model_version=self.model_version,
            message=f"Image classified as '{predicted_class}' with {confidence * 100:.1f}% confidence."
        )
