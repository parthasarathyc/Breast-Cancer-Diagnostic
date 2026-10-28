"""
Agent 1 – Tumor Detection Module (YOLOv8)
Provides modular, reusable tumor localization for breast ultrasound images.
"""

import io
import time
import os
from pathlib import Path
from typing import Union, List, Optional
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .schemas import TumorDetectionResponse, BoundingBox


class TumorDetectionAgent:
    """
    Agent 1: Detects and localizes tumor lesions in breast ultrasound images
    using YOLOv8. Handles benign, malignant, and normal (tumor-free) cases.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detection agent.
        Loads model once to avoid repetitive loading overhead.
        """
        self.model_version = "yolov8-tumor-detector-v1.0"
        self.model_path = self._resolve_model_path(model_path)
        print(f"[Agent 1] Loading YOLOv8 model from: {self.model_path}")
        self.model = YOLO(self.model_path)

    def _resolve_model_path(self, custom_path: Optional[str]) -> str:
        """Resolve weight file path with robust fallback hierarchy."""
        if custom_path and os.path.exists(custom_path):
            return custom_path

        default_trained = Path("models/detection/yolov8_tumor_detection.pt")
        if default_trained.exists():
            return str(default_trained)

        # Fallback to pretrained YOLOv8 nano baseline
        print("[Agent 1] Fine-tuned weights not found. Using baseline yolov8n.pt.")
        return "yolov8n.pt"

    def _load_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Validates and loads various image input formats into a BGR numpy array.
        Raises ValueError if format is corrupt or unsupported.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image file does not exist: {path_str}")
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {path_str}")
            return img

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from provided byte buffer.")
            return img

        elif isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input)
            if rgb_arr.ndim == 2:  # Grayscale
                return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            elif rgb_arr.shape[2] == 4:  # RGBA
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input

        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> TumorDetectionResponse:
        """
        Runs tumor detection on a single breast ultrasound image.

        Args:
            image_input: File path, bytes, PIL Image, or numpy array.
            conf_threshold: Confidence threshold for candidate detections.
            iou_threshold: NMS IoU threshold.

        Returns:
            TumorDetectionResponse with detected bounding boxes and metadata.
        """
        start_time = time.perf_counter()

        try:
            img = self._load_image(image_input)
        except Exception as e:
            return TumorDetectionResponse(
                status="error",
                has_tumor=False,
                tumors_detected=0,
                detections=[],
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                model_version=self.model_version,
                message=f"Image loading failed: {str(e)}"
            )

        h, w, c = img.shape

        # Run YOLO inference
        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections: List[BoundingBox] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Calculate normalized coordinates
                    x1, y1, x2, y2 = xyxy
                    box_w = max(0, x2 - x1)
                    box_h = max(0, y2 - y1)
                    norm_xc = (x1 + box_w / 2.0) / w
                    norm_yc = (y1 + box_h / 2.0) / h
                    norm_w = box_w / float(w)
                    norm_h = box_h / float(h)
                    area = box_w * box_h

                    detections.append(
                        BoundingBox(
                            box_xyxy=xyxy,
                            normalized_xywh=[
                                round(norm_xc, 4),
                                round(norm_yc, 4),
                                round(norm_w, 4),
                                round(norm_h, 4),
                            ],
                            confidence=round(conf, 4),
                            label="tumor",
                            area_pixels=area,
                        )
                    )

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        has_tumor = len(detections) > 0
        message = (
            f"Detected {len(detections)} potential tumor region(s)."
            if has_tumor
            else "No tumor regions detected (compatible with normal ultrasound)."
        )

        return TumorDetectionResponse(
            status="success",
            has_tumor=has_tumor,
            tumors_detected=len(detections),
            detections=detections,
            image_shape=[h, w, c],
            inference_time_ms=inference_time_ms,
            model_version=self.model_version,
            message=message
        )

    def draw_detections(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        detection_response: TumorDetectionResponse,
        color: tuple = (0, 0, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on an image."""
        img = self._load_image(image_input).copy()
        for det in detection_response.detections:
            x1, y1, x2, y2 = det.box_xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
        return img
