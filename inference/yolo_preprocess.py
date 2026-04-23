"""
YOLOv8 preprocessing module for OpenVLA.
Detects target objects and injects bbox priors into VLA inference.

Usage:
    python inference/yolo_preprocess.py --image path/to/img.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_yolo_model(model_path: str = "yolov8n.pt", device: str = "cuda:0"):
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.to(device)
    return model


def detect_objects(model, image: Image.Image, confidence_threshold: float = 0.5) -> list:
    results = model(image, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            if box.conf.item() >= confidence_threshold:
                detections.append({
                    "bbox": box.xyxy[0].cpu().tolist(),
                    "confidence": box.conf.item(),
                    "class_id": int(box.cls.item()),
                    "class_name": r.names[int(box.cls.item())],
                })
    return detections


def find_target_object(detections: list, target_name: str) -> dict | None:
    target_lower = target_name.lower()
    candidates = [d for d in detections if target_lower in d["class_name"].lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["confidence"])


def bbox_to_spatial_prior(bbox: list, image_size: tuple) -> dict:
    x1, y1, x2, y2 = bbox
    w, h = image_size
    return {
        "center_x": ((x1 + x2) / 2) / w,
        "center_y": ((y1 + y2) / 2) / h,
        "width": (x2 - x1) / w,
        "height": (y2 - y1) / h,
    }


def inject_bbox_into_prompt(instruction: str, spatial_prior: dict) -> str:
    cx, cy = spatial_prior["center_x"], spatial_prior["center_y"]
    region = []
    region.append("left" if cx < 0.4 else "right" if cx > 0.6 else "center")
    region.append("top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle")
    location_hint = f" [target located at {region[0]}-{region[1]}, " \
                    f"x={cx:.2f}, y={cy:.2f}]"
    return instruction + location_hint


def visualize_detections(image: Image.Image, detections: list) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1, y1 - 12), label, fill="red")
    return img_copy


class YOLOPreprocessor:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda:0",
                 confidence_threshold: float = 0.5):
        self.model = load_yolo_model(model_path, device)
        self.confidence_threshold = confidence_threshold

    def process(self, image: Image.Image, instruction: str, target_name: str | None = None):
        detections = detect_objects(self.model, image, self.confidence_threshold)

        if target_name:
            target = find_target_object(detections, target_name)
            if target:
                spatial = bbox_to_spatial_prior(target["bbox"], image.size)
                enhanced_instruction = inject_bbox_into_prompt(instruction, spatial)
                return enhanced_instruction, detections, target
        return instruction, detections, None


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Preprocessing")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target", type=str, default=None, help="Target object name to locate")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-viz", type=str, default=None, help="Save visualization to path")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    yolo = load_yolo_model(args.model)
    detections = detect_objects(yolo, image, args.threshold)

    print(f"Detected {len(detections)} objects:")
    for d in detections:
        print(f"  {d['class_name']}: conf={d['confidence']:.3f}, bbox={d['bbox']}")

    if args.target:
        target = find_target_object(detections, args.target)
        if target:
            spatial = bbox_to_spatial_prior(target["bbox"], image.size)
            print(f"\nTarget '{args.target}' found: {spatial}")
        else:
            print(f"\nTarget '{args.target}' not found in detections.")

    if args.save_viz:
        viz = visualize_detections(image, detections)
        viz.save(args.save_viz)
        print(f"Visualization saved to {args.save_viz}")


if __name__ == "__main__":
    main()
