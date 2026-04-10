"""
End-to-end traffic-light pipeline: Image → YOLO → crops → color classifier → KG-style scoring → scene color → action.

Pipeline:
  1. Run YOLO to get N bounding boxes (traffic lights).
  2. Crop each box and run the color classifier → list of (box, color).
  3. KG-based scoring to pick the "relevant" traffic light → sceneTrafficLightColor.
  4. Action rule: red → Stop, yellow → Slow, green → Go.

Run from project root or use scripts/run_tl_inference_pipeline.py. Can be imported and call infer_image(path).
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

from . import config as pkg_config

VALID_TL_COLORS = ["red", "yellow", "green"]
COLOR_TO_ACTION = {"red": "Stop", "yellow": "Slow", "green": "Go"}


def traffic_light_score(box: Tuple[float, float, float, float], img_w: float, img_h: float) -> float:
    """
    KG-style relevance score: prefer center, higher in frame, and reasonable size.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)

    cx_n = cx / img_w if img_w else 0.5
    cy_n = cy / img_h if img_h else 0.5
    area_n = area / (img_w * img_h) if (img_w and img_h) else 0.0

    center_score = 1 - abs(cx_n - 0.5)
    vertical_score = 1 - cy_n
    size_score = min(area_n * 50, 1.0)
    return 0.5 * center_score + 0.3 * vertical_score + 0.2 * size_score


def load_color_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    img_size: int = 224,
):
    """Load ResNet18 color classifier from checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model, transform, class_names, device


def infer_image(
    image_path: str,
    detector: Optional[YOLO] = None,
    color_model=None,
    color_transform=None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    yolo_model_path: Optional[str] = None,
    color_model_path: Optional[str] = None,
    img_size: int = 224,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run full pipeline on one image.

    Returns dict with keys: scene_color, action, detections (list of {box, color, score}).
    Returns None if no traffic lights detected.
    """
    root = pkg_config.project_root()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy-load models if not provided
    if detector is None:
        yolo_path = yolo_model_path or os.path.join(root, "runs", "detect", "train", "weights", "best.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        detector = YOLO(yolo_path)

    if color_model is None or color_transform is None or class_names is None:
        color_path = color_model_path or os.path.join(root, "artifacts", "models", "tl_color_best.pt")
        if not os.path.exists(color_path):
            raise FileNotFoundError(f"Color model not found: {color_path}")
        color_model, color_transform, class_names, device = load_color_model(color_path, device, img_size)

    full_path = os.path.join(root, image_path) if not os.path.isabs(image_path) else image_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")

    image = Image.open(full_path).convert("RGB")
    img_w, img_h = image.size

    results = detector(full_path)[0]
    if len(results.boxes) == 0:
        if verbose:
            print("No traffic lights detected.")
        return None

    detections = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        crop = image.crop((x1, y1, x2, y2))
        input_tensor = color_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = color_model(input_tensor)
            _, pred = torch.max(output, 1)
        predicted_color = class_names[pred.item()]
        score = traffic_light_score((x1, y1, x2, y2), img_w, img_h)

        detections.append({
            "box": (x1, y1, x2, y2),
            "color": predicted_color,
            "score": float(score),
        })

    detections.sort(key=lambda x: x["score"], reverse=True)
    best_tl = detections[0]
    scene_color = best_tl["color"]
    action = COLOR_TO_ACTION.get(scene_color, "Unknown")

    if verbose:
        print("\nDetections:")
        for d in detections:
            print(d)
        print("\nScene Traffic Light Color:", scene_color)
        print("Vehicle Action:", action)

    return {
        "scene_color": scene_color,
        "action": action,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run TL pipeline: YOLO → color classifier → KG scoring → action."
    )
    parser.add_argument("--image", "-i", required=True, help="Path to image (relative to project root or absolute)")
    parser.add_argument("--yolo-model", default=None, help="Path to YOLO best.pt")
    parser.add_argument("--color-model", default=None, help="Path to tl_color_best.pt")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    result = infer_image(
        args.image,
        yolo_model_path=args.yolo_model,
        color_model_path=args.color_model,
        verbose=not args.quiet,
    )
    if result is None:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
