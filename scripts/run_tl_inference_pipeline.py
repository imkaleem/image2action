#!/usr/bin/env python3
"""
Run the full TL pipeline on an image: YOLO → color classifier → KG scoring → action.

Usage (from project root):
  python scripts/run_tl_inference_pipeline.py --image data/bdd_100k_val/images/b4542860-0b880bb4.jpg
  python scripts/run_tl_inference_pipeline.py -i path/to/image.jpg --yolo-model artifacts/runs/detect/train7/weights/best.pt --color-model artifacts/models/tl_color_best_20260223_231908.pt
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.tl_inference_pipeline import main

if __name__ == "__main__":
    main()
