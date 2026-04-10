#!/usr/bin/env python3
"""
Train the traffic-light color classifier (ResNet18) on data/datasets/tl_color.

Usage (from project root):
  python scripts/run_train_tl_color_classifier.py
  python scripts/run_train_tl_color_classifier.py --epochs 10 --batch-size 32
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.train_tl_color_classifier import main

if __name__ == "__main__":
    main()
