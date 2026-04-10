#!/usr/bin/env python3
"""
Build YOLO TL and TrafficLight color datasets from the KG (schema + BDD TTL).

Usage (from project root):
  python scripts/run_build_tl_datasets.py
  python scripts/run_build_tl_datasets.py --bdd-ttl out/bdd_sample_1000.ttl --output-root data/datasets
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.build_tl_datasets import main

if __name__ == "__main__":
    main()
