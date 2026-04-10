#!/usr/bin/env python3
"""
Run COCO traffic JSON → RDF conversion. Use from project root.

  python scripts/run_convert_coco.py --input data/coco_traffic/instances_val_traffic.json --out out --sample 500
"""

import sys
import os
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.converters.coco import convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO traffic JSON to RDF.")
    parser.add_argument("--input", "-i", required=True, help="Input COCO JSON file")
    parser.add_argument("--out", "-o", default="out", help="Output directory")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()
    convert(args.input, args.out, args.sample)
