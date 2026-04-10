#!/usr/bin/env python3
"""
Run BDD JSON → RDF conversion. Use from project root.

  python scripts/run_convert_bdd.py --input data/bdd100k_labels_images_val.json --out out --sample 1000
"""

import sys
import os
import argparse

# Allow importing from src without installing the package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.converters.bdd import convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BDD JSON to RDF (Turtle).")
    parser.add_argument("--input", "-i", required=True, help="Input BDD JSON file")
    parser.add_argument("--out", "-o", default="out", help="Output directory")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()
    path = convert(args.input, args.out, args.sample)
    print("Done. Output:", path)
