#!/usr/bin/env python3
"""
Export BDD JSON to CSV. Use from project root.

  python scripts/run_bdd_to_dataframe.py --input data/bdd100k_labels_images_val.json --out data/bdd_100k_val.csv
"""

import sys
import os
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.bdd_to_dataframe import export_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BDD JSON to CSV.")
    parser.add_argument("--input", "-i", default=None, help="Input BDD JSON (default: data/bdd100k_labels_images_val.json)")
    parser.add_argument("--out", "-o", default=None, help="Output CSV path")
    args = parser.parse_args()
    out_path = export_csv(args.input, args.out)
    print("Done. Output:", out_path)
