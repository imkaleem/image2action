#!/usr/bin/env python3
"""
Convenience script to train the ResNet18 action classifier from the KG.

Usage (from project root):

    python scripts/run_train_from_kg.py --config config/tutorial.yaml
"""

import os
import sys
import argparse


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from images2action.train_from_kg import main as train_main

    parser = argparse.ArgumentParser(
        description="Train ResNet18 from KG using a YAML config.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/tutorial.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train_main(args.config)


if __name__ == "__main__":
    main()

