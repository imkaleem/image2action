#!/usr/bin/env python3
"""
Run SHACL validation on RDF data. Use from project root.

  python scripts/run_validate_kg.py --data out/bdd_sample_1000.ttl
  python scripts/run_validate_kg.py --data out/coco_traffic_sample_500.ttl
"""

import sys
import os
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from images2action.validate_kg import main

if __name__ == "__main__":
    main()
