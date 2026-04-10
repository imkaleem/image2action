"""
Project paths and ontology locations.
Resolves paths relative to the project root (parent of src/).
"""

import os

# Project root: directory containing 'src' and 'ontology'
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def project_root():
    return _ROOT

def ontology_dir():
    return os.path.join(_ROOT, "ontology")

def schema_path():
    return os.path.join(ontology_dir(), "schema.ttl")

def shapes_path():
    return os.path.join(ontology_dir(), "shapes.ttl")

def data_dir():
    return os.path.join(_ROOT, "data")

def output_dir():
    return os.path.join(_ROOT, "out")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
