"""
SHACL validation of RDF data against ontology shapes.
"""

import argparse
import os
from rdflib import Graph
from pyshacl import validate

from .config import schema_path, shapes_path


def validate_data(data_path, shapes_path_arg=None, schema_path_arg=None):
    schema = schema_path_arg or schema_path()
    shapes = shapes_path_arg or shapes_path()

    data_g = Graph()
    data_g.parse(data_path, format="turtle")
    if schema and os.path.exists(schema):
        data_g.parse(schema, format="turtle")

    shapes_g = Graph().parse(shapes, format="turtle")

    conforms, report_graph, report_text = validate(
        data_g,
        shacl_graph=shapes_g,
        inference="rdfs",
        serialize_report_graph=True,
    )
    return conforms, report_text


def main():
    parser = argparse.ArgumentParser(description="Validate RDF data with SHACL shapes.")
    parser.add_argument("--data", "-d", required=True, help="RDF data TTL file")
    parser.add_argument("--shapes", "-s", default=None, help="SHACL shapes TTL (default: ontology/shapes.ttl)")
    parser.add_argument("--schema", default=None, help="Schema TTL (default: ontology/schema.ttl)")
    args = parser.parse_args()

    conforms, report_text = validate_data(args.data, args.shapes, args.schema)

    print("\n=== SHACL VALIDATION RESULT ===")
    print("Conforms:", conforms)
    print(report_text)


if __name__ == "__main__":
    main()
