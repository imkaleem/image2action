"""
Export BDD JSON annotations to a flat CSV/DataFrame.
"""

import os
import json
import pandas as pd

from .config import data_dir, ensure_dir


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def to_dataframe(data):
    records = []
    for image in data:
        image_name = image["name"]
        attrs = image.get("attributes", {})
        weather = attrs.get("weather", "")
        scene = attrs.get("scene", "")
        timeofday = attrs.get("timeofday", "")

        for label in image.get("labels", []):
            if "box2d" not in label:
                continue
            box = label["box2d"]
            label_attrs = label.get("attributes", {})
            records.append({
                "id": label.get("id"),
                "image_name": image_name,
                "weather": weather,
                "scene": scene,
                "timeofday": timeofday,
                "category": label.get("category"),
                "annotation_occluded": label_attrs.get("occluded"),
                "annotation_truncated": label_attrs.get("truncated"),
                "annotation_traffic_light_color": label_attrs.get("trafficLightColor"),
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
            })
    return pd.DataFrame(records)


def export_csv(input_json_path, output_csv_path=None):
    data = read_json(input_json_path)
    df = to_dataframe(data)
    out = output_csv_path or os.path.join(data_dir(), "bdd_100k_val.csv")
    ensure_dir(os.path.dirname(out))
    df.to_csv(out, index=False)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert BDD JSON to CSV.")
    parser.add_argument("--input", "-i", default=None, help="Input BDD JSON path")
    parser.add_argument("--out", "-o", default=None, help="Output CSV path")
    args = parser.parse_args()

    input_path = args.input or os.path.join(data_dir(), "bdd100k_labels_images_val.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    out_path = export_csv(input_path, args.out)
    print(f"BDD annotations saved to {out_path}.")


if __name__ == "__main__":
    main()
