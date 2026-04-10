"""
Build YOLO traffic-light detection and TL color classification datasets from the KG.

Pipeline:
  - Load RDF graph (schema + BDD TTL), run SPARQL for (image, box, color).
  - Train/val split by image.
  - For each image: write full image + YOLO labels to yolo_tl/; crop each TrafficLight box
    into tl_color/{split}/{red|yellow|green}/.
    - Write data/datasets/yolo_tl/data.yaml for YOLO training.

Run from project root or use scripts/run_build_tl_datasets.py.
"""

import os
import argparse
from urllib.parse import urlparse
from collections import defaultdict

import rdflib
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as pkg_config


SPARQL_TL_BOXES = """
PREFIX cv: <http://vision.semkg.org/onto/v0.1/>

SELECT ?img ?file ?box ?x ?y ?w ?h ?color
WHERE {
  ?img a cv:Image ;
       cv:filePath ?file ;
       cv:hasAnnotation ?ann .

  ?ann a cv:ObjectDetectionAnnotation ;
       cv:category "traffic light" ;
       cv:hasBox ?box ;
       cv:annotatesObject ?tl .

  ?box cv:x ?x ;
       cv:y ?y ;
       cv:width ?w ;
       cv:height ?h .

  ?tl cv:hasColor ?color .

  FILTER(?color IN (cv:RedColor, cv:YellowColor, cv:GreenColor))
}
"""

VALID_COLORS = {"red", "yellow", "green"}


def _color_from_uri(uri):
    """Map cv:RedColor etc. to 'red', 'yellow', 'green'."""
    s = str(uri).split("#")[-1].replace("Color", "").lower()
    return s if s in VALID_COLORS else None


def build_datasets(
    schema_path=None,
    bdd_ttl_path=None,
    output_root=None,
    project_root=None,
    split_ratio=0.2,
    seed=42,
):
    root = project_root or pkg_config.project_root()
    schema_path = schema_path or os.path.join(root, "ontology", "schema.ttl")
    bdd_ttl_path = bdd_ttl_path or os.path.join(root, "out", "bdd_sample_1000.ttl")
    output_root = output_root or os.path.join(root, "data", "datasets")

    yolo_img_train = os.path.join(output_root, "yolo_tl", "images", "train")
    yolo_img_val = os.path.join(output_root, "yolo_tl", "images", "val")
    yolo_label_train = os.path.join(output_root, "yolo_tl", "labels", "train")
    yolo_label_val = os.path.join(output_root, "yolo_tl", "labels", "val")
    color_root = os.path.join(output_root, "tl_color")

    for p in [yolo_img_train, yolo_img_val, yolo_label_train, yolo_label_val]:
        pkg_config.ensure_dir(p)
    for split in ["train", "val"]:
        for c in VALID_COLORS:
            pkg_config.ensure_dir(os.path.join(color_root, split, c))

    if not os.path.exists(schema_path) or not os.path.exists(bdd_ttl_path):
        raise FileNotFoundError(
            f"Schema or BDD TTL missing: {schema_path!r}, {bdd_ttl_path!r}"
        )

    g = rdflib.Graph()
    g.parse(schema_path, format="turtle")
    g.parse(bdd_ttl_path, format="turtle")
    print(f"Loaded graph with {len(g)} triples")

    results = g.query(SPARQL_TL_BOXES)
    print("results: ", results)
    image_data = defaultdict(list)

    image_data = defaultdict(list)

    for row in results:
        print(row)
        image_path = str(row.file)
        x = float(row.x)
        y = float(row.y)
        w = float(row.w)
        h = float(row.h)
        color = str(row.color).split("#")[-1].replace("Color", "").lower()

        image_data[image_path].append((x, y, w, h, color))

    print(f"Total images: {len(image_data)}")

    print(f"Total images with traffic lights: {len(image_data)}")

    image_paths = list(image_data.keys())
    train_imgs, val_imgs = train_test_split(
        image_paths, test_size=split_ratio, random_state=seed
    )

    def process_split(image_list, split):
        if split == "train":
            img_out_dir, label_out_dir = yolo_img_train, yolo_label_train
        else:
            img_out_dir, label_out_dir = yolo_img_val, yolo_label_val

        for img_path in image_list:
            full_path = os.path.join(root, img_path) if not os.path.isabs(img_path) else img_path
            if not os.path.exists(full_path):
                continue

            image = Image.open(full_path).convert("RGB")
            W, H = image.size
            img_name = os.path.basename(img_path)

            image.save(os.path.join(img_out_dir, img_name))
            label_lines = []

            for i, (x, y, w, h, color) in enumerate(image_data[img_path]):
                x_center = (x + w / 2) / W
                y_center = (y + h / 2) / H
                w_norm, h_norm = w / W, h / H
                label_lines.append(f"0 {x_center} {y_center} {w_norm} {h_norm}")

                crop = image.crop((x, y, x + w, y + h))
                crop_name = f"{img_name.replace('.jpg', '')}_{i}.jpg"
                color = urlparse(str(color)).path.split("/")[-1]
                crop.save(os.path.join(color_root, split, color, crop_name))

            label_file = img_name.replace(".jpg", ".txt")
            with open(os.path.join(label_out_dir, label_file), "w") as f:
                f.write("\n".join(label_lines))

    print("Processing train split...")
    process_split(train_imgs, "train")
    print("Processing val split...")
    process_split(val_imgs, "val")

    yolo_data_yaml = os.path.join(output_root, "yolo_tl", "data.yaml")
    yaml_content = f"""path: {os.path.abspath(os.path.join(output_root, 'yolo_tl'))}
train: images/train
val: images/val

names:
  0: traffic_light
"""
    with open(yolo_data_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Dataset creation complete. YOLO config: {yolo_data_yaml}")
    return yolo_data_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO TL and TL color datasets from KG (schema + BDD TTL)."
    )
    parser.add_argument("--schema", default=None, help="Path to schema.ttl")
    parser.add_argument("--bdd-ttl", default=None, help="Path to BDD TTL")
    parser.add_argument("--output-root", default=None, help="Output root (default: data/datasets)")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_datasets(
        schema_path=args.schema,
        bdd_ttl_path=args.bdd_ttl,
        output_root=args.output_root,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
