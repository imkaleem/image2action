"""
Convert COCO Traffic validation JSON to RDF aligned with
Vision Semantic KG Ontology v0.1+.

- Ontology-aware
- Scene-level traffic light abstraction
- Explainable rule-based inference
- OWL-RL compatible, GraphDB-ready
- FIXED: Object identity collision (image-scoped URIs)
"""

import csv
import json
import os
import uuid
from collections import defaultdict
from tqdm import tqdm

from rdflib import Graph, Literal, RDF, XSD

from ..config import schema_path, ensure_dir, output_dir
from ..vocab import CV, SCHEMA, EX, COLOR_MAP


# -------------------------------------------------------------------
# Dataset URI
# -------------------------------------------------------------------

DATASET_URI = EX["COCO_Traffic_val"]


# -------------------------------------------------------------------
# COCO category_id → ontology class (+ optional color)
# -------------------------------------------------------------------

COCO_CATEGORY_MAP = {
    2:  (CV.Bicycle, None),
    3:  (CV.Car, None),
    4:  (CV.Motorcycle, None),
    6:  (CV.Bus, None),
    7:  (CV.Train, None),
    8:  (CV.Truck, None),

    # Traffic lights
    92: (CV.TrafficLight, CV.RedColor),
    93: (CV.TrafficLight, CV.GreenColor),
    10: (CV.TrafficLight, CV.YellowColor),
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def id_to_uri(prefix, identifier):
    safe = str(identifier).replace(" ", "_").replace("/", "_")
    return EX[f"{prefix}_{safe}"]


# -------------------------------------------------------------------
# Main conversion
# -------------------------------------------------------------------

def convert(input_json_path, out_dir=None, sample=None, verbose=True):
    out_dir = out_dir or output_dir()
    ensure_dir(out_dir)

    g = Graph()
    g.bind("cv", CV)
    g.bind("schema", SCHEMA)
    g.bind("ex", EX)

    # Load ontology
    if os.path.exists(schema_path()):
        g.parse(schema_path(), format="turtle")

    # Dataset node
    g.add((DATASET_URI, RDF.type, CV.Dataset))
    g.add((DATASET_URI, SCHEMA.name, Literal("coco_traffic_val")))

    # Load COCO JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if sample:
        images = images[:sample]

    annots_by_image = defaultdict(list)
    for ann in annotations:
        annots_by_image[ann["image_id"]].append(ann)

    rows = []
    stats = {
        "n_images": 0,
        "n_annotations": 0,
        "class_counts": {},
    }

    # -------------------------------------------------------------------
    # Iterate images
    # -------------------------------------------------------------------
    for img in tqdm(images, desc="Processing images"):

        img_id = img["id"]
        fname = img["file_name"]
        img_uri = id_to_uri("image", img_id)

        image_path = f"data/coco_traffic_val/images/{fname}"

        # Image
        g.add((img_uri, RDF.type, CV.Image))
        g.add((img_uri, SCHEMA.name, Literal(fname)))
        g.add((img_uri, CV.fromDataset, DATASET_URI))
        g.add((DATASET_URI, SCHEMA.hasPart, img_uri))
        g.add((img_uri, CV.filePath, Literal(image_path)))

        stats["n_images"] += 1

        # Ego vehicle
        ego_uri = id_to_uri("ego_vehicle", img_id)
        g.add((ego_uri, RDF.type, CV.EgoVehicle))
        g.add((img_uri, CV.containsObject, ego_uri))

        has_red = False
        has_yellow = False
        has_green = False
        tl_objects = []

        # -------------------------------------------------------------------
        # Annotations
        # -------------------------------------------------------------------
        for ann in annots_by_image.get(img_id, []):

            cat_id = ann.get("category_id")
            if cat_id not in COCO_CATEGORY_MAP:
                continue

            cls, color_cls = COCO_CATEGORY_MAP[cat_id]

            ann_id = ann.get("id", uuid.uuid4())

            # FIX: Image-scoped URIs
            ann_uri = id_to_uri("anno", f"{img_id}_{ann_id}")
            obj_uri = id_to_uri("object", f"{img_id}_{ann_id}")
            box_uri = id_to_uri("box", f"{img_id}_{ann_id}")

            # Annotation
            g.add((ann_uri, RDF.type, CV.ObjectDetectionAnnotation))
            g.add((img_uri, CV.hasAnnotation, ann_uri))

            # Object
            g.add((obj_uri, RDF.type, cls))
            g.add((img_uri, CV.containsObject, obj_uri))
            g.add((obj_uri, CV.describedByAnnotation, ann_uri))
            g.add((ann_uri, CV.annotatesObject, obj_uri))
            g.add((ann_uri, CV.hasCategoryClass, cls))

            stats["n_annotations"] += 1
            cname = cls.split("#")[-1]
            stats["class_counts"][cname] = stats["class_counts"].get(cname, 0) + 1

            # Bounding box
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]

                g.add((box_uri, RDF.type, CV.Box))
                g.add((box_uri, CV.x, Literal(x, datatype=XSD.float)))
                g.add((box_uri, CV.y, Literal(y, datatype=XSD.float)))
                g.add((box_uri, CV.width, Literal(w, datatype=XSD.float)))
                g.add((box_uri, CV.height, Literal(h, datatype=XSD.float)))
                g.add((ann_uri, CV.hasBox, box_uri))

            # Traffic light color
            if cls == CV.TrafficLight:
                g.add((obj_uri, CV.hasColor, color_cls))
                tl_objects.append(obj_uri)

                if color_cls == CV.RedColor:
                    has_red = True
                elif color_cls == CV.YellowColor:
                    has_yellow = True
                elif color_cls == CV.GreenColor:
                    has_green = True

            rows.append({
                "image_id": img_id,
                "file_name": fname,
                "annotation_id": ann_id,
                "category": cname,
                "dataset": "COCO_Traffic_val",
                "action": "",
            })

        # -------------------------------------------------------------------
        # Scene-level traffic light inference
        # -------------------------------------------------------------------
        if has_red:
            scene_color = "red"
            action_cls = CV.StopAction
        elif has_yellow:
            scene_color = "yellow"
            action_cls = CV.SlowAction
        elif has_green:
            scene_color = "green"
            action_cls = CV.GoAction
        else:
            scene_color = None
            action_cls = CV.UnknownAction

        if scene_color:

            scene_tl_uri = id_to_uri("scene_tl", img_id)
            g.add((scene_tl_uri, RDF.type, CV.SceneTrafficLight))
            g.add((scene_tl_uri, CV.hasColor, COLOR_MAP[scene_color]))

            g.add((img_uri, CV.hasSceneTrafficLight, scene_tl_uri))
            g.add((img_uri, CV.sceneTrafficLightColor, COLOR_MAP[scene_color]))

            if tl_objects:
                g.add((img_uri, CV.derivedFromTrafficLight, tl_objects[0]))

            g.add((img_uri, CV.inferenceMethod, Literal("rule_based_v1")))
            g.add((img_uri, CV.inferenceConfidence, Literal(1.0, datatype=XSD.float)))

        # Ego vehicle action
        g.add((ego_uri, CV.action, action_cls))

        rows.append({
            "image_id": img_id,
            "file_name": fname,
            "annotation_id": "",
            "category": "",
            "dataset": "COCO_Traffic_val",
            "action": action_cls.split("#")[-1],
        })

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    ttl_path = os.path.join(out_dir, f"coco_traffic_val_sample{sample}.ttl")
    nt_path = os.path.join(out_dir, f"coco_traffic_val_sample{sample}.nt")
    csv_path = os.path.join(out_dir, f"coco_traffic_val_sample{sample}_rows.csv")

    g.serialize(ttl_path, format="turtle")
    g.serialize(nt_path, format="nt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "annotation_id",
                "category",
                "dataset",
                "action",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: str(v) for k, v in r.items()})

    if verbose:
        print("\n=== Summary ===")
        print(f"Images processed: {stats['n_images']}")
        print(f"Annotations stored: {stats['n_annotations']}")
        print(f"Saved TTL → {ttl_path}")
        print(f"Saved NT  → {nt_path}")
        print(f"Saved CSV → {csv_path}")

    return ttl_path, nt_path, stats