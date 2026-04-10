"""
Convert BDD-like JSON annotations to RDF (Turtle),
aligned with Vision Semantic KG Ontology v0.1+.

- Ontology-aware (OWL-RL compatible)
- Scene-level traffic light abstraction
- Explainable, traceable inference
"""

import json
import os
import uuid
from tqdm import tqdm
import pandas as pd

from rdflib import Graph, Literal, RDF, XSD

from ..config import schema_path, ensure_dir, output_dir
from ..vocab import CV, SCHEMA, EX, COLOR_MAP, VALID_TL_COLORS


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def id_to_uri(prefix, identifier):
    safe = str(identifier).replace(" ", "_").replace("/", "_")
    return EX[f"{prefix}_{safe}"]


def box_center(box2d):
    return (
        (box2d["x1"] + box2d["x2"]) / 2.0,
        (box2d["y1"] + box2d["y2"]) / 2.0,
    )


def box_area(box2d):
    return max(0.0, box2d["x2"] - box2d["x1"]) * max(0.0, box2d["y2"] - box2d["y1"])


# -------------------------------------------------------------------
# Ontology-aware category mapping
# -------------------------------------------------------------------

CATEGORY_CLASS_MAP = {
    "car": CV.Car,
    "truck": CV.Truck,
    "bus": CV.Bus,
    "train": CV.Train,
    "motorcycle": CV.Motorcycle,
    "bike": CV.Bicycle,
    "bicycle": CV.Bicycle,
    "person": CV.Person,
    "rider": CV.Rider,
    "traffic light": CV.TrafficLight,
}


# -------------------------------------------------------------------
# Traffic light heuristics (explainable)
# -------------------------------------------------------------------

def infer_traffic_light_color(ann):
    color = ann.get("attributes", {}).get("trafficLightColor")
    if color is None:
        return None
    color = str(color).lower()
    return color if color in VALID_TL_COLORS else None


def traffic_light_score(ann, img_w, img_h):
    """
    Explainable heuristic:
    - Centrality
    - Vertical relevance
    - Relative size
    - Color confidence
    """
    box = ann["box2d"]
    cx, cy = box_center(box)
    area = box_area(box)

    cx_n = cx / img_w if img_w else 0.5
    cy_n = cy / img_h if img_h else 0.5
    area_n = area / (img_w * img_h) if img_w and img_h else 0.0

    center_score = 1 - abs(cx_n - 0.5)
    vertical_score = 1 - cy_n
    size_score = min(area_n * 50, 1.0)

    score = (
        0.5 * center_score +
        0.3 * vertical_score +
        0.2 * size_score
    )

    if infer_traffic_light_color(ann):
        score += 1.0

    return score


def select_relevant_traffic_light(annotations, img_w, img_h):
    tls = [
        a for a in annotations
        if a.get("category") == "traffic light" and "box2d" in a
    ]
    if not tls:
        return None, 0.0

    ranked = sorted(
        tls,
        key=lambda a: traffic_light_score(a, img_w, img_h),
        reverse=True,
    )

    best = ranked[0]
    return best, traffic_light_score(best, img_w, img_h)


# -------------------------------------------------------------------
# Main conversion
# -------------------------------------------------------------------

def convert(input_json_path, out_dir=None, sample=None):
    out_dir = out_dir or output_dir()
    ensure_dir(out_dir)

    g = Graph()
    g.bind("cv", CV)
    g.bind("schema", SCHEMA)
    g.bind("ex", EX)

    # Load ontology
    if os.path.exists(schema_path()):
        g.parse(schema_path(), format="turtle")

    # Dataset
    dataset_uri = EX["BDD100K"]
    g.add((dataset_uri, RDF.type, CV.Dataset))
    g.add((dataset_uri, SCHEMA.name, Literal("bdd100k_val")))

    with open(input_json_path, "r") as f:
        data = json.load(f)

    if sample:
        data = data[:sample]

    rows = []

    # -------------------------------------------------------------------
    # Iterate images
    # -------------------------------------------------------------------
    for img in tqdm(data, desc="Processing images"):
        fname = img.get("name")
        if not fname:
            continue

        img_id = fname.replace(".jpg", "")
        img_uri = id_to_uri("image", img_id)

        # Image node
        g.add((img_uri, RDF.type, CV.Image))
        g.add((img_uri, SCHEMA.name, Literal(fname)))
        g.add((img_uri, CV.fromDataset, dataset_uri))
        g.add((dataset_uri, SCHEMA.hasPart, img_uri))

        image_path = f"data/bdd_100k_val/images/{fname}"
        g.add((img_uri, CV.filePath, Literal(image_path)))

        # Ego vehicle (explicit semantic role)
        ego_uri = id_to_uri("ego_vehicle", img_id)
        g.add((ego_uri, RDF.type, CV.EgoVehicle))
        g.add((img_uri, CV.containsObject, ego_uri))

        # Scene metadata
        attrs = img.get("attributes", {}) or {}
        if "weather" in attrs:
            g.add((img_uri, CV.inWeather, Literal(attrs["weather"])))
        if "timeofday" in attrs:
            g.add((img_uri, CV.inLight, Literal(attrs["timeofday"])))
        if "scene" in attrs:
            g.add((img_uri, CV.scene, Literal(attrs["scene"])))

        # -------------------------------------------------------------------
        # Object annotations
        # -------------------------------------------------------------------
        for ann in img.get("labels", []):
            if "box2d" not in ann:
                continue

            ann_id = ann.get("id", uuid.uuid4())
            ann_uri = id_to_uri("anno", ann_id)
            obj_uri = id_to_uri("object", ann_id)

            # Annotation
            g.add((ann_uri, RDF.type, CV.ObjectDetectionAnnotation))
            g.add((img_uri, CV.hasAnnotation, ann_uri))

            # Object
            g.add((img_uri, CV.containsObject, obj_uri))
            g.add((obj_uri, CV.describedByAnnotation, ann_uri))
            g.add((ann_uri, CV.annotatesObject, obj_uri))

            category = ann.get("category", "unknown").lower()
            g.add((ann_uri, CV.category, Literal(category)))

            cls = CATEGORY_CLASS_MAP.get(category)
            if cls:
                g.add((obj_uri, RDF.type, cls))
                g.add((ann_uri, CV.hasCategoryClass, cls))
            else:
                g.add((obj_uri, RDF.type, CV.Object))

            # Bounding box
            box = ann["box2d"]
            x1, y1 = float(box["x1"]), float(box["y1"])
            w = float(box["x2"] - box["x1"])
            h = float(box["y2"] - box["y1"])

            box_uri = id_to_uri("box", ann_id)
            g.add((box_uri, RDF.type, CV.Box))
            g.add((box_uri, CV.x, Literal(x1, datatype=XSD.float)))
            g.add((box_uri, CV.y, Literal(y1, datatype=XSD.float)))
            g.add((box_uri, CV.width, Literal(w, datatype=XSD.float)))
            g.add((box_uri, CV.height, Literal(h, datatype=XSD.float)))
            g.add((ann_uri, CV.hasBox, box_uri))

            # Traffic light color (object-level)
            if category == "traffic light":
                raw_color = ann.get("attributes", {}).get("trafficLightColor")
                color = str(raw_color).lower() if raw_color else None

                if color in COLOR_MAP:
                    g.add((obj_uri, CV.hasColor, COLOR_MAP[color]))
                else:
                    g.add((obj_uri, CV.hasColor, CV.NAColor))

        # -------------------------------------------------------------------
        # Scene-level traffic light inference
        # -------------------------------------------------------------------
        img_w = img.get("width", 1280)
        img_h = img.get("height", 720)

        best_tl, score = select_relevant_traffic_light(
            img.get("labels", []),
            img_w,
            img_h
        )

        scene_color = infer_traffic_light_color(best_tl) if best_tl else None

        if scene_color in COLOR_MAP:
            # Scene abstraction
            scene_tl_uri = id_to_uri("scene_tl", img_id)
            g.add((scene_tl_uri, RDF.type, CV.SceneTrafficLight))
            g.add((scene_tl_uri, CV.hasColor, COLOR_MAP[scene_color]))

            g.add((img_uri, CV.hasSceneTrafficLight, scene_tl_uri))
            g.add((img_uri, CV.sceneTrafficLightColor, COLOR_MAP[scene_color]))

            if best_tl.get("id"):
                g.add((
                    img_uri,
                    CV.derivedFromTrafficLight,
                    id_to_uri("object", best_tl["id"]),
                ))

            g.add((img_uri, CV.inferenceMethod, Literal("heuristic_tl_v1")))
            g.add((img_uri, CV.inferenceConfidence, Literal(min(score / 2.0, 1.0), datatype=XSD.float)))

        # Ego vehicle action inference
        if scene_color == "red":
            action = CV.StopAction
        elif scene_color == "yellow":
            action = CV.SlowAction
        elif scene_color == "green":
            action = CV.GoAction
        else:
            action = CV.UnknownAction

        g.add((ego_uri, CV.action, action))

        rows.append({
            "image": fname,
            "scene_color": scene_color or "",
            "action": action.split("#")[-1],
        })

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    ttl_path = os.path.join(out_dir, "bdd.ttl")

    g.serialize(ttl_path, format="turtle")
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "bdd_summary.csv"),
        index=False
    )

    return ttl_path
