# Project: From KG-to-AI in Automotive Scenario

Instructions (solution) for building a vision Knowledge Graph from BDD/COCO annotations and validating with SHACL. Please adapt the paths according to your OS and files and folder structure.

---

## Project structure

```
images2action/
├── ontology/           # Ontology and SHACL shapes
│   ├── schema.ttl     # Vision Semantic KG ontology (cv:, schema:)
│   └── shapes.ttl     # SHACL validation shapes
├── src/
│   └── images2action/ # Python package
│       ├── config.py       # Paths (ontology, data, out)
│       ├── vocab.py        # Namespaces and controlled vocabularies
│       ├── converters/     # BDD and COCO → RDF
│       ├── validate_kg.py  # SHACL validation
│       ├── bdd_to_dataframe.py
│       ├── nb_config.py              # Helper to load YAML configs
│       ├── train_from_kg.py          # Train ResNet18 from KG (action classifier)
│       ├── build_tl_datasets.py      # KG → YOLO TL + TL color datasets
│       ├── train_tl_color_classifier.py  # Train ResNet18 TL color (red/yellow/green)
│       └── tl_inference_pipeline.py  # Image → YOLO → color → KG scoring → action
├── data/              # All data (raw inputs + generated training datasets)
├── out/               # Generated RDF (TTL/NT) and CSV
├── config/            # YAML configs (train, tl_pipeline, etc.)
├── artifacts/         # Generated models, metrics, figures, YOLO runs, downloaded images
├── scripts/           # Runnable entrypoints
│   ├── run_convert_bdd.py
│   ├── run_convert_coco.py
│   ├── run_validate_kg.py
│   ├── run_bdd_to_dataframe.py
│   ├── run_train_from_kg.py
│   ├── run_build_tl_datasets.py
│   ├── run_train_tl_color_classifier.py
│   └── run_tl_inference_pipeline.py
├── docker/            # Docker Compose + Dockerfiles
├── docs/              # Project notes/prompts
└── requirements.txt
```

---

## Setup

```bash
cd images2action
pip install -r requirements.txt
```

---

## Subtask 1: Build a KG

### 1.1 Explore the domain and schema

- **Download the datasets** from the link below and extract it into `data/`:
  - https://tubcloud.tu-berlin.de/s/d6qnRsWdkkJWf3P
- The **schema** is in `ontology/schema.ttl`. No need to copy it to the working directory; scripts resolve paths from the project root.

### 1.2 Transform data to RDF

Run from the **project root** (`images2action/`).

**BDD100K-style JSON → RDF (Turtle):**

```bash
python scripts/run_convert_bdd.py --input data/bdd_100k_val/bdd100k_labels_images_val.json --out out/
# Optional: limit to first N images
python scripts/run_convert_bdd.py --input data/bdd_100k_val/bdd100k_labels_images_val.json --out out/ --sample 1000
```

**COCO traffic JSON → RDF (Turtle + N-Triples):**

```bash
python scripts/run_convert_coco.py --input data/coco_traffic_val/instances_val_traffic.json --out out/
python scripts/run_convert_coco.py --input data/coco_traffic_val/instances_val_traffic.json --out out/ --sample 500
```

Outputs are written to `out/` (e.g. `bdd_sample_1000.ttl`, `coco_traffic_sample_500.ttl`, plus CSV summaries).

### 1.3 Validate with SHACL

Shapes are in `ontology/shapes.ttl`. Validation uses them by default:

```bash
python scripts/run_validate_kg.py --data out/bdd_sample_1000.ttl
python scripts/run_validate_kg.py --data out/coco_traffic_sample_500.ttl
```

If the data conforms, you (user) will see **Conforms: True**. To use custom shapes or schema:

```bash
python scripts/run_validate_kg.py --data out/bdd_sample_1000.ttl --shapes path/to/shapes.ttl --schema path/to/schema.ttl
```

---

## Extra: BDD JSON → CSV

To export BDD annotations to a flat CSV (e.g. for analysis):

```bash
python scripts/run_bdd_to_dataframe.py --input data/bdd100k_labels_images_val.json --out data/bdd_100k_val.csv
```

---

## Subtask 2: Train a model from the KG

Training is configuration-driven via YAML. The default config is `config/train.yaml`.

### 2.1 Configure an experiment

Edit `config/train.yaml` (or create a copy) to control:

- **data**: paths to `schema.ttl`, BDD TTL, COCO TTL
- **query.text**: SPARQL query selecting `?path` and `?action`
- **training**: `batch_size`, `val_split`, `num_epochs`, `learning_rate`, `seed`
- **output**: `model_dir`, `metrics_dir`, `figures_dir`, `run_name`

### 2.2 Run training (script, no notebook)

From the project root:

```bash
python scripts/run_train_from_kg.py --config config/train.yaml
```

This will:

- Load the RDF graphs and run the SPARQL query
- Build a dataset of image paths and action labels
- Train a ResNet18 classifier (transfer learning)
- Save:
  - Model checkpoints under `artifacts/models/`
  - Metrics JSON under `artifacts/metrics/`
  - (Best-effort) confusion-matrix figure under `artifacts/figures/`

### 2.3 Notebook usage

The notebook `notebooks/tutorial.ipynb` mirrors the same pipeline but is meant for exploration:

- Uses `config/tutorial.yaml` via `nb_config.load_experiment_config`
- Loads the RDF graphs, runs the SPARQL query, and builds the dataset
- Trains a model and can plot the confusion matrix interactively

You can load a checkpoint saved by `scripts/run_train_from_kg.py` in the notebook to inspect results and plots in a more interactive way.

---

## Traffic-light pipeline (Image → TL detector → color → scene color → action)

This pipeline matches the flow in `notebooks/build_tl_datasets.ipynb`: build datasets from the KG, train a TL color classifier, then run inference with YOLO + color model + KG-style scoring to get **sceneTrafficLightColor** and **action** (Stop / Slow / Go).

**Dependencies:** `ultralytics` (YOLO). Install with `pip install ultralytics` if needed.

### 3.1 Build YOLO TL and TL color datasets from the KG

Requires BDD TTL with traffic-light annotations and `cv:filePath` on images (e.g. from a BDD RDF export that includes file paths).

```bash
python scripts/run_build_tl_datasets.py
python scripts/run_build_tl_datasets.py --bdd-ttl out/bdd_sample_1000.ttl --output-root data/datasets --split-ratio 0.2
```

- Writes **data/datasets/yolo_tl/** (images + labels in YOLO format, train/val) and **data/datasets/yolo_tl/data.yaml**.
- Writes **data/datasets/tl_color/** with cropped traffic lights in **train/val** and **red**, **yellow**, **green** subfolders.

### 3.2 Train YOLO (traffic-light detector)

Train outside this repo with Ultralytics, e.g.:

```bash
yolo detect train data=data/datasets/yolo_tl/data.yaml model=yolov8n.pt epochs=15 imgsz=640
```

Then note the path to `artifacts/runs/detect/train/weights/best.pt` (or your run name) for inference.

### 3.3 Train the TL color classifier

```bash
python scripts/run_train_tl_color_classifier.py
python scripts/run_train_tl_color_classifier.py --epochs 10 --batch-size 32 --data-root data/datasets/tl_color
```

- Reads **data/datasets/tl_color/train** and **data/datasets/tl_color/val** (ImageFolder: red, yellow, green).
- Saves best ResNet18 checkpoint under **artifacts/models/** and a confusion matrix under **artifacts/figures/**.

### 3.4 Run inference on an image

```bash
python scripts/run_tl_inference_pipeline.py --image data/bdd_100k_val/images/b4542860-0b880bb4.jpg
python scripts/run_tl_inference_pipeline.py -i path/to/image.jpg --yolo-model artifacts/runs/detect/train7/weights/best.pt --color-model artifacts/models/tl_color_best_20260223_231908.pt
```

- Runs YOLO → crops → color classifier → KG-style scoring (center, vertical position, size) → picks best TL → **scene_color** → **action** (red→Stop, yellow→Slow, green→Go).

Optional config: **config/tl_pipeline.yaml** holds default paths for build, train, and inference; scripts can be extended to load it.

---