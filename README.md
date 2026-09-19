<h1 align="center">?? Computational Pathology</h1>
<h3 align="center">Tumor Detection from Histopathology Images using Deep Learning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Computational%20Pathology-blueviolet?style=for-the-badge"/>
</p>

<p align="center">
  A full deep learning pipeline that analyzes digitized tissue biopsy images (H&E stained slides) and predicts whether a tissue sample is <strong>cancerous (tumor)</strong> or <strong>normal</strong> — and highlights <em>which regions</em> drove the prediction via attention heatmaps.
</p>

---

## ?? Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Why This Matters](#-why-this-matters)
- [Background: Computational Pathology 101](#-background-computational-pathology-101)
- [Pipeline Overview](#-pipeline-overview)
- [Key Technical Concepts](#-key-technical-concepts)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Project Roadmap](#-project-roadmap)
- [Results](#-results)
- [Honest Limitations](#-honest-limitations)
- [References & Inspiration](#-references--inspiration)

---

## ?? What This Project Does

In plain terms: **give it a picture of a tissue sample ? it tells you if the tissue is cancerous or normal.**

Beyond just a prediction, it also generates an **attention heatmap** that highlights *which part* of the image the model found suspicious — so you can sanity-check whether the model is actually looking at real tumor morphology (irregular nuclei, dense cell packing) rather than some artifact.

Think of it like a spam filter for tissue slides:

```
Email  ? spam filter       ? spam / not-spam
Slide  ? this pipeline     ? tumor / normal  +  heatmap of suspicious regions
```

This is the actual problem that companies like **PathAI**, **Paige**, and **Ibex** are built around — automated pathology is a real, active industry.

---

## ?? Why This Matters

Pathologists diagnose cancer by examining tissue samples under a microscope — a slow, highly expertise-dependent process. Two real problems this addresses:

| Problem | How This Helps |
|---|---|
| **Pathologist shortage** | Many regions lack enough pathologists for biopsy volume. An automated first-pass screen can flag "likely normal" vs. "needs urgent review" — speeding up triage in under-resourced settings. |
| **Inter-observer variability** | Agreement between pathologists on borderline cases is not perfect — a documented, known issue. A model can act as a calibrated second opinion. |

> ?? **Important caveat**: This is a **portfolio/learning project**, not a clinical tool. It has no regulatory approval, no hospital-scale validation, and should never be used as a diagnostic instrument. Its purpose is to demonstrate understanding of the real pipeline, failure modes, and rigorous evaluation.

---

## ?? Background: Computational Pathology 101

### Whole-Slide Images (WSIs)

When a lab scans a biopsy slide, it does not produce a normal-sized photo — it produces a **gigapixel-scale file** (often 50,000 × 50,000 pixels). Think of it like a satellite image of a city: too large to load or process all at once. These are called **Whole-Slide Images (WSIs)**.

### H&E Staining

Tissue is chemically stained before scanning using **Hematoxylin & Eosin (H&E)** — the gold standard in pathology:

| Stain | Color | What It Highlights |
|---|---|---|
| **Hematoxylin** | Purple / Blue | Cell nuclei — key diagnostic structure |
| **Eosin** | Pink | Cytoplasm & connective tissue — background context |

### What Cancer Looks Like

Cancerous cells tend to look **larger, darker, and more irregularly shaped** compared to normal cells, which appear more organized and uniform. That visual difference in nuclear morphology is the actual signal the model learns to detect.

---

## ?? Pipeline Overview

Because WSIs are too large to process directly, the pipeline is broken into stages:

```
  WSI (gigapixel image)
       |
       v
  [ Tiling ]  ------------------  Cut into thousands of 96x96 or 256x256 tiles
       |                           + background / whitespace filtering
       v
  [ CNN Backbone ]  ------------  Extract feature vector from each tile
  (ResNet / EfficientNet)          Tile-level tumor probability (Phase 1)
       |
       v
  [ MIL Aggregator ]  ----------  Attention-weighted combination of tile features
  (Attention-MIL)                  Produces single slide-level prediction
       |
       v
  [ Heatmap Output ]  ----------  Attention scores overlaid back onto original slide
                                   Shows which regions drove the diagnosis
```

This project builds the full pipeline **incrementally in phases** — starting simple (pre-cut patches, no MIL) and working up to the complete whole-slide version.

---

## ?? Key Technical Concepts

### Multiple Instance Learning (MIL)

The core technique this project is built around.

**The problem**: We only have a label for the *whole slide* (tumor / normal) — not for each individual tile. We do not know *which* tiles contain the tumor.

**MIL solution**: Treat each slide as a "bag" of tile instances. The model learns to **attend to** (weight) the most relevant tiles automatically, using only the slide-level label as supervision. No pixel-level annotation required — which is crucial since annotating every tile would require thousands of hours of pathologist time.

```
Slide label: TUMOR
  Bag: [tile_1, tile_2, tile_3, ..., tile_N]
           |        |       |
       attention weights learned automatically
           |
       slide-level prediction: TUMOR
```

### Attention Heatmaps

Beyond just getting a prediction, showing *where* the model is looking matters in this domain. It lets you:
- Sanity-check the model is keying off real tumor morphology
- Build trust with domain experts
- Catch shortcut learning early

### Batch Effects & Shortcut Learning

A well-documented failure mode in computational pathology: different hospitals and scanners stain slides slightly differently. Models can accidentally learn to detect *which scanner produced the image* instead of *whether the tissue is cancerous*. This project's evaluation is designed to detect this, rather than just reporting raw accuracy.

---

## ?? Datasets

### Phase 1 — Kaggle Histopathologic Cancer Detection (PCam)

| Property | Value |
|---|---|
| **Source** | [Kaggle: Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) |
| **Based on** | PatchCamelyon (PCam) — derived from Camelyon16 WSIs |
| **Task** | Binary classification: tumor patch vs. normal patch |
| **Image size** | 96 × 96 pixels, `.tif` format |
| **Train set** | 13,590 labeled images |
| **Test set** | 57,458 images |
| **Label** | `1` = tumor tissue present in center 32×32 region, `0` = normal |
| **Why start here** | Pre-cropped patches — no WSI handling or tiling needed yet |

> **Label definition**: A patch is labeled positive (tumor) only if the **center 32×32 pixel region** contains at least one pixel of tumor tissue. The model cannot just look at the patch edges — it must focus on the center.

### Phase 2+ — Camelyon16

| Property | Value |
|---|---|
| **Source** | [Camelyon16 Grand Challenge](https://camelyon16.grand-challenge.org/) |
| **Task** | Detect lymph node metastases in breast cancer WSIs |
| **Format** | Full whole-slide images (`.tif` / `.svs`), gigapixel-scale |
| **Why use this** | Real clinical task — requires full tiling + MIL pipeline |

---

## ?? Project Structure

```
computational-pathology/
+-- data/
¦   +-- raw/                          # Original downloaded datasets (gitignored)
¦   ¦   +-- train/                    # 13,590 labeled .tif patches (Phase 1)
¦   ¦   +-- test/                     # 57,458 .tif patches for inference
¦   ¦   +-- sample_submission.csv     # Kaggle submission format
¦   +-- processed/                    # Tiled/filtered data ready for training (gitignored)
¦   +-- splits/                       # Train/val/test split CSV files (gitignored)
¦
+-- notebooks/
¦   +-- 01_data_exploration.ipynb     # Visualize samples, class balance, pixel stats
¦   +-- 02_staining_check.ipynb       # Sanity-check H&E patterns, catch data issues
¦   +-- 03_results_analysis.ipynb     # Confusion matrix, AUC curves, heatmap inspection
¦
+-- src/
¦   +-- data/
¦   ¦   +-- dataset.py                # PyTorch Dataset classes for PCam and WSI
¦   ¦   +-- preprocessing.py          # Tiling, background filtering, stain normalization
¦   +-- models/
¦   ¦   +-- cnn.py                    # Baseline CNN / pretrained backbone (ResNet, EfficientNet)
¦   ¦   +-- mil.py                    # Attention-based MIL aggregator
¦   +-- train.py                      # Training loop with logging and checkpointing
¦   +-- evaluate.py                   # Accuracy, AUC, confusion matrix, per-class metrics
¦   +-- visualize.py                  # Attention heatmap generation and overlay
¦
+-- configs/
¦   +-- config.yaml                   # All hyperparameters, paths, model choice
¦
+-- checkpoints/                      # Saved model weights per run (gitignored)
¦
+-- results/
¦   +-- metrics/                      # JSON/CSV logs of evaluation scores per run
¦   +-- heatmaps/                     # Visual attention outputs saved as images
¦
+-- tests/
¦   +-- test_processing.py            # Unit tests for preprocessing pipeline
¦
+-- requirements.txt
+-- README.md
```

---

## ??? Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (CPU works for Phase 1 at small scale)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/computational-pathology.git
cd computational-pathology
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset setup

**Phase 1 (PCam / Kaggle)** — Download from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data) and place in `data/raw/`:
```
data/raw/
+-- train/              <- labeled .tif images
+-- test/               <- unlabeled .tif images
+-- sample_submission.csv
```

**Phase 2+ (Camelyon16)** — Requires the [OpenSlide system library](https://openslide.org/download/) installed separately:
```bash
# After installing OpenSlide system lib:
pip install openslide-python
```

---

## ?? Usage

> *(Filled in incrementally as each phase is implemented)*

### Train — Phase 1 patch classifier
```bash
python src/train.py --config configs/config.yaml
```

### Evaluate a trained model
```bash
python src/evaluate.py --checkpoint checkpoints/<run_name>.pt
```

### Generate attention heatmaps (Phase 4+)
```bash
python src/visualize.py --checkpoint checkpoints/<run_name>.pt --slide data/raw/<slide_id>.tif
```

---

## ??? Project Roadmap

| Phase | Description | Status |
|---|---|---|
| **Setup** | Project structure, README, config, requirements | ? Done |
| **Phase 1** | Patch-level CNN classifier on PCam — baseline binary classifier, no MIL needed | ?? In Progress |
| **Phase 2** | Whole-slide image handling — tissue segmentation, tiling real WSIs, background filtering | ? Planned |
| **Phase 3** | Multiple Instance Learning — attention-MIL aggregator for slide-level predictions | ? Planned |
| **Phase 4** | Heatmap visualization — attention weights overlaid on original slides | ? Planned |
| **Phase 5** | FastAPI demo — upload an image, get prediction + heatmap in a browser | ? Optional |

---

## ?? Results

> *(To be populated as training runs are completed)*

| Phase | Model | Dataset | AUC | Accuracy | Notes |
|---|---|---|---|---|---|
| Phase 1 | — | PCam | — | — | Baseline TBD |

---

## ?? Honest Limitations

- **Not a clinical tool.** No regulatory approval, no IRB, not intended for real diagnostic use.
- **Public datasets only.** No guarantee of generalization to real hospital data from different scanners or staining protocols.
- **Shortcut learning risk.** Scanner/hospital-specific staining differences could be learned instead of true tumor morphology — evaluation is designed to watch for this.
- **Scale.** Phase 1 uses a 13k-image subset. Full PCam has ~220k images; Camelyon16 adds real WSI complexity.

---

## ?? References & Inspiration

- Lu, M.Y. et al. [*"Data-efficient and weakly supervised computational pathology on whole-slide images."*](https://www.nature.com/articles/s41551-020-00682-w) *Nature Biomedical Engineering*, 2021 — the **CLAM paper**, reference for the MIL architecture used here.
- [**CLAM** (Mahmood Lab)](https://github.com/mahmoodlab/CLAM) — attention-MIL reference implementation.
- [**PathML** (Dana-Farber AIOS)](https://github.com/Dana-Farber-AIOS/pathml) — WSI preprocessing toolkit reference.
- [**Camelyon16 Challenge**](https://camelyon16.grand-challenge.org/) — dataset and task definition.
- [**PatchCamelyon**](https://github.com/basveeling/pcam) — original PCam benchmark.

---

<p align="center">Made with ?? by <strong>Pranit Bharat More</strong> ??</p>
