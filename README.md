# Computational Pathology — Tumor Detection from Histopathology Images

A deep learning pipeline that looks at digitized tissue biopsy images and predicts whether 
they show cancerous (tumor) or normal tissue. Built as a portfolio project to explore 
computational pathology — an active, real-world research and industry field (companies like 
PathAI, Paige, and Ibex are built entirely around this problem).

---

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Why This Matters](#why-this-matters)
- [Background: What is Computational Pathology?](#background-what-is-computational-pathology)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Key Technical Concepts](#key-technical-concepts)
- [Project Roadmap](#project-roadmap)
- [Datasets Used](#datasets-used)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Honest Limitations](#honest-limitations)
- [References & Inspiration](#references--inspiration)

---

## What This Project Does

In plain terms: you give it a picture of a tissue sample, and it tells you whether that 
tissue looks cancerous or normal — and optionally, *which part* of the image looks 
suspicious, shown as a heatmap.

It's not a chatbot and there's no back-and-forth conversation. It's a straightforward 
pipeline: **image in → prediction out** — similar in spirit to how a spam filter reads an 
email and labels it spam/not-spam, except here the input is tissue image pixels instead of 
words, and the output is tumor/normal instead of spam/not-spam.

## Why This Matters

Pathologists diagnose cancer by examining tissue samples under a microscope — a slow, 
expertise-heavy process. Two real-world problems this kind of tool helps address:

- **Pathologist shortage**: many regions don't have enough pathologists for the volume of 
  biopsies that need review. An automated first-pass screen (flagging "likely normal, lower 
  priority" vs. "needs urgent review") can meaningfully speed up triage in under-resourced 
  settings.
- **Consistency**: agreement between pathologists on borderline cases isn't perfect — this is 
  a documented, known issue in the field. A model can act as a second opinion to catch misses.

**Important caveat**: this project is not clinically deployable. It has no regulatory 
approval, no hospital-scale validation, and shouldn't be treated as a diagnostic tool. Its 
purpose is to demonstrate understanding of the real pipeline, its failure modes, and how to 
evaluate it properly — which is exactly what's valued in a portfolio/interview context.

## Background: What is Computational Pathology?

When a lab scans a biopsy slide, it doesn't produce a normal-sized photo — it produces a 
**whole-slide image (WSI)**, a gigapixel-scale file (often 50,000 × 50,000 pixels). Think of 
it like a zoomed-in satellite image of a city: too large to load or process all at once.

Most slides are stained using **H&E staining** (Hematoxylin & Eosin):
- **Hematoxylin** stains cell nuclei purple/blue — this is the key structure to examine
- **Eosin** stains cytoplasm and connective tissue pink — mostly background context

Cancerous cells tend to look **larger, darker, and irregularly shaped** compared to normal 
cells, which appear more organized and uniform. That visual difference is the actual signal 
the model learns to detect.

## How the Pipeline Works

Because WSIs are too large to process directly, the pipeline breaks the problem into stages:

1. **Tiling**: the giant slide image is cut into thousands of small tiles (e.g., 256×256px) — 
   like cutting a huge photo into a jigsaw puzzle so each piece is small enough to feed into a 
   model.
2. **Tile-level prediction**: a CNN (Convolutional Neural Network) looks at each individual 
   tile and predicts tumor/normal for that tile alone.
3. **Aggregation into a slide-level answer**: since we usually only know the label for the 
   *whole slide* (not each individual tile), the model needs to combine thousands of 
   tile-level guesses into one final answer — this is done using **Multiple Instance 
   Learning (MIL)**, explained below.
4. **Heatmap generation**: the model's attention weights (i.e., which tiles it considered 
   important) get visualized back onto the original slide, showing which regions drove the 
   diagnosis.

This project builds up to that full pipeline in phases — starting simple (pre-cut patches, 
no MIL needed) and working up to the harder whole-slide + MIL version.

## Key Technical Concepts

**Multiple Instance Learning (MIL)** — the core technique this project is built around. 
Since slide-level labels don't tell you which specific tiles contain the tumor, MIL treats 
each slide as a "bag" of tile instances and learns to weight/attend to the most relevant 
tiles automatically, using only the slide-level label as supervision. No pixel-level 
annotation required.

**Batch effects / shortcut learning** — a well-documented failure mode in this field: 
different hospitals and scanners stain slides slightly differently, and models can 
accidentally learn to detect *which scanner produced the image* instead of *whether the 
tissue is cancerous*. This project's evaluation is designed to watch for this rather than 
just reporting a raw accuracy number.

**Attention heatmaps** — beyond just getting a prediction, showing *where* the model is 
looking matters in this domain — it's how you sanity-check that the model is actually keying 
off tumor morphology rather than an artifact.

## Project Roadmap

- [x] Project structure set up
- [ ] **Phase 1**: Patch-level CNN classifier on PatchCamelyon (PCam) — pre-cropped tiles, 
      no tiling/MIL needed yet, just get a baseline classifier working
- [ ] **Phase 2**: Whole-slide image handling on Camelyon16 — tissue segmentation, tiling 
      real WSIs, background filtering
- [ ] **Phase 3**: Multiple Instance Learning — combine tile-level features into slide-level 
      predictions using attention-based MIL
- [ ] **Phase 4**: Heatmap visualization of model attention over whole slides
- [ ] **Phase 5 (optional)**: FastAPI wrapper so a user can upload an image and get a live 
      prediction + heatmap, as a demo-able mini product

## Datasets Used

| Dataset | Used In | Description |
|---|---|---|
| [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) | Phase 1 | 96×96 pre-cropped tissue patches, binary tumor/normal labels — good starting point since no WSI handling is needed |
| [Camelyon16](https://camelyon16.grand-challenge.org/) | Phase 2+ | Real whole-slide images with lymph node metastasis labels — the actual clinical task (breast cancer staging) |

## Project Structure

computational-pathology/
├── data/
│ ├── raw/ # original downloaded datasets (gitignored)
│ ├── processed/ # tiled/filtered data ready for training (gitignored)
│ └── splits/ # train/val/test split files (gitignored)
├── notebooks/
│ ├── 01_data_exploration.ipynb # visualize samples, check class balance
│ ├── 02_staining_check.ipynb # sanity-check H&E patterns, catch data issues early
│ └── 03_results_analysis.ipynb # confusion matrix, heatmap inspection
├── src/
│ ├── data/
│ │ ├── dataset.py # PyTorch Dataset classes
│ │ └── preprocessing.py # tiling, background filtering, normalization
│ ├── models/
│ │ ├── cnn.py # baseline CNN / pretrained backbone
│ │ └── mil.py # Multiple Instance Learning aggregator
│ ├── train.py
│ ├── evaluate.py # accuracy, AUC, confusion matrix
│ └── visualize.py # heatmap / attention visualization
├── configs/
│ └── config.yaml # hyperparameters, paths, model choice
├── checkpoints/ # saved model weights (gitignored)
├── results/
│ ├── metrics/ # logged evaluation scores per run
│ └── heatmaps/ # visual outputs
├── tests/
├── requirements.txt
└── README.md


## Setup & Installation

```bash
git clone <your-repo-url>
cd computational-pathology
pip install -r requirements.txt
```

Note: `openslide-python` (needed from Phase 2 onward, for reading real WSI files) requires 
the OpenSlide system library installed separately — see 
[openslide.org](https://openslide.org/download/) for OS-specific instructions.

## Usage

_(To be filled in as each phase is implemented)_

```bash
# Phase 1 — train baseline patch classifier
python src/train.py --config configs/config.yaml

# Evaluate a trained model
python src/evaluate.py --checkpoint checkpoints/<run_name>.pt
```

## Honest Limitations

- This is a portfolio/learning project, not a validated clinical tool.
- Trained and evaluated on public research datasets only — no guarantee of generalization to 
  real hospital data from different scanners or staining protocols.
- No IRB approval, no regulatory clearance — not intended for any real diagnostic use.

## References & Inspiration

- Lu, M.Y. et al. ["Data-efficient and weakly supervised computational pathology on 
  whole-slide images."](https://www.nature.com/articles/s41551-020-00682-w) *Nature 
  Biomedical Engineering*, 2021 — the CLAM paper, reference for the MIL architecture used here
- [CLAM (Mahmood Lab)](https://github.com/mahmoodlab/CLAM) — attention-MIL reference implementation
- [PathML (Dana-Farber AIOS)](https://github.com/Dana-Farber-AIOS/pathml) — WSI preprocessing toolkit reference
- [Camelyon16 Challenge](https://camelyon16.grand-challenge.org/) — dataset and task definition
---

made by --- Pranit Bharat More 🐐