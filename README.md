# Computational Pathology — Tumor Classification

A deep learning pipeline for classifying histopathology tissue images as tumor or normal, 
starting with patch-level classification (PatchCamelyon) and extending toward whole-slide 
image analysis with Multiple Instance Learning (MIL).

## Project Goal

Given a tissue biopsy image, predict whether it shows cancerous tissue — starting with small 
pre-cropped image patches, and eventually scaling up to full whole-slide images (WSIs) where 
the model must learn from slide-level labels alone.

## Status

🚧 In progress — currently at the patch-classification stage (Phase 1).

## Roadmap

- [x] Project structure set up
- [ ] Phase 1: Patch-level CNN classifier on PatchCamelyon (PCam)
- [ ] Phase 2: Whole-slide image handling (tiling, tissue segmentation) on Camelyon16
- [ ] Phase 3: Multiple Instance Learning (MIL) for slide-level predictions
- [ ] Phase 4: Heatmap visualization of model attention
- [ ] Phase 5 (optional): FastAPI wrapper for image upload → prediction demo

## Datasets

- **PatchCamelyon (PCam)** — 96x96 labeled tissue patches, used for Phase 1
- **Camelyon16** — full whole-slide images with lymph node metastasis labels, used for Phase 2+

## Project Structure

computational-pathology/
├── data/ # raw/processed data and splits (gitignored)
├── notebooks/ # exploration and analysis notebooks
├── src/
│ ├── data/ # dataset loading and preprocessing
│ ├── models/ # CNN and MIL model definitions
│ ├── train.py
│ ├── evaluate.py
│ └── visualize.py
├── configs/ # hyperparameter configs
├── checkpoints/ # saved model weights (gitignored)
├── results/ # metrics and heatmaps (gitignored)
└── tests/


## Setup

```bash
git clone <your-repo-url>
cd computational-pathology
pip install -r requirements.txt
```

## Usage

_(fill in as each phase gets built — e.g., how to run training, evaluation, etc.)_

```bash
python src/train.py --config configs/config.yaml
```

## Key Concepts Used

- **Multiple Instance Learning (MIL)**: since whole-slide labels are per-slide but training 
  happens on patches, MIL learns which patches matter most without patch-level annotations.
- **Batch effect awareness**: models can accidentally learn to detect scanner/staining 
  differences instead of actual tumor signal — evaluation is designed to catch this.

## References

- [PatchCamelyon dataset](https://github.com/basveeling/pcam)
- [Camelyon16 challenge](https://camelyon16.grand-challenge.org/)
- [CLAM (Mahmood Lab)](https://github.com/mahmoodlab/CLAM) — reference for MIL architecture

---

made by --- Pranit Bharat More 🐐