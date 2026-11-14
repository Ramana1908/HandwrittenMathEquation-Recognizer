# Handwritten Mathematical Expression Recognition

This project implements a pipeline that recognizes, parses, and solves handwritten mathematical expressions from images.

## Overview

Given a photo or drawing of a single-line expression, the system:
1. Segments the image into individual symbols using contour detection.
2. Classifies each symbol with a CNN (a Vision Transformer is trained for comparison).
3. Reconstructs the expression with a rule-based, layout-aware parser (superscripts, fractions, integrals, derivatives) and converts it to LaTeX.
4. Solves or evaluates it symbolically with SymPy.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook CNN.ipynb
```

Run all cells. The notebook loads the trained weights from `weights/` and runs on the images in `examples/`. No dataset or GPU is needed.

## Repository Contents

| Path | Description |
|---|---|
| `CNN.ipynb` | Full pipeline using a 3-layer CNN classifier |
| `ViT.ipynb` | Same pipeline using a Vision Transformer classifier |
| `weights/cnn.pth`, `weights/vit.pth` | Trained model weights (0.5 MB and 0.9 MB) |
| `examples/` | 11 handwritten expressions covering each supported feature |
| `scripts/download_data.py` | Downloads the training data from Kaggle and builds `data/` |
| `requirements.txt` | Python dependencies |

## Dataset

- **Source**: [Kaggle - Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols) (430 MB download)
- We train on a **24-class subset** of the dataset: digits `0`–`9`, `+`, `-`, `(`, `)`, `=`, `x`, `d`, `/`, `∫`, `<`, `>`, `≤`, `≥`, `≠`.
- 232,465 images in that subset, split 95/5 into 220,841 training and 11,624 validation images (fixed seed).
- Preprocessing: grayscale, resize to 45×45, adaptive mean thresholding, then inversion (white symbol on black).

The `/` class has only 199 images in the Kaggle data. The original project runs used a custom set of 1,339 slash images, which isn't published, so the committed weights recognize `/` less reliably than the originals did.

## Pipeline Details

**Symbol classifiers**
- **CNN**: three 3×3 conv layers (32 filters each) with ReLU and max-pooling, followed by a 128-unit dense layer. Trained for 3 epochs with Adam (lr 1e-3), batch size 32.
- **ViT**: 5×5 patches (81 patches per 45×45 image), 64-dim embeddings, 6 encoder layers, 8 attention heads. Trained for 4 epochs with the same optimizer settings.

**Segmentation**
- Illumination normalization (divide by a Gaussian blur), adaptive thresholding, and morphological opening.
- External contours are boxed; boxes that are too small or nearly empty are discarded.
- Boxes that overlap by at least 70% horizontally are merged, so multi-stroke symbols such as `=` and `≤` become one box.
- Each box is cropped and padded to 45×45 to match the training images. Boxes with more than 30% ink are treated as a dot (`*`, multiplication).

**Parsing**
- Symbols are read left to right. A symbol shorter than 70% of the previous base symbol and positioned above it is treated as a superscript (`x2` → `x^2`); multi-symbol exponents are grouped (`5^(8-4)`).
- Tokens are converted to LaTeX: `a / b` → `\frac{a}{b}`, `∫ … d x` → integral, `d / d x (…)` → derivative.

**Solving** (SymPy)
- Derivatives and integrals are computed.
- Inequalities (`<`, `>`, `≤`, `≥`, `≠`) are simplified.
- Equations are solved for `x`.
- Plain arithmetic is evaluated; implicit multiplication such as `2x` is supported. Other expressions fall back to SymPy's LaTeX parser.

## Results

**Symbol classification** (5% validation split; no separate held-out test set):

| Model | Validation accuracy | Training time (Apple M-series, MPS) |
|---|---|---|
| CNN | **99.17%** (3 epochs) | ~4 min |
| ViT | 98.31% (4 epochs) | ~23 min |

The original project runs reported 98.98% (CNN) and 97.04% (ViT) with a slightly different dataset (see the `/` note above).

**End-to-end** on 21 handwritten test expressions that use only the 24 supported symbols, counting an expression as correct when it is read exactly and solved correctly:

| Model | Correct |
|---|---|
| CNN | **14 / 21** |
| ViT | 8 / 21 |

The ViT's per-symbol accuracy is close to the CNN's, but it transfers worse to real handwritten equations: it often reads `1` as `d`, for example. The CNN's misses are mostly `/` read as `x`, a missed multiplication dot, and similar-looking digits.

All 11 images in `examples/` are recognized and solved correctly by the CNN, for example:
- `x^2 - 9 = 0` → `[-3, 3]`
- `∫(5+2x) dx` → `x**2 + 5*x`
- `4^2 + 3^(-1)` → `49/3`
- `-6 + 7 ≥ -9` → `True`

## Retraining

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

This downloads the Kaggle dataset with `kagglehub`, extracts it, and copies the 24 classes into `data/`. If you already have the extracted dataset, use `python scripts/download_data.py --source path/to/extracted_images`. Then set `TRAIN = True` in the first code cell of either notebook and run all cells. The new weights are saved to `weights/`, and a per-class validation report is printed.

## Limitations

- Supports single-line expressions only. Stacked fractions, square roots, subscripts, and multi-line layouts are not handled.
- Only the 24 symbol classes above are recognized; for example, `×` and `÷` are not.
- In `a / b`, only the single token after `/` is used as the denominator.
- Equations are solved for `x` only.
