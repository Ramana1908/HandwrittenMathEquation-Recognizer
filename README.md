# Handwritten Mathematical Expression Recognition

This project implements a pipeline that recognizes, parses, and solves handwritten mathematical expressions from images.

## Overview

Given a photo or drawing of a single-line expression, the system:
1. Segments the image into individual symbols using contour detection.
2. Classifies each symbol with a CNN (a Vision Transformer is trained for comparison).
3. Reconstructs the expression with a rule-based, layout-aware parser (superscripts, fractions, integrals, derivatives) and converts it to LaTeX.
4. Solves or evaluates it symbolically with SymPy.

## Repository Contents

| File | Description |
|---|---|
| `CNN.ipynb` | Full pipeline using a 3-layer CNN classifier |
| `ViT.ipynb` | Same pipeline using a Vision Transformer classifier, plus a per-class classification report |

## Dataset

- **Source**: [Kaggle - Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)
- We train on a **24-class subset** of the dataset: digits `0`–`9`, `+`, `-`, `(`, `)`, `=`, `x`, `d`, `/`, `∫`, `<`, `>`, `≤`, `≥`, `≠`.
- 233,605 images in that subset, split 95/5 into 221,924 training and 11,681 validation images.
- Preprocessing: grayscale, resize to 45×45, adaptive mean thresholding, then inversion (white symbol on black).

## Pipeline Details

**Symbol classifiers**
- **CNN**: three 3×3 conv layers (32 filters each) with ReLU and max-pooling, followed by a 128-unit dense layer. Trained for 3 epochs with Adam (lr 1e-3), batch size 32.
- **ViT**: 5×5 patches (81 patches per 45×45 image), 64-dim embeddings, 6 encoder layers, 8 attention heads. Trained for 4 epochs with the same optimizer settings.

**Segmentation**
- Illumination normalization (divide by a Gaussian blur), adaptive thresholding, and morphological opening.
- External contours are boxed; boxes that are too small or nearly empty are discarded.
- Boxes that overlap by at least 70% horizontally are merged, so multi-stroke symbols such as `=`, `≤`, and `÷` become one box.
- Each box is cropped and padded to 45×45 to match the training images. Boxes with more than 30% ink are treated as a dot (`*`, multiplication).

**Parsing**
- Symbols are read left to right. A symbol shorter than 70% of the previous base symbol and positioned above it is treated as a superscript (`x2` → `x^2`).
- Tokens are converted to LaTeX: `a / b` → `\frac{a}{b}`, `∫ … d x` → integral, `d / d x (…)` → derivative.

**Solving** (SymPy)
- Derivatives and integrals are computed.
- Inequalities (`<`, `>`, `≤`, `≥`, `≠`) are simplified.
- Equations are solved for `x`.
- Plain arithmetic is evaluated; other expressions fall back to `latex2sympy2`.

## Results

| Model | Validation accuracy |
|---|---|
| CNN | **98.98%** (after 3 epochs) |
| ViT | 97.04% (after 4 epochs; peak 97.84% after epoch 3) |

These are accuracies on the 5% validation split; no separate held-out test set was used. In the ViT's per-class report, the weakest classes are the rare comparison symbols (`≥` F1 0.81, `>` 0.82, `≠` 0.86), which have only 8–53 validation samples each.

End-to-end examples from the notebooks:
- CNN: handwritten `x^2 - 9 = 0` → `[-3, 3]`
- ViT: handwritten `1 + 3 ÷ 4` → `7/4`

## Running the Notebooks

Requirements: `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`, `sympy`, `latex2sympy2`, `tqdm`, `Pillow`.

The dataset, trained weights, and test images are not included in this repository. To run:
1. Download the Kaggle dataset and place the class folders under `data/` (one folder per class, as `ImageFolder` expects).
2. Run the training cells. They save `main1.pth` (CNN) or `eqn-detect1-model3.pth` (ViT).
3. Put test images in `test/test/` and set `test_filename` in the test cell.
4. Run the last cell (which defines `solve_tokens`) before the test cell.

## Limitations

- Supports single-line expressions only. Stacked fractions, square roots, subscripts, and multi-line layouts are not handled.
- In `a / b`, only the single token after `/` is used as the denominator.
- Equations are solved for `x` only.
