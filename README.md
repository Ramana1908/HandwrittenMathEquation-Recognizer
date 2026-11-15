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

## Web Demo

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860, then upload a photo, take one with the webcam, paste from the clipboard, or draw an equation on the canvas. The page shows how the equation was read, the result, the detected symbols and the 45×45 patches the classifier saw. You can switch between the CNN and the ViT.

To publish it as a free [Hugging Face Space](https://huggingface.co/spaces), log in once with `hf auth login`, then run:

```bash
python scripts/deploy_space.py YOUR_USERNAME/handwritten-math
```

The pipeline is also importable from Python:

```python
from recognizer import load_model, recognize
result = recognize('examples/quadratic.png', load_model('cnn'))
print(result['equation'], result['result'])   # x^2-9=0 [-3, 3]
```

## Repository Contents

| Path | Description |
|---|---|
| `CNN.ipynb` | Full pipeline using a 3-layer CNN classifier |
| `ViT.ipynb` | Same pipeline using a Vision Transformer classifier |
| `weights/cnn.pth`, `weights/vit.pth` | Trained model weights (0.5 MB and 0.9 MB) |
| `examples/` | 11 handwritten expressions covering each supported feature |
| `app.py` | Web demo (Gradio): upload or draw an equation |
| `recognizer.py` | The pipeline as an importable module, with the same code as the notebooks |
| `scripts/download_data.py` | Downloads the training data from Kaggle and builds `data/` |
| `scripts/deploy_space.py` | Publishes the web demo as a Hugging Face Space |
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

No threshold is a fixed pixel count. Each one scales with the image's own measured stroke width (from a distance transform) and typical symbol height, so the same settings work on small scans and large phone photos.
- **Binarization**: divide by a Gaussian blur sized to the image, which removes shadows and paper tint, then apply Otsu's threshold. Hysteresis keeps faint pencil strokes when they connect to darker ink.
- **Components**: gaps up to one stroke width are bridged, for example where a faint stroke crosses a dark one. Specks smaller than a pen dot are dropped.
- **Grouping**: pieces of one symbol are merged into one box.
  - Stacked strokes of similar width, either centred or two flat bars, form one symbol: `=`, `≤`, `≥`, `≠`.
  - Small broken-off fragments within a stroke width of a larger piece are joined to it.
  - Dots and exponents are never glued onto a neighbouring `)`.
- **Dots**: a small, solid, roundish blob inside the line of text is a multiplication dot (`*`).
- **Patches**: each symbol's own ink is centred in a 45×45 patch, keeping its aspect ratio, and thinned to 1-pixel strokes. The training images have 1-pixel strokes, so the patches match them.

**Parsing**
- All symbols are classified in one batch. If a bracket is unmatched, the classifier's second choices fix it: the symbol most likely to be the missing `(` or `)` is changed, since `(` and `1` look alike.
- Symbols are read left to right. A symbol is a superscript when, relative to the base symbol before it, it is noticeably smaller and raised, or sits almost entirely above it (`x2` → `x^2`).
- Multi-symbol exponents are grouped (`5^(8-4)`). An exponent can't start with an operator, and `-` starts one only before another raised symbol (`3^(-1)`).
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

**End-to-end** on handwritten equations that use only the 24 supported symbols, counting an equation as correct when it is read exactly (and, for the first set, also solved correctly):

| Model | 21 test equations (read + solved) | 10 more equations (read) | 16 single-symbol images |
|---|---|---|---|
| CNN, original segmentation | 14 / 21 | 5 / 10 | — |
| ViT, original segmentation | 8 / 21 | 2 / 10 | — |
| **CNN** | **21 / 21** | **10 / 10** | **16 / 16** |
| ViT | 19 / 21 | 10 / 10 | 16 / 16 |

Most of the gain comes from the scale-adaptive segmentation and the 1-pixel stroke thinning, not from the classifier. The ViT's two remaining misses both read `/` as `1`; `/` has only 199 training images.

The segmentation rules were developed while looking at the 31 equations, so expect lower accuracy on new handwriting. The single-symbol images weren't used during development.

All 11 images in `examples/` are recognized and solved correctly by both models, for example:
- `x^2 - 9 = 0` → `[-3, 3]`
- `∫(5+2x) dx` → `x**2 + 5*x`
- `4^2 + 3^(-1)` → `\frac{49}{3}`
- `-6 + 7 ≥ -9` → `True`

**Speed**: about 40 ms per image on an Apple M-series laptop, mostly segmentation. The CNN classifies all symbols of an equation in about 4 ms on a single CPU core.

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
- Decimal points aren't supported: a dot is always read as multiplication.
