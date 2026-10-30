# Multimodal Fusion: a reproducible image–text experiment

A compact, **fully runnable and synthetic** multimodal classification project. It implements a CNN image encoder, a controlled-vocabulary text encoder, and concatenation-based fusion using **one Python source file**. The reference report was the design brief; all results below come from an independent execution of this repository's code.

## Measured results (seed `20261030`; 450 held-out test examples)

| Condition | Test accuracy | Correct / 450 |
| --- | ---: | ---: |
| Image-only | 80.00% | 360 |
| Text-only | 77.11% | 347 |
| Joint fusion | 100.00% | 450 |
| Trained fusion, text embedding zeroed at inference | 47.78% | 215 |

**Important:** This is an intentionally engineered XOR-complementary synthetic task. Perfect fusion on its single held-out split is **not** evidence of real-world generalization. These values differ from the example values in the supplied design PDF because they are measured on a fresh, documented implementation rather than copied from that document.

## Run in Google Colab (no LaTeX required)

1. Open [`colab_demo.ipynb`](colab_demo.ipynb) in Google Colab using the GitHub import option.
2. Execute **Runtime → Run all**. The notebook installs dependencies, runs the training, prints the measured test metrics, and displays the figures inline.
3. The notebook requires the GitHub repository URL: set `REPO_URL` in its first code cell to your uploaded repository's HTTPS URL (for example, `https://github.com/USERNAME/REPOSITORY.git`). Alternatively, upload the ZIP to Colab and run `python project.py` after extracting it.

No dataset download, API key, pretrained weights, TeX installation, or GPU is required. Colab needs an internet connection only for installing Python dependencies and cloning the GitHub repository.

## Run locally

Python 3.10+ recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python project.py
python project.py --check
```

The Python program **does not contain or generate LaTeX**; it creates the dataset, trains/evaluates models, and writes `figures/` and `results/`. To rebuild the optional report independently, run `pdflatex -interaction=nonstopmode -halt-on-error report.tex` twice. The checked-in `report.tex` and `report.pdf` document the supplied reference run; retraining does *not* automatically update their numerical text.

## Layout

- `project.py`: dataset generation, PyTorch architectures, training, testing, figures and consistency checks, with no LaTeX.
- `colab_demo.ipynb`: Colab entry point for run-and-view results.
- `report.pdf`: polished research report for the checked-in reference run.
- `report.tex`: standalone, editable LaTeX source for the existing report.
- `figures/`: six real-data/model figures included and regenerated on each run.
- `results/metrics.json`: configuration, split IDs, true labels, predictions, confusion matrices, summary metrics and training logs.
- `results/*_weights.pt`: trained checkpoints for the three models.
- `requirements.txt`: Python dependencies.

## Experimental controls

The 1,800 generated examples are shuffled once with an independent fixed RNG. The split sizes are 1,260 train / 90 validation / 450 test. Every model uses the same split, 12 training epochs, minibatches of 64, Adam with learning rate 0.003, and cross-entropy. Validation is tracked and no model checkpoint is chosen using the held-out test results. One pair of independent hidden bits controls a label via XOR; each modality reveals only one bit plus an independently noisy label hint.

The text ablation sets the fusion model's **learned text embedding** to zero at inference. It does not retrain the model and should not be confused with the separately trained image-only baseline. Precision, recall and F1 are macro-averaged. Figures (including PCA, confusion matrix and learning trajectories) are based on generated examples or actual inference/training outputs, never hand-drawn hypothetical scores.

## Verification

`python project.py --check` recomputes all four accuracies and confusion matrices from saved test labels and predictions; also checks split sizes, required figures, and 12 recorded training epochs. The validation procedure is intentionally a **consistency check**, not a claim that a single synthetic split is statistically representative. For an independent replication, run `python project.py --run` again and compare `results/metrics.json` (environment version text may differ).

## Limitations and ethics

These are 24×24 synthetic grayscale arrays with keyword-bag text. The text is not free-form clinical language and the images are not patient scans. There is no clinical use case, external evaluation, confidence interval, multi-seed uncertainty analysis, or robustness benchmark. The task was designed so fusion could learn cross-modal XOR and therefore is best understood as a teaching/research demonstration of complementary evidence, not a deployment-ready multimodal system.

## Source design

Rebuilt from the user-provided *Multimodal Fusion — Detailed Technical Project Report* (October 2026). Its reported historical/example results are **not independently verifiable from the PDF alone**; this repository instead supplies inspectable code, outputs, and model checkpoints for the newly executed study.
