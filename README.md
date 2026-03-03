## EEG_DLG: Reproducing “User Identity Protection in EEG-Based Brain–Computer Interfaces”

This repository is intended to support the reproducibility of the experiments reported in **“User Identity Protection in EEG-Based Brain–Computer Interfaces”** (hereafter referred to as the *UIP-EEG* paper).

The primary objective is to provide an organized workspace for running and documenting experiments related to **identity protection** in EEG-based brain–computer interfaces (BCIs), including (but not limited to) privacy-oriented training/evaluation settings and the management of intermediate artifacts produced during experimentation.

### Scope

The reproduction effort focuses on:
- **EEG motor imagery (MI) classification** under subject-wise evaluation protocols
- **User identity protection** objectives and threat models discussed in the UIP-EEG paper
- **Experiment traceability**, including logs, cached artifacts, and result outputs

### Repository Layout and Data Management

To ensure the repository remains lightweight and suitable for distribution, **raw EEG datasets and large intermediate files are intentionally excluded from version control**.

- `data/`: reserved for raw/downloaded datasets and processed caches  
  - Contents are ignored by default via `.gitignore`.
  - Use `.gitkeep` files if you need to preserve directory structure in Git.
- `output/`: reserved for experimental outputs (e.g., checkpoints, metrics, figures)  
  - Contents are ignored by default via `.gitignore`.
- `run_scripts/`: reserved for experiment launch scripts  
  - `*.log` files are ignored by default.

### Reproducibility Recommendations

For rigorous reproducibility consistent with academic practice:
- Record **software versions** (Python, PyTorch, CUDA/cuDNN) and hardware (GPU model).
- Fix **random seeds** for Python/NumPy/PyTorch.
- Archive the exact configuration of hyperparameters and data splits used to generate each result.

### Reference

If you use this repository in a scientific context, please cite the UIP-EEG paper:

> “User Identity Protection in EEG-Based Brain–Computer Interfaces.”
