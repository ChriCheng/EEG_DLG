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

eg for train_user_only.py\
`
python train_user_only.py   --mi1_dir ../data/MI1   --epochs 600   --batch_size 8   --lr 2e-3   --user_hidden_dim 128   --user_dropout 0.5   --normalize channel --weight_decay 1e-4
`

eg for train_eegnet_MI1.py\
`
python train_eegnet_MI1.py   --mi1_dir ../data/MI1   --save_root ./checkpoints_MI1_LOSO_2stage   --task_epochs 100   --user_epochs 100   --batch_size 8   --task_lr 2e-3   --user_lr 1e-3  --weight_decay 1e-4 --seeds 0
`