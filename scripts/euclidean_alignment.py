from __future__ import annotations

import numpy as np
import scipy.linalg as la


def euclidean_align_trials(data: np.ndarray) -> np.ndarray:
    """Apply subject-wise Euclidean Alignment to EEG trials.

    Args:
        data: EEG trials with shape (N, C, T).

    Returns:
        Aligned trials with shape (N, C, T).
    """
    if data.ndim != 3:
        raise ValueError(f"EA expects data with shape (N, C, T), got {data.shape}")
    if data.shape[0] == 0:
        return data

    reference = np.zeros((data.shape[1], data.shape[1]), dtype=np.float64)
    for trial in data:
        trial_64 = trial.astype(np.float64, copy=False)
        reference += np.dot(trial_64, trial_64.T)
    reference /= data.shape[0]

    transform = la.inv(la.sqrtm(reference))
    transform = np.real_if_close(transform, tol=1000)
    if np.iscomplexobj(transform):
        transform = transform.real

    aligned = np.asarray([np.dot(transform, trial) for trial in data], dtype=np.float32)
    return aligned
