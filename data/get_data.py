# tools/download_mi_datasets.py
from __future__ import annotations

import os
import time
import sys
import argparse
import random

import numpy as np
import scipy.io as scio

from moabb.paradigms import MotorImagery

# ---- IMPORTANT: datasets (MOABB) ----
from moabb.datasets import PhysionetMI, BNCI2014_001


def patch_pooch_keep_progress(
    *, max_tries=25, backoff_base=1.6, timeout=(10, 300), verify_tls=True
):
    """
    Monkey-patch pooch.downloaders.HTTPDownloader.__call__:
    - keep pooch tqdm progress bar
    - add retry/backoff & longer timeouts
    """
    import requests
    from pooch.downloaders import HTTPDownloader

    if getattr(HTTPDownloader, "_keep_progress_patched", False):
        return

    orig_call = HTTPDownloader.__call__

    def wrapped_call(self, url, output_file, pooch_obj):
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                kwargs = dict(getattr(self, "kwargs", {}))
                kwargs["timeout"] = timeout
                kwargs["verify"] = verify_tls
                kwargs.setdefault("allow_redirects", True)
                self.kwargs = kwargs
                return orig_call(self, url, output_file, pooch_obj)
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_err = e
                sleep_s = min(60.0, backoff_base ** (attempt - 1))
                print(f"[retry] attempt {attempt}/{max_tries} failed: {e} -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
        raise RuntimeError(f"Download failed after {max_tries} attempts. Last error: {last_err}")

    HTTPDownloader.__call__ = wrapped_call
    HTTPDownloader._keep_progress_patched = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _extract_session_run_group(metadata):
    """
    Returns (session_arr, run_arr, group_arr)
    - group is integer encoding of (session, run) if both exist,
      else encoding of whichever exists, else None.
    """
    cols = list(metadata.columns)
    session_arr = None
    run_arr = None

    if "session" in cols:
        session_arr = metadata["session"].to_numpy()
    if "run" in cols:
        run_arr = metadata["run"].to_numpy()

    group_arr = None
    if session_arr is not None and run_arr is not None:
        combo = (metadata["session"].astype(str) + "__" + metadata["run"].astype(str)).to_numpy()
        _, group_arr = np.unique(combo, return_inverse=True)
    elif session_arr is not None:
        _, group_arr = np.unique(metadata["session"].astype(str).to_numpy(), return_inverse=True)
    elif run_arr is not None:
        _, group_arr = np.unique(metadata["run"].astype(str).to_numpy(), return_inverse=True)

    return session_arr, run_arr, group_arr


def save_subject_mat(out_file: str, X: np.ndarray, y: np.ndarray, metadata):
    """
    Save X,y and grouping info for session-CV.
    X: (N, C, T)
    y: (N,) int labels
    """
    session_arr, run_arr, group_arr = _extract_session_run_group(metadata)

    payload = {"X": X.astype(np.float32), "y": y.astype(np.int64)}
    if session_arr is not None:
        payload["session"] = session_arr
    if run_arr is not None:
        payload["run"] = run_arr
    if group_arr is not None:
        payload["group"] = group_arr.astype(np.int64)

    scio.savemat(out_file, payload, do_compression=True)


def download_mi1_physionet(
    out_root: str,
    *,
    resample: int,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    seed: int,
):
    """
    Paper MI1:
    - 109 users
    - 64 channels
    - Task2: left vs right fist imagery
    - 3 sessions
    """

    set_seed(seed)

    dataset = PhysionetMI()
    subjects = dataset.subject_list
    print(f"[MI1] PhysionetMI subjects: {len(subjects)}")

    paradigm = MotorImagery(
        n_classes=2,
        fmin=fmin,
        fmax=fmax,
        resample=resample,
        tmin=tmin,
        tmax=tmax,
    )

    os.makedirs(out_root, exist_ok=True)

    for idx, subj in enumerate(subjects, start=1):

        out_file = os.path.join(out_root, f"{subj}.mat")

        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            print(f"[skip] MI1 subject {subj}: {out_file}")
            continue

        print(f"[MI1] Downloading subject {subj} ({idx}/{len(subjects)}) ...")

        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subj])

        y = np.asarray(y).reshape(-1)

        # -----------------------------
        # 1. 只保留 left/right trials
        # -----------------------------
        keep = np.isin(y, ["left_hand", "right_hand"])

        X = X[keep]
        y = y[keep]
        metadata = metadata.iloc[np.where(keep)[0]].reset_index(drop=True)

        # -----------------------------
        # 2. 字符串标签 → 整数标签
        # -----------------------------
        label_map = {
            "left_hand": 0,
            "right_hand": 1,
        }

        y = np.array([label_map[v] for v in y], dtype=np.int64)

        print(
            "  filtered X:", X.shape,
            "y:", y.shape,
            "meta cols:", list(metadata.columns)
        )

        save_subject_mat(out_file, X, y, metadata)

        print(f"  Saved: {out_file}")


def download_mi2_bci2a(
    out_root: str,
    *,
    resample: int,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    seed: int,
):
    """
    Paper MI2:
    - BCI Competition IV 2a
    - 9 users, 22-ch, 4 classes, 2 sessions
    MOABB: BNCI2014_001
    """
    set_seed(seed)

    dataset = BNCI2014_001()
    subjects = dataset.subject_list
    print(f"[MI2] BNCI2014_001 subjects: {len(subjects)}")

    paradigm = MotorImagery(
        n_classes=4,
        fmin=fmin,
        fmax=fmax,
        resample=resample,
        tmin=tmin,
        tmax=tmax,
    )

    os.makedirs(out_root, exist_ok=True)

    for idx, subj in enumerate(subjects, start=1):
        out_file = os.path.join(out_root, f"{subj}.mat")  # 用真实 subject id 命名
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            print(f"[skip] MI2 subject {subj}: {out_file}")
            continue

        print(f"[MI2] Downloading subject {subj} ({idx}/{len(subjects)}) ...")
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subj])
        y = np.asarray(y).reshape(-1)

        print("  X:", X.shape, "y:", y.shape, "meta cols:", list(metadata.columns))
        save_subject_mat(out_file, X, y, metadata)
        print(f"  Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".", help="output root dir")
    parser.add_argument("--which", type=str, default="MI1,MI2", help="MI1, MI2, or MI1,MI2")
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patch_pooch_keep_progress(max_tries=25, backoff_base=1.6, timeout=(10, 300), verify_tls=True)

    which = [w.strip().upper() for w in args.which.split(",") if w.strip()]
    if "MI1" in which:
        download_mi1_physionet(
            out_root=os.path.join(args.out_dir, "MI1"),
            resample=args.resample,
            fmin=args.fmin,
            fmax=args.fmax,
            tmin=args.tmin,
            tmax=args.tmax,
            seed=args.seed,
        )
    if "MI2" in which:
        download_mi2_bci2a(
            out_root=os.path.join(args.out_dir, "MI2"),
            resample=args.resample,
            fmin=args.fmin,
            fmax=args.fmax,
            tmin=args.tmin,
            tmax=args.tmax,
            seed=args.seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()