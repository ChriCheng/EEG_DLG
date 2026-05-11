from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal


RAW_SFREQ = 2048.0
TARGET_SFREQ = 128.0
PRE_FIRST_EVENT_SECONDS = 5.0
EPOCH_SECONDS = 1.0

# EEGDash/NeMAR metadata lists 34 recorded channels: 32 EEG + 2 misc.
# MA1/MA2 are the two misc mastoid channels in positions 20/21, so they are
# excluded to match the paper's "32-channel EEG data".
CHANNEL_NAMES_34 = [
    "AF3", "AF4", "C3", "C4", "CP1", "CP2", "CP5", "CP6", "Cz",
    "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Fp1",
    "Fp2", "Fz", "MA1", "MA2", "O1", "O2", "Oz", "P3", "P4",
    "P7", "P8", "PO3", "PO4", "Pz", "T7", "T8",
]
MISC_CHANNELS = {"MA1", "MA2"}
EEG_CHANNEL_INDICES = [
    idx for idx, name in enumerate(CHANNEL_NAMES_34) if name not in MISC_CHANNELS
]
EEG_CHANNEL_NAMES = [CHANNEL_NAMES_34[idx] for idx in EEG_CHANNEL_INDICES]


def event_times_to_relative_seconds(events: np.ndarray) -> np.ndarray:
    datetimes = []
    for row in events:
        year, month, day, hour, minute, second = row
        whole_second = int(second)
        microsecond = int(round((float(second) - whole_second) * 1_000_000))
        datetimes.append(
            datetime(
                int(year),
                int(month),
                int(day),
                int(hour),
                int(minute),
                whole_second,
                microsecond,
            )
        )
    first = datetimes[0]
    return np.asarray([(stamp - first).total_seconds() for stamp in datetimes])


def parse_subject_id(path: Path) -> int:
    match = re.search(r"subject(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject id from {path}")
    return int(match.group(1))


def parse_session_id(name: str) -> int:
    match = re.search(r"/session(\d+)/", name)
    if match is None:
        raise ValueError(f"Cannot parse session id from {name}")
    return int(match.group(1))


def extract_mat_from_zip(zip_file: zipfile.ZipFile, name: str) -> dict:
    with zip_file.open(name) as src, tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
        tmp.write(src.read())
        tmp.flush()
        return sio.loadmat(tmp.name)


def preprocess_run(
    mat: dict,
    *,
    sos: np.ndarray,
    raw_epoch_samples: int,
    output_epoch_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(mat["data"], dtype=np.float64)
    events = np.asarray(mat["events"], dtype=np.float64)
    stimuli = np.asarray(mat["stimuli"]).reshape(-1)
    target = int(np.asarray(mat["target"]).reshape(-1)[0])

    if data.shape[0] != len(CHANNEL_NAMES_34):
        raise ValueError(f"Expected 34 channels, got data.shape={data.shape}")
    if len(events) != len(stimuli):
        raise ValueError(f"events/stimuli length mismatch: {len(events)} vs {len(stimuli)}")

    x = data[EEG_CHANNEL_INDICES]
    x = signal.sosfiltfilt(sos, x, axis=1)

    relative_seconds = event_times_to_relative_seconds(events)
    onset_samples = np.rint(
        (relative_seconds + PRE_FIRST_EVENT_SECONDS) * RAW_SFREQ
    ).astype(np.int64)

    epochs = []
    labels = []
    for onset_sample, stimulus in zip(onset_samples, stimuli):
        start = int(onset_sample)
        stop = start + raw_epoch_samples
        if start < 0 or stop > x.shape[1]:
            continue

        epoch = x[:, start:stop]
        epoch = signal.resample_poly(epoch, up=1, down=16, axis=1)
        if epoch.shape[1] != output_epoch_samples:
            epoch = signal.resample(epoch, output_epoch_samples, axis=1)

        epochs.append(epoch.astype(np.float32, copy=False))
        labels.append(1 if int(stimulus) == target else 0)

    if not epochs:
        raise ValueError("No epochs extracted from run")

    return np.stack(epochs, axis=0), np.asarray(labels, dtype=np.int64)


def process_subject(zip_path: Path, output_dir: Path, sos: np.ndarray) -> dict:
    subject_id = parse_subject_id(zip_path)
    raw_epoch_samples = int(round(EPOCH_SECONDS * RAW_SFREQ))
    output_epoch_samples = int(round(EPOCH_SECONDS * TARGET_SFREQ))

    xs = []
    ys = []
    sessions = []
    run_names = []
    per_session_counts: dict[int, int] = {}

    with zipfile.ZipFile(zip_path) as zf:
        mat_names = sorted(name for name in zf.namelist() if name.endswith("_epochs.mat"))
        for mat_name in mat_names:
            session_id = parse_session_id(mat_name)
            mat = extract_mat_from_zip(zf, mat_name)
            x_run, y_run = preprocess_run(
                mat,
                sos=sos,
                raw_epoch_samples=raw_epoch_samples,
                output_epoch_samples=output_epoch_samples,
            )

            xs.append(x_run)
            ys.append(y_run)
            sessions.append(np.full(len(y_run), session_id, dtype=np.int64))
            run_names.extend([mat_name] * len(y_run))
            per_session_counts[session_id] = per_session_counts.get(session_id, 0) + len(y_run)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    session = np.concatenate(sessions, axis=0)

    output_path = output_dir / f"{subject_id}.mat"
    sio.savemat(
        output_path,
        {
            "X": X,
            "y": y,
            "session": session,
            "subject_original_id": np.asarray([subject_id], dtype=np.int64),
            "sfreq": np.asarray([TARGET_SFREQ], dtype=np.float32),
            "channel_names": np.asarray(EEG_CHANNEL_NAMES, dtype=object),
            "class_names": np.asarray(["non-target", "target"], dtype=object),
        },
        do_compression=True,
    )

    return {
        "subject_id": subject_id,
        "output": str(output_path),
        "n_trials": int(X.shape[0]),
        "shape": list(X.shape),
        "class_counts": {
            "non_target": int((y == 0).sum()),
            "target": int((y == 1).sum()),
        },
        "session_counts": {str(k): int(v) for k, v in sorted(per_session_counts.items())},
        "n_runs": len(set(run_names)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess EPFL/Hoffmann P300 raw zip files into EEG_DLG .mat files."
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("data/P300/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/P300"))
    parser.add_argument("--low_hz", type=float, default=1.0)
    parser.add_argument("--high_hz", type=float, default=40.0)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    zip_paths = sorted(args.raw_dir.glob("subject*.zip"), key=parse_subject_id)
    if not zip_paths:
        raise FileNotFoundError(f"No subject*.zip files found in {args.raw_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sos = signal.butter(
        args.order,
        [args.low_hz, args.high_hz],
        btype="bandpass",
        fs=RAW_SFREQ,
        output="sos",
    )

    summaries = []
    for zip_path in zip_paths:
        summary = process_subject(zip_path, args.output_dir, sos)
        summaries.append(summary)
        print(
            f"subject {summary['subject_id']}: "
            f"X={tuple(summary['shape'])}, "
            f"non_target={summary['class_counts']['non_target']}, "
            f"target={summary['class_counts']['target']}"
        )

    manifest = {
        "dataset": "EPFL/Hoffmann P300",
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "raw_sfreq": RAW_SFREQ,
        "target_sfreq": TARGET_SFREQ,
        "bandpass_hz": [args.low_hz, args.high_hz],
        "epoch_seconds": [0.0, EPOCH_SECONDS],
        "channels": EEG_CHANNEL_NAMES,
        "dropped_channels": sorted(MISC_CHANNELS),
        "labels": {"non_target": 0, "target": 1},
        "subjects": summaries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
