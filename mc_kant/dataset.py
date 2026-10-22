"""
Phase 1 — Multimodal Signal Ingestion & Preprocessing
======================================================
Provides two Dataset abstractions:

  SleepEpochDataset   – Reads already-preprocessed SN_*.npz files (the output
                         of the existing iSLEEPS_preprocess_v2 pipeline).  This
                         is the default fast-path: channels are normalised and
                         returned as independent, per-modality tensors ready for
                         the dual-stream feature extractor.

  MultimodalSleepDataset – Raw EDF path.  Reads EDF files directly via MNE,
                            performs all filtering / resampling on-the-fly, and
                            returns the same (eeg, eog, emg, resp) tuple.  Use
                            this when starting from raw recordings.

Channel conventions used throughout the project
------------------------------------------------
  EEG  : C4:M1 or C4:A1           → 100 Hz   bandpass 0.3–35 Hz
  EOG  : E1:M2 or EOG1:A2         → 100 Hz   (lowpass already applied by MNE)
  EMG  : EMG / Chin 1             → 100 Hz   bandpass 10–70 Hz
  RESP : Sum Effort / RIP Thorax  →  32 Hz   (thoracic effort, lowpass 1 Hz)

Each 30-second epoch therefore has the following sample counts:
  EEG  : 3 000 samples
  EOG  : 3 000 samples
  EMG  : 3 000 samples
  RESP :   960 samples

Label mapping (AASM):
  Wake=0, N1=1, N2=2, N3=3, REM=4
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAPPING: Dict[str, int] = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
EPOCH_SEC: float = 30.0

# Target sampling frequencies per modality
EEG_SFREQ: int = 100   # → 3 000 samples per 30-s epoch
EOG_SFREQ: int = 100   # → 3 000 samples per 30-s epoch
EMG_SFREQ: int = 100   # → 3 000 samples per 30-s epoch
RESP_SFREQ: int = 32   # →   960 samples per 30-s epoch

EEG_SAMPLES: int = int(EPOCH_SEC * EEG_SFREQ)   # 3 000
EOG_SAMPLES: int = int(EPOCH_SEC * EOG_SFREQ)   # 3 000
EMG_SAMPLES: int = int(EPOCH_SEC * EMG_SFREQ)   # 3 000
RESP_SAMPLES: int = int(EPOCH_SEC * RESP_SFREQ)  #   960

# Known channel aliases in the iSLEEPS dataset
EEG_CANDIDATES: List[str] = ["C4:M1", "C4:A1", "C3:M2", "C3:A2"]
EOG_CANDIDATES: List[str] = ["E1:M2", "EOG1:A2"]
EMG_CANDIDATES: List[str] = ["EMG", "EMG2", "Chin 1", "Chin 2"]
RESP_CANDIDATES: List[str] = ["Sum Effort", "Sum RIPs", "RIP Thorax", "Thorax", "Abdomen"]

# Keys produced by the existing iSLEEPS_preprocess_v2 numpy pipeline
# (SN_*.npz files contain a stacked 'x' array with shape [epochs, time, channels]
#  and a 'y' array with shape [epochs])
NPZ_X_KEY: str = "x"
NPZ_Y_KEY: str = "y"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pick_first_available(ch_names: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate present in ch_names, else None."""
    for c in candidates:
        if c in ch_names:
            return c
    return None


def subject_id_from_path(path: str) -> int:
    """Extract numeric subject ID from 'SN_<id>.npz'."""
    name = os.path.basename(path).replace(".npz", "")
    return int(name.split("_")[1])


def compute_norm_stats(
    files: List[str],
    channel_idx_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std over all training epochs.

    Handles mixed-channel datasets by using the MINIMUM channel count
    across all files (avoids crash when some subjects have extra channels).

    Args:
        files: List of paths to SN_*.npz files.
        channel_idx_map: Unused; kept for backward compatibility.

    Returns:
        mean: shape [C_min]
        std:  shape [C_min]
    """
    # First pass: find minimum number of channels across all files
    min_c = None
    for f in files:
        d = np.load(f)
        c = d[NPZ_X_KEY].shape[-1]
        min_c = c if min_c is None else min(min_c, c)

    sum_c, sumsq_c, total = (
        np.zeros(min_c, dtype=np.float64),
        np.zeros(min_c, dtype=np.float64),
        0,
    )
    for f in files:
        d = np.load(f, mmap_mode='r', allow_pickle=False)
        x = np.array(d[NPZ_X_KEY], dtype=np.float32)  # [E, T, C] — explicit copy
        x = x[:, :, :min_c]                            # clip to min_c
        flat = x.reshape(-1, min_c)                    # [E*T, C_min]
        sum_c   += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)
        total   += flat.shape[0]


    mean = sum_c / total
    var  = (sumsq_c / total) - (mean ** 2)
    std  = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)



def _bandpass_numpy(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a 1-D numpy array.

    Args:
        signal : 1-D float array of length T
        low    : lower cutoff frequency (Hz)
        high   : upper cutoff frequency (Hz)
        fs     : sampling frequency (Hz)
        order  : filter order

    Returns:
        Filtered signal of the same shape.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def _resample_numpy(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample a 1-D signal from orig_fs to target_fs using scipy.

    Args:
        signal    : 1-D float array
        orig_fs   : original sampling rate
        target_fs : desired sampling rate

    Returns:
        Resampled 1-D float array.
    """
    from scipy.signal import resample_poly
    from math import gcd

    orig_fs = int(round(orig_fs))
    target_fs = int(round(target_fs))
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(signal, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset 1 – pre-processed NPZ files (fast, default path)
# ---------------------------------------------------------------------------

class SleepEpochDataset(Dataset):
    """
    PyTorch Dataset over pre-processed SN_*.npz files produced by the
    iSLEEPS_preprocess_v2 pipeline.

    The stored 'x' tensor has shape [epochs, time_samples, channels].
    This dataset splits each epoch into per-modality streams and returns:

        eeg   : Tensor[1, EEG_SAMPLES]       (single EEG channel, C4)
        eog   : Tensor[1, EOG_SAMPLES]       (single EOG channel, E1)
        emg   : Tensor[1, EMG_SAMPLES]       (single EMG channel)
        label : LongTensor scalar            (0–4 AASM stage)

    NOTE: Respiratory data is NOT present in the preprocessed NPZ files
    (the preprocessing pipeline only stores EEG+EOG by default).  The
    MultimodalSleepDataset should be used if respiratory features are needed.

    Normalization is applied channel-wise using externally-computed stats.

    Args:
        files      : List of paths to SN_*.npz
        mean       : np.ndarray[C] per-channel mean
        std        : np.ndarray[C] per-channel std
        eeg_ch_idx : Index (within the C dimension) of the EEG channel.
                     Default 0 → C4:M1 or whatever was stored first.
        eog_ch_idx : Index of the EOG channel.  Default 1 → E1:M2.
        emg_ch_idx : Index of the EMG channel if present, else None.
        seq_len    : If > 1, return a sliding window of seq_len consecutive
                     epochs as [seq_len, C, T] and label the center epoch.
    """

    def __init__(
        self,
        files: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        eeg_ch_idx: int = 0,
        eog_ch_idx: int = 1,
        emg_ch_idx: Optional[int] = None,
        seq_len: int = 1,
    ):
        self.files = files
        self.mean = mean            # [C]
        self.std = std              # [C]
        self.eeg_ch_idx = eeg_ch_idx
        self.eog_ch_idx = eog_ch_idx
        self.emg_ch_idx = emg_ch_idx
        self.seq_len = seq_len

        # Build flat index: list of (file_idx, epoch_start_idx)
        self.mapping: List[Tuple[int, int]] = []
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for fi, f in enumerate(self.files):
            d = np.load(f)
            n_epochs = d[NPZ_Y_KEY].shape[0]
            if seq_len <= 1:
                for e in range(n_epochs):
                    self.mapping.append((fi, e))
            else:
                for s in range(n_epochs - seq_len + 1):
                    self.mapping.append((fi, s))

    def __len__(self) -> int:
        return len(self.mapping)

    def _load_cached(self, fi: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load, normalise, and cache all epochs for file index fi."""
        if fi not in self.cache:
            d = np.load(self.files[fi])
            # x: [E, T, C]  →  normalise  →  [E, C, T]
            x = d[NPZ_X_KEY].astype(np.float32)
            x = (x - self.mean[None, None, :]) / self.std[None, None, :]
            x = np.transpose(x, (0, 2, 1))  # [E, C, T]
            y = d[NPZ_Y_KEY].astype(np.int64)
            self.cache[fi] = (x, y)
        return self.cache[fi]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            eeg   : [1, T_eeg]  or  [seq_len, 1, T_eeg]
            eog   : [1, T_eog]  or  [seq_len, 1, T_eog]
            emg   : [1, T_emg]  or  [seq_len, 1, T_emg]  (zeros if absent)
            label : scalar LongTensor  (center epoch for seq_len > 1)
        """
        fi, start = self.mapping[idx]
        x, y = self._load_cached(fi)  # x: [E, C, T]

        if self.seq_len <= 1:
            epoch = x[start]  # [C, T]
            label = y[start]

            eeg = torch.from_numpy(epoch[self.eeg_ch_idx : self.eeg_ch_idx + 1])  # [1, T]
            eog = torch.from_numpy(epoch[self.eog_ch_idx : self.eog_ch_idx + 1])  # [1, T]

            if self.emg_ch_idx is not None:
                emg = torch.from_numpy(epoch[self.emg_ch_idx : self.emg_ch_idx + 1])
            else:
                # Zero placeholder so downstream code is always channel-consistent
                emg = torch.zeros(1, epoch.shape[-1])

        else:
            seq_x = x[start : start + self.seq_len]           # [S, C, T]
            center = start + (self.seq_len // 2)
            label = y[center]

            eeg = torch.from_numpy(seq_x[:, self.eeg_ch_idx : self.eeg_ch_idx + 1])
            eog = torch.from_numpy(seq_x[:, self.eog_ch_idx : self.eog_ch_idx + 1])
            if self.emg_ch_idx is not None:
                emg = torch.from_numpy(seq_x[:, self.emg_ch_idx : self.emg_ch_idx + 1])
            else:
                emg = torch.zeros(self.seq_len, 1, seq_x.shape[-1])

        return eeg, eog, emg, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset 2 – raw EDF files (full multimodal with RESP, on-the-fly DSP)
# ---------------------------------------------------------------------------

class MultimodalSleepDataset(Dataset):
    """
    Full multimodal dataset reading directly from raw EDF + XLSX annotation
    files.  All filtering, resampling, and epoch windowing is done here so
    no pre-processing scripts need to be run first.

    Returns per-epoch tensors:
        eeg   : Tensor[1, 3000]   (100 Hz × 30 s, bandpassed 0.3–35 Hz)
        eog   : Tensor[1, 3000]   (100 Hz × 30 s, lowpass 35 Hz via MNE)
        emg   : Tensor[1, 3000]   (100 Hz × 30 s, bandpassed 10–70 Hz)
        resp  : Tensor[1,  960]   ( 32 Hz × 30 s, raw effort belt)
        label : LongTensor scalar

    Args:
        edf_files      : List[str] – paths to .edf recordings
        ann_files      : List[str] – matching .xlsx annotation files (same order)
        mean_std       : Optional dict with 'mean' and 'std' arrays per modality
                         for z-scoring.  If None, no normalisation is applied.
        seq_len        : Sliding window length (epochs). 1 = single epoch mode.
        cache_in_ram   : Pre-load all subjects into RAM.  Only feasible for
                         small cohorts or machines with ≥ 64 GB RAM.
    """

    def __init__(
        self,
        edf_files: List[str],
        ann_files: List[str],
        mean_std: Optional[Dict[str, np.ndarray]] = None,
        seq_len: int = 1,
        cache_in_ram: bool = False,
    ):
        assert len(edf_files) == len(ann_files), (
            f"Number of EDF files ({len(edf_files)}) must match "
            f"annotation files ({len(ann_files)})"
        )
        self.edf_files = edf_files
        self.ann_files = ann_files
        self.mean_std = mean_std
        self.seq_len = seq_len

        # Build index: list of (subject_idx, epoch_start_idx)
        self.mapping: List[Tuple[int, int]] = []
        self._epoch_counts: List[int] = []
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_in_ram = cache_in_ram

        for si in range(len(edf_files)):
            n = self._count_epochs(edf_files[si], ann_files[si])
            self._epoch_counts.append(n)
            if seq_len <= 1:
                for e in range(n):
                    self.mapping.append((si, e))
            else:
                for s in range(n - seq_len + 1):
                    self.mapping.append((si, s))

            if cache_in_ram:
                self._cache[si] = self._load_subject(si)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _count_epochs(self, edf_path: str, ann_path: str) -> int:
        """Quickly count valid epochs without loading signal data."""
        import pandas as pd
        ann = pd.read_excel(ann_path, sheet_name="Sleep profile", header=None).iloc[8:]
        stages = ann.iloc[:, 1].tolist()
        valid = [s for s in stages if s in ("Wake", "N1", "N2", "N3", "REM")]
        return len(valid)

    def _load_subject(self, si: int) -> Dict[str, np.ndarray]:
        """
        Load one subject's full recording and return a dict with arrays:
            'eeg'   : [E, 3000]
            'eog'   : [E, 3000]
            'emg'   : [E, 3000]
            'resp'  : [E,  960]
            'labels': [E]
        """
        import mne
        import pandas as pd
        from scipy.signal import butter, sosfiltfilt

        mne.set_log_level("WARNING")

        edf_path = self.edf_files[si]
        ann_path = self.ann_files[si]

        # ---- Read annotations (Excel) ------------------------------------
        ann_df = pd.read_excel(ann_path, sheet_name="Sleep profile", header=None).iloc[8:]
        ann_df = ann_df.reset_index(drop=True)
        ann_df.columns = ["timestamp", "stage"]

        stage_to_int = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        labels = []
        for s in ann_df["stage"]:
            if s in stage_to_int:
                labels.append(stage_to_int[s])
        labels = np.array(labels, dtype=np.int64)  # [E]
        n_valid = len(labels)

        # ---- Read raw EDF ------------------------------------------------
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        ch_names = raw.info["ch_names"]
        orig_fs = raw.info["sfreq"]

        def _extract_channel(candidates: List[str], target_fs: int) -> Optional[np.ndarray]:
            """Pick, resample, return raw signal [T_total]."""
            ch = _pick_first_available(ch_names, candidates)
            if ch is None:
                return None
            data = raw.get_data(picks=[ch])[0]  # [T_total]
            if int(orig_fs) != target_fs:
                data = _resample_numpy(data, orig_fs, target_fs)
            return data.astype(np.float32)

        # ---- EEG  (C4:M1 preferred, bandpass 0.3–35 Hz) -----------------
        eeg_raw = _extract_channel(EEG_CANDIDATES, EEG_SFREQ)
        if eeg_raw is not None:
            eeg_raw = _bandpass_numpy(eeg_raw, low=0.3, high=35.0, fs=EEG_SFREQ)

        # ---- EOG  (E1:M2 preferred, lowpass 35 Hz via bandpass 0.3–35) --
        eog_raw = _extract_channel(EOG_CANDIDATES, EOG_SFREQ)
        if eog_raw is not None:
            eog_raw = _bandpass_numpy(eog_raw, low=0.3, high=35.0, fs=EOG_SFREQ)

        # ---- EMG  (bandpass 10–70 Hz) ------------------------------------
        emg_raw = _extract_channel(EMG_CANDIDATES, EMG_SFREQ)
        if emg_raw is not None:
            emg_raw = _bandpass_numpy(emg_raw, low=10.0, high=70.0, fs=EMG_SFREQ)

        # ---- RESP (sum effort belt, 32 Hz, lowpass 1 Hz) -----------------
        resp_raw = _extract_channel(RESP_CANDIDATES, RESP_SFREQ)
        if resp_raw is not None:
            from scipy.signal import butter, sosfiltfilt
            nyq = RESP_SFREQ / 2.0
            sos = butter(4, 1.0 / nyq, btype="low", output="sos")
            resp_raw = sosfiltfilt(sos, resp_raw).astype(np.float32)

        # ---- Epoch slicing -----------------------------------------------
        def _epoch_signal(sig: Optional[np.ndarray], fs: int, n_samples: int) -> np.ndarray:
            """Slice signal into n_valid epochs of n_samples each."""
            total_needed = n_valid * n_samples
            if sig is None:
                return np.zeros((n_valid, n_samples), dtype=np.float32)
            # Pad or truncate to an exact multiple
            if len(sig) < total_needed:
                sig = np.pad(sig, (0, total_needed - len(sig)))
            epochs = sig[:total_needed].reshape(n_valid, n_samples)
            return epochs.astype(np.float32)

        eeg_epochs = _epoch_signal(eeg_raw, EEG_SFREQ, EEG_SAMPLES)   # [E, 3000]
        eog_epochs = _epoch_signal(eog_raw, EOG_SFREQ, EOG_SAMPLES)   # [E, 3000]
        emg_epochs = _epoch_signal(emg_raw, EMG_SFREQ, EMG_SAMPLES)   # [E, 3000]
        resp_epochs = _epoch_signal(resp_raw, RESP_SFREQ, RESP_SAMPLES) # [E,  960]

        return {
            "eeg": eeg_epochs,   # [E, 3000]
            "eog": eog_epochs,   # [E, 3000]
            "emg": emg_epochs,   # [E, 3000]
            "resp": resp_epochs, # [E,  960]
            "labels": labels,    # [E]
        }

    def _get_subject_data(self, si: int) -> Dict[str, np.ndarray]:
        if si in self._cache:
            return self._cache[si]
        data = self._load_subject(si)
        if self._cache_in_ram:
            self._cache[si] = data
        return data

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            eeg   : Tensor[1, 3000]  or [seq_len, 1, 3000]
            eog   : Tensor[1, 3000]  or [seq_len, 1, 3000]
            emg   : Tensor[1, 3000]  or [seq_len, 1, 3000]
            resp  : Tensor[1,  960]  or [seq_len, 1,  960]
            label : scalar LongTensor
        """
        si, start = self.mapping[idx]
        data = self._get_subject_data(si)

        def _z(arr: np.ndarray, key: str) -> np.ndarray:
            """Optional z-score normalisation."""
            if self.mean_std is not None and key in self.mean_std:
                m = self.mean_std[key]["mean"]
                s = self.mean_std[key]["std"]
                return (arr - m) / (s + 1e-8)
            return arr

        if self.seq_len <= 1:
            # Single epoch mode
            eeg = _z(data["eeg"][start], "eeg")[None, :]   # [1, 3000]
            eog = _z(data["eog"][start], "eog")[None, :]   # [1, 3000]
            emg = _z(data["emg"][start], "emg")[None, :]   # [1, 3000]
            resp = _z(data["resp"][start], "resp")[None, :] # [1,  960]
            label = data["labels"][start]
        else:
            sl = slice(start, start + self.seq_len)
            eeg = _z(data["eeg"][sl], "eeg")[:, None, :]   # [S, 1, 3000]
            eog = _z(data["eog"][sl], "eog")[:, None, :]   # [S, 1, 3000]
            emg = _z(data["emg"][sl], "emg")[:, None, :]   # [S, 1, 3000]
            resp = _z(data["resp"][sl], "resp")[:, None, :] # [S, 1,  960]
            label = data["labels"][start + self.seq_len // 2]

        return (
            torch.from_numpy(eeg),
            torch.from_numpy(eog),
            torch.from_numpy(emg),
            torch.from_numpy(resp),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Utilities for cross-validation splitting (used by train.py)
# ---------------------------------------------------------------------------

def resolve_npz_files(data_dir: str) -> List[str]:
    """Return sorted list of SN_*.npz files in data_dir."""
    pattern = os.path.join(data_dir, "SN_*.npz")
    files = sorted(glob.glob(pattern), key=subject_id_from_path)
    if not files:
        raise RuntimeError(
            f"No SN_*.npz files found in: {data_dir}\n"
            "Run the iSLEEPS_preprocess_v2 pipeline first."
        )
    return files


def get_subject_labels(files: List[str]) -> np.ndarray:
    """
    Return the majority sleep stage for each subject file.
    Used for stratified cross-validation splitting at subject level.
    """
    majority_labels = []
    for f in files:
        d = np.load(f, mmap_mode='r', allow_pickle=False)
        y = d[NPZ_Y_KEY].astype(np.int64)
        counts = np.bincount(y, minlength=5)
        majority_labels.append(int(np.argmax(counts)))
    return np.array(majority_labels, dtype=np.int64)
