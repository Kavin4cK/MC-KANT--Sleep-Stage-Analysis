"""Quick timing test: how fast is fold 1 dataset init now?"""
import time, sys
sys.stdout.reconfigure(line_buffering=True)

from mc_kant.dataset import resolve_npz_files, get_subject_labels, compute_norm_stats
from mc_kant.train import MultiChannelDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

DATA_DIR = r"C:\Users\kavin\Music\Documents\DataSets\Sleep Project\processed_data\numpy_subjects"

t0 = time.time()
all_files = resolve_npz_files(DATA_DIR)
print(f"[{time.time()-t0:.1f}s] resolve_npz_files: {len(all_files)} files")

t1 = time.time()
strat_labels = get_subject_labels(all_files)
print(f"[{time.time()-t0:.1f}s] get_subject_labels done ({time.time()-t1:.1f}s)")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_val_idx, test_idx = next(skf.split(np.arange(len(all_files)), strat_labels))

inner_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
strat_tv = strat_labels[train_val_idx]
train_local_idx, val_local_idx = next(inner_skf.split(np.arange(len(train_val_idx)), strat_tv))

train_val_files = [all_files[i] for i in train_val_idx]
test_files = [all_files[i] for i in test_idx]
train_files = [train_val_files[i] for i in train_local_idx]
val_files   = [train_val_files[i] for i in val_local_idx]
print(f"  train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

t2 = time.time()
mean, std = compute_norm_stats(train_files)
print(f"[{time.time()-t0:.1f}s] compute_norm_stats done ({time.time()-t2:.1f}s) channels={len(mean)}")

t3 = time.time()
epoch_counts = {}
ds1 = MultiChannelDataset(train_files, mean, std, seq_len=1,  epoch_counts=epoch_counts)
print(f"[{time.time()-t0:.1f}s] pretrain_ds built ({time.time()-t3:.2f}s): {len(ds1):,} samples")

t4 = time.time()
ds2 = MultiChannelDataset(train_files, mean, std, seq_len=20, epoch_counts=epoch_counts)
print(f"[{time.time()-t0:.1f}s] seq_train_ds built ({time.time()-t4:.2f}s): {len(ds2):,} samples")

t5 = time.time()
ds3 = MultiChannelDataset(val_files, mean, std, seq_len=20, epoch_counts=epoch_counts)
print(f"[{time.time()-t0:.1f}s] seq_val_ds built ({time.time()-t5:.2f}s): {len(ds3):,} samples")

t6 = time.time()
ds4 = MultiChannelDataset(test_files, mean, std, seq_len=20, epoch_counts=epoch_counts)
print(f"[{time.time()-t0:.1f}s] seq_test_ds built ({time.time()-t6:.2f}s): {len(ds4):,} samples")

print(f"\nTotal fold-1 setup time: {time.time()-t0:.1f}s")
print("READY TO TRAIN!")
