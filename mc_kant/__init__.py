"""
MC-KANT: Multimodal Contrastive KAN-Transformer
================================================
A novel deep learning architecture for sleep stage classification
on the iSLEEPS polysomnography dataset (ischemic stroke patients).

Package structure:
    mc_kant/
    ├── __init__.py          - Package entry point
    ├── dataset.py           - Phase 1: Multimodal data ingestion & preprocessing
    ├── feature_extractor.py - Phase 2: Dual-stream temporal + spectral branches
    ├── unet_encoder.py      - Phase 3: U-Net contextual sequence encoder
    ├── contrastive.py       - Phase 4: Supervised contrastive representation learning
    ├── kan_head.py          - Phase 5: KAN (B-spline) classification head
    ├── model.py             - Full MC-KANT model assembly
    └── train.py             - Phase 6: Training loop with cross-validation
"""

from .model import MCKANT
from .dataset import SleepEpochDataset, MultimodalSleepDataset
from .train import train_kfold

__all__ = ["MCKANT", "SleepEpochDataset", "MultimodalSleepDataset", "train_kfold"]
