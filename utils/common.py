
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def print_config(params_dict: dict):
    print("=" * 60)
    print("Configuration:")
    print("-" * 60)
    for key, value in params_dict.items():
        print(f"  {key}: {value}")
    print("=" * 60)

def parse_activation_from_path(path_or_name: str) -> Dict[str, any]:
    import re
    from pathlib import Path

    result = {
        'activation_type': 'Tanh',
        'siren_omega_0': 30.0,
        'siren_omega_hidden': 30.0,
    }

    path_str = str(path_or_name)

    siren_pattern = r'[_-]SIREN_w([\d.]+)_([\d.]+)'
    siren_match = re.search(siren_pattern, path_str)
    if siren_match:
        result['activation_type'] = 'SIREN'
        result['siren_omega_0'] = float(siren_match.group(1))
        result['siren_omega_hidden'] = float(siren_match.group(2))
        return result

    if re.search(r'[_-]SIREN(?![_\w])', path_str):
        result['activation_type'] = 'SIREN'
        return result

    if re.search(r'[_-]Sin(?![_\w])', path_str):
        result['activation_type'] = 'Sin'
        return result

    if re.search(r'[_-]Tanh(?![_\w])', path_str):
        result['activation_type'] = 'Tanh'
        return result

    return result

def max_deflection(w: np.ndarray) -> float:
    return float(np.abs(w).max())

def safe_mkdir(path: str | Path) -> None:
    import os
    from pathlib import Path

    path_str = str(Path(path).resolve())
    if os.name == 'nt' and len(path_str) > 200:
        if not path_str.startswith("\\\\?\\"):
            path_str = "\\\\?\\" + path_str
    
    os.makedirs(path_str, exist_ok=True)

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float('nan')
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    
    return float(1.0 - ss_res / ss_tot)
