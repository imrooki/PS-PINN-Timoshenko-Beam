
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict

import torch

class BoundaryConditionType(str, Enum):

    CLAMPED_CLAMPED = "C-C"
    SIMPLE_SIMPLE = "S-S"
    HINGED_HINGED = "H-H"
    CLAMPED_SIMPLE = "C-S"
    CLAMPED_HINGED = "C-H"
    CLAMPED_FREE = "C-F"

class DistributionType(str, Enum):

    X_TYPE = "X"
    O_TYPE = "O"
    U_TYPE = "U"

@dataclass
class MaterialCoeffs:

    a11: Callable[[torch.Tensor], torch.Tensor]
    b11: Callable[[torch.Tensor], torch.Tensor]
    d11: Callable[[torch.Tensor], torch.Tensor]
    a55: Callable[[torch.Tensor], torch.Tensor]

@dataclass
class PhysicalParams:

    lambda_val: float = 1.0
    q: float = 0.0
    n_xT: float = 0.0
    m_xT: float = 0.0
    alpha_t: float = 0.0
    DeltaT: float = 0.0

    k1: float = 0.0
    k2: float = 0.0

    A0: float = 0.0
    a: float = 0.0
    b: float = 0.0
    c: float = 0.5

@dataclass
class BoundaryConditions:

    type: str = "C-C"
    u_left: Optional[float] = 0.0
    u_right: Optional[float] = None
    w_left: Optional[float] = 0.0
    w_right: Optional[float] = 0.0
    phi_left: Optional[float] = 0.0
    phi_right: Optional[float] = 0.0

__all__ = [
    "MaterialCoeffs",
    "PhysicalParams",
    "BoundaryConditions",
    "BoundaryConditionType",
    "DistributionType",
]
