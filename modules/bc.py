
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

try:
    from .data_types import BoundaryConditions, BoundaryConditionType, MaterialCoeffs, PhysicalParams
    from .numerics import safe_divide, as_shape
    from .physics import compute_w0_manual_derivative
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.data_types import BoundaryConditions, BoundaryConditionType, MaterialCoeffs, PhysicalParams
    from modules.numerics import safe_divide, as_shape
    from modules.physics import compute_w0_manual_derivative

def lifting(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    xi = x_norm

    def lift_one(vhat: torch.Tensor, vL: Optional[float], vR: Optional[float]) -> torch.Tensor:
        if vL is not None and vR is not None:
            return vL + xi * (vR - vL) + xi * (1.0 - xi) * vhat
        elif vL is not None and vR is None:
            return vL + xi * vhat
        elif vL is None and vR is not None:
            return vR + (1.0 - xi) * vhat
        else:
            return vhat

    u = lift_one(raw_u, bc.u_left, bc.u_right)
    w = lift_one(raw_w, bc.w_left, bc.w_right)
    phi = lift_one(raw_phi, bc.phi_left, bc.phi_right)
    return u, w, phi

def lifting_trig(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math
    pi = math.pi
    xi = x_norm

    def lift_one_trig(vhat: torch.Tensor, vL: Optional[float], vR: Optional[float]) -> torch.Tensor:
        if vL is not None and vR is not None:
            sin_term = torch.sin(pi * xi)
            return vL + xi * (vR - vL) + sin_term * vhat
        elif vL is not None and vR is None:
            sin_half = torch.sin(pi * xi / 2.0)
            return vL + sin_half * vhat
        elif vL is None and vR is not None:
            cos_half = torch.cos(pi * xi / 2.0)
            return vR + cos_half * vhat
        else:
            return vhat

    u = lift_one_trig(raw_u, bc.u_left, bc.u_right)
    w = lift_one_trig(raw_w, bc.w_left, bc.w_right)
    phi = lift_one_trig(raw_phi, bc.phi_left, bc.phi_right)
    return u, w, phi

def lifting_none(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return raw_u, raw_w, raw_phi

def get_lifting_function(lifting_basis: str = "poly"):
    basis_norm = str(lifting_basis).lower().strip()

    if basis_norm in ("poly", "polynomial"):
        return lifting
    elif basis_norm in ("trig", "sin", "sincos", "galerkin"):
        return lifting_trig
    elif basis_norm in ("none", "identity", "soft", "raw"):
        return lifting_none
    else:
        raise ValueError(
            f"Unsupported lifting_basis: {lifting_basis}. "
            "Available options: ['poly', 'trig', 'none']"
        )

class BoundaryConditionPenalty:

    def __init__(self, bc: BoundaryConditions, coeffs: MaterialCoeffs, params: PhysicalParams, device: Optional[torch.device] = None, *, is_nonlinear: bool = False) -> None:
        self.bc = bc
        self.coeffs = coeffs
        self.params = params
        self.lambda_val = float(params.lambda_val)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_nonlinear = bool(is_nonlinear)

    def _evaluate_at_boundary(self, x_val: float, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        x = torch.tensor([[x_val]], device=self.device, dtype=torch.float32)
        x.requires_grad_(True)
        fields = field_eval(x)

        w0, dw0_dx, _ = compute_w0_manual_derivative(x, self.params)

        b11 = as_shape(self.coeffs.b11, x)
        d11 = as_shape(self.coeffs.d11, x)

        moment = b11 * fields["ux"] + d11 * fields["phix"]

        moment = moment + safe_divide(b11 * fields["wx"] * dw0_dx, self.lambda_val)

        if self.is_nonlinear:
            moment = moment + safe_divide(b11 * (fields["wx"] ** 2), 2.0 * self.lambda_val)

        moment = moment - self.params.m_xT

        fields["M"] = moment
        return fields

    def compute(self, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> torch.Tensor:

        left = self._evaluate_at_boundary(0.0, field_eval)
        right = self._evaluate_at_boundary(1.0, field_eval)

        loss = torch.tensor(0.0, device=self.device)
        bc_type = self.bc.type

        if self.bc.u_left is not None:
            loss += ((left["u"] - self.bc.u_left) ** 2).mean()
        if self.bc.u_right is not None:
            loss += ((right["u"] - self.bc.u_right) ** 2).mean()
        if self.bc.w_left is not None:
            loss += ((left["w"] - self.bc.w_left) ** 2).mean()
        if self.bc.w_right is not None:
            loss += ((right["w"] - self.bc.w_right) ** 2).mean()

        if bc_type == BoundaryConditionType.CLAMPED_CLAMPED.value:
            loss += (left["phi"] ** 2).mean() + (right["phi"] ** 2).mean()
        elif bc_type in (BoundaryConditionType.SIMPLE_SIMPLE.value, BoundaryConditionType.HINGED_HINGED.value):
            loss += (left["M"] ** 2).mean() + (right["M"] ** 2).mean()
        elif bc_type in (BoundaryConditionType.CLAMPED_SIMPLE.value, BoundaryConditionType.CLAMPED_HINGED.value):
            loss += (left["phi"] ** 2).mean() + (right["M"] ** 2).mean()
        elif bc_type == BoundaryConditionType.CLAMPED_FREE.value:
            loss += (left["phi"] ** 2).mean() + (right["M"] ** 2).mean()
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

        return loss

__all__ = ["lifting", "lifting_trig", "lifting_none", "get_lifting_function", "BoundaryConditionPenalty"]

def make_bc_spec(bc_type: str) -> BoundaryConditions:
    if bc_type == BoundaryConditionType.CLAMPED_CLAMPED.value:
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=0.0,
            phi_right=0.0,
        )
    elif bc_type in (BoundaryConditionType.SIMPLE_SIMPLE.value, BoundaryConditionType.HINGED_HINGED.value):
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=None,
            phi_right=None,
        )
    elif bc_type in (BoundaryConditionType.CLAMPED_SIMPLE.value, BoundaryConditionType.CLAMPED_HINGED.value):
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=0.0,
            phi_right=None,
        )
    elif bc_type == BoundaryConditionType.CLAMPED_FREE.value:
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=None,
            w_left=0.0,
            w_right=None,
            phi_left=0.0,
            phi_right=None,
        )
    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

__all__.extend(["make_bc_spec"])
