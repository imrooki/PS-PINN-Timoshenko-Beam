
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .bc import BoundaryConditionPenalty

try:
    from .data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from .numerics import safe_divide, mean_integral, as_shape, quad_nodes_weights
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from modules.numerics import safe_divide, mean_integral, as_shape, quad_nodes_weights

def compute_w0_manual_derivative(x: torch.Tensor, params: PhysicalParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    A0, a, b, c = params.A0, params.a, params.b, params.c

    if abs(A0) < 1e-12:
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    xi = x - c
    sech_arg = a * xi

    bpi = b * math.pi
    phase_angle = bpi * xi

    sech_val = 1.0 / torch.cosh(sech_arg)
    cos_val = torch.cos(phase_angle)
    sin_val = torch.sin(phase_angle)
    tanh_val = torch.tanh(sech_arg)

    w0 = A0 * sech_val * cos_val

    dw0_dx = -A0 * sech_val * (
        a * tanh_val * cos_val + bpi * sin_val
    )

    tanh_squared = tanh_val ** 2
    d2w0_dx2 = A0 * sech_val * (
        (2 * a**2 * tanh_squared - a**2 - bpi**2) * cos_val
        + 2 * a * bpi * tanh_val * sin_val
    )

    return w0, dw0_dx, d2w0_dx2

class EnergyLoss:

    def __init__(
        self,
        coeffs: MaterialCoeffs,
        params: PhysicalParams,
        bc: BoundaryConditions,
        device: Optional[torch.device] = None,
        *,
        is_nonlinear: bool = False,
    ) -> None:
        self.coeffs = coeffs
        self.params = params
        self.bc = bc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_nonlinear = bool(is_nonlinear)
        if self.params.lambda_val <= 0:
            raise ValueError(f"λ值必须大于0，当前：{self.params.lambda_val}")

    def compute_elastic_foundation_energy_density(self, x: torch.Tensor, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        k1, k2 = self.params.k1, self.params.k2
        w, wx = fields["w"], fields["wx"]

        foundation_density = k1 * (w ** 2) + k2 * (wx ** 2)

        return foundation_density

    def compute_strain_energy_density(self, x: torch.Tensor, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        import warnings

        for field_name, field_tensor in fields.items():
            if torch.isnan(field_tensor).any():
                raise ValueError(f"位移场 '{field_name}' 包含NaN值")
            if torch.isinf(field_tensor).any():
                warnings.warn(f"位移场 '{field_name}' 包含无穷大值，继续计算但需谨慎")

        w0, dw0_dx, _ = compute_w0_manual_derivative(x, self.params)

        a11 = as_shape(self.coeffs.a11, x)
        b11 = as_shape(self.coeffs.b11, x)
        d11 = as_shape(self.coeffs.d11, x)
        a55 = as_shape(self.coeffs.a55, x)

        ux, wx, phix = fields["ux"], fields["wx"], fields["phix"]
        phi = fields["phi"]
        lambda_val = self.params.lambda_val

        term1 = (a11 * ux
                + b11 * phix
                + safe_divide(a11 * wx * dw0_dx, lambda_val)
                - lambda_val * self.params.n_xT)

        term2 = (ux
                + safe_divide(wx * dw0_dx, lambda_val)
                - lambda_val * self.params.alpha_t * self.params.DeltaT)

        term3 = (b11 * ux
                + d11 * phix
                + safe_divide(b11 * wx * dw0_dx, lambda_val)
                - lambda_val * self.params.m_xT)

        shear_term = a55 * (wx + lambda_val * phi) ** 2

        if self.is_nonlinear:
            term1 = term1 + safe_divide(a11 * (wx ** 2), 2.0 * lambda_val)

            term2 = term2 + safe_divide(wx ** 2, 2.0 * lambda_val)

            term3 = term3 + safe_divide(b11 * (wx ** 2), 2.0 * lambda_val)

        strain_energy_density = term1 * term2 + term3 * phix + shear_term

        if torch.isnan(strain_energy_density).any():
            raise RuntimeError("应变能密度计算产生了NaN值")
        if torch.isinf(strain_energy_density).any():
            warnings.warn("应变能密度包含无穷大值 - 可能存在数值不稳定")
        if (strain_energy_density < 0).any():
            warnings.warn("检测到负应变能 - 请检查物理参数和边界条件")

        return strain_energy_density

    def compute_total_energy(self, x: torch.Tensor, fields: Dict[str, torch.Tensor], weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        strain_density = self.compute_strain_energy_density(x, fields)

        foundation_density = self.compute_elastic_foundation_energy_density(x, fields)

        Pi_str = 0.5 * mean_integral(strain_density, weights=weights)
        Pi_w = 0.5 * mean_integral(foundation_density, weights=weights)

        Pi_e = mean_integral(self.params.q * fields["w"], weights=weights)

        Pi_all = Pi_str + Pi_w - Pi_e

        return {
            "Pi_str": Pi_str,
            "Pi_w": Pi_w,
            "Pi_str_T": torch.tensor(0.0, device=x.device),
            "Pi_e": Pi_e,
            "Pi_all": Pi_all,
            "strain_density": strain_density,
            "foundation_density": foundation_density,
        }

class WeightedEnergyLoss:

    def __init__(
        self,
        energy_loss: "EnergyLoss",
        bc_penalty: BoundaryConditionPenalty,
        bc_weight: float = 1000.0,
        adaptive_weights: bool = False,
    ) -> None:
        self.energy_loss = energy_loss
        self.bc_penalty = bc_penalty
        self.bc_weight = float(bc_weight)
        self.adaptive_weights = bool(adaptive_weights)
        if adaptive_weights:
            from collections import deque
            self._history = {
                "energy": deque(maxlen=100),
                "bc": deque(maxlen=100)
            }
        else:
            self._history = None
        self.integrator: str = 'mc'

        self.agq_rule: str = 'G10K21'
        self.agq_abs_tol: float = 1e-6
        self.agq_rel_tol: float = 1e-4
        self.agq_max_points: int = 4096
        self.agq_max_depth: int = 100
        self.agq_refine_every: int = 0
        self.agq_fail_policy: str = 'use_partial'

        self._agq_nodes: Optional[torch.Tensor] = None
        self._agq_weights: Optional[torch.Tensor] = None
        self._agq_build_count: int = 0
        self._agq_last_nodes: int = 0
        self._agq_last_intervals: int = 0
        self._agq_last_hit_limit: bool = False
        self._agq_info_printed: bool = False

    def _adaptive_w(self, energy: torch.Tensor, bc_loss: torch.Tensor) -> float:
        bc_loss_val = abs(bc_loss.item())
        energy_val = abs(energy.item())
        
        if bc_loss_val > 1e-10 and energy_val > 1e-15:
            ratio = energy_val / bc_loss_val
            ratio = max(min(ratio, self.bc_weight * 10.0), self.bc_weight * 0.1)
            return float(ratio)
        return float(self.bc_weight)

    def compute_total_loss(self, x: torch.Tensor, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        integ_norm = (
            (self.integrator or 'mc')
            .lower()
            .replace('_', '')
            .replace('-', '')
            .replace('–', '')
            .replace('—', '')
            .replace(' ', '')
        )
        quad_x: Optional[torch.Tensor] = None
        weights: Optional[torch.Tensor] = None
        if integ_norm == 'agq':
            need_build = self._agq_nodes is None or self._agq_weights is None
            if not need_build and self.agq_refine_every and self.agq_refine_every > 0:
                self._agq_build_count += 1
                if self._agq_build_count % int(self.agq_refine_every) == 0:
                    need_build = True
            if need_build:
                try:
                    quad_x, weights = self._build_agq_nodes_and_weights(field_eval, device=x.device, dtype=x.dtype)
                    self._agq_nodes, self._agq_weights = quad_x, weights
                    if not self._agq_info_printed:
                        try:
                            print(
                                f"AGQ nodes built: n_nodes={self._agq_last_nodes}, "
                                f"n_intervals={self._agq_last_intervals}, hit_limit={self._agq_last_hit_limit}"
                            )
                        except Exception:
                            pass
                        self._agq_info_printed = True
                except Exception as e:
                    if (self.agq_fail_policy or 'use_partial').lower() == 'fallback_gauss':
                        quad_x, weights = quad_nodes_weights('gauss', int(x.shape[0]), device=x.device, dtype=x.dtype)
                    else:
                        raise e
            else:
                quad_x, weights = self._agq_nodes, self._agq_weights
        else:
            n = int(x.shape[0])
            if integ_norm in ('gauss', 'gausslegendre', 'legendre'):
                quad_x, weights = quad_nodes_weights('gauss', n, device=x.device, dtype=x.dtype)
            elif integ_norm in ('clenshaw', 'clenshawcurtis', 'cc'):
                quad_x, weights = quad_nodes_weights('clenshaw', n, device=x.device, dtype=x.dtype)

        x_use = quad_x if quad_x is not None else x
        fields = field_eval(x_use)
        energy = self.energy_loss.compute_total_energy(x_use, fields, weights=weights)
        bc_loss = self.bc_penalty.compute(field_eval)
        boundary_weight = self._adaptive_w(energy["Pi_all"], bc_loss) if self.adaptive_weights else self.bc_weight
        total = energy["Pi_all"] + boundary_weight * bc_loss
        if self._history is not None:
            self._history["energy"].append(float(energy["Pi_all"].item()))
            self._history["bc"].append(float(bc_loss.item()))
        return {"total": total, "bc": bc_loss, "bc_weight": boundary_weight, **energy}

    def _build_agq_nodes_and_weights(
        self,
        field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as _np

        def leggauss(n: int):
            xi, w = _np.polynomial.legendre.leggauss(int(n))
            return xi.astype(_np.float64), w.astype(_np.float64)

        def map_to(a: float, b: float, xi: _np.ndarray, w: _np.ndarray):
            c = 0.5 * (b - a)
            m = 0.5 * (a + b)
            x = m + c * xi
            ww = c * w
            return x, ww

        import re
        rule = getattr(self, 'agq_rule', 'G10K21')
        match = re.match(r'G(\d+)K(\d+)', rule, re.IGNORECASE)
        if match:
            n_lo, n_hi = int(match.group(1)), int(match.group(2))
        else:
            n_lo, n_hi = 10, 21
            import warnings
            warnings.warn(f"无法解析AGQ规则 '{rule}'，使用默认值 G10K21")

        xi_hi, w_hi = leggauss(n_hi)
        xi_lo, w_lo = leggauss(n_lo)

        abs_tol = float(self.agq_abs_tol)
        rel_tol = float(self.agq_rel_tol)
        max_pts = max(1, int(self.agq_max_points))
        max_depth = max(0, int(self.agq_max_depth))

        stack: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]
        X_list: list[_np.ndarray] = []
        W_list: list[_np.ndarray] = []
        total_pts = 0
        hit_limit = False
        while stack:
            a, b, depth = stack.pop()
            xh_np, wh_np = map_to(a, b, xi_hi, w_hi)
            xl_np, wl_np = map_to(a, b, xi_lo, w_lo)

            xh = torch.tensor(xh_np, device=device, dtype=dtype).reshape(-1, 1)
            xh.requires_grad_(True)
            fields_h = field_eval(xh)
            strain_h = self.energy_loss.compute_strain_energy_density(xh, fields_h)
            foundation_h = self.energy_loss.compute_elastic_foundation_energy_density(xh, fields_h)
            integrand_h = 0.5 * strain_h + 0.5 * foundation_h - self.energy_loss.params.q * fields_h["w"]
            q_hi = float((torch.tensor(wh_np, device=device, dtype=dtype).view(-1) * integrand_h.view(-1)).sum().detach().cpu().item())

            xl = torch.tensor(xl_np, device=device, dtype=dtype).reshape(-1, 1)
            xl.requires_grad_(True)
            fields_l = field_eval(xl)
            strain_l = self.energy_loss.compute_strain_energy_density(xl, fields_l)
            foundation_l = self.energy_loss.compute_elastic_foundation_energy_density(xl, fields_l)
            integrand_l = 0.5 * strain_l + 0.5 * foundation_l - self.energy_loss.params.q * fields_l["w"]
            q_lo = float((torch.tensor(wl_np, device=device, dtype=dtype).view(-1) * integrand_l.view(-1)).sum().detach().cpu().item())

            err = abs(q_hi - q_lo)
            tol = max(abs_tol, rel_tol * abs(q_hi))

            force_accept = False
            if depth >= max_depth:
                force_accept = True
            if total_pts + n_hi > max_pts:
                force_accept = True

            if (err <= tol) or force_accept:
                X_list.append(xh_np)
                W_list.append(wh_np)
                total_pts += n_hi
                if total_pts >= max_pts:
                    hit_limit = True
                    while stack:
                        a_rem, b_rem, _ = stack.pop()
                        x_rem, w_rem = map_to(a_rem, b_rem, xi_lo, w_lo)
                        X_list.append(x_rem)
                        W_list.append(w_rem)
                    break
            else:
                m = 0.5 * (a + b)
                if depth + 1 <= max_depth or total_pts + 2 * n_hi <= max_pts:
                    stack.append((m, b, depth + 1))
                    stack.append((a, m, depth + 1))
                else:
                    X_list.append(xh_np)
                    W_list.append(wh_np)
                    total_pts += n_hi
                    if total_pts >= max_pts:
                        hit_limit = True
                        while stack:
                            a_rem, b_rem, _ = stack.pop()
                            x_rem, w_rem = map_to(a_rem, b_rem, xi_lo, w_lo)
                            X_list.append(x_rem)
                            W_list.append(w_rem)
                        break

        if not X_list:
            xi, w = xi_hi, w_hi
            x_np, w_np = map_to(0.0, 1.0, xi, w)
            X_list.append(x_np)
            W_list.append(w_np)

        X_all = _np.concatenate(X_list, axis=0)
        W_all = _np.concatenate(W_list, axis=0)
        x_all = torch.tensor(X_all, device=device, dtype=dtype).reshape(-1, 1)
        w_all = torch.tensor(W_all, device=device, dtype=dtype).reshape(-1)
        self._agq_build_count = 0
        self._agq_last_nodes = int(w_all.numel())
        self._agq_last_intervals = int(len(X_list))
        self._agq_last_hit_limit = bool(hit_limit)

        weights_sum = float(w_all.sum().item())
        domain_min, domain_max = float(X_all.min()), float(X_all.max())
        if abs(weights_sum - 1.0) > 0.01 or domain_min > 0.001 or domain_max < 0.999:
            import warnings
            warnings.warn(
                f"[AGQ] Integration domain may be incomplete: weights_sum={weights_sum:.4f} (expected~1.0), "
                f"coverage=[{domain_min:.4f}, {domain_max:.4f}] (expected [0,1]), "
                f"hit_limit={hit_limit}, n_nodes={w_all.numel()}, n_intervals={len(X_list)}"
            )

        return x_all.requires_grad_(True), w_all

def create_loss_function(
    problem_type: str,
    coeffs: MaterialCoeffs,
    params: PhysicalParams,
    bc: BoundaryConditions,
    bc_weight: float = 1000.0,
    adaptive_weights: bool = False,
    device: Optional[torch.device] = None,
) -> WeightedEnergyLoss:
    
    def _get_boundary_penalty_class():
        try:
            from .bc import BoundaryConditionPenalty
            return BoundaryConditionPenalty
        except ImportError:
            from modules.bc import BoundaryConditionPenalty
            return BoundaryConditionPenalty

    p = problem_type.lower()
    if p not in ("linear", "nonlinear"):
        raise ValueError(f"不支持的问题类型: {problem_type}")
    
    energy = EnergyLoss(coeffs, params, bc, device, is_nonlinear=(p == "nonlinear"))

    BoundaryConditionPenalty = _get_boundary_penalty_class()
    bc_pen = BoundaryConditionPenalty(bc, coeffs, params, device, is_nonlinear=(p == "nonlinear"))
    
    return WeightedEnergyLoss(energy, bc_pen, bc_weight, adaptive_weights)

__all__ = [
    "EnergyLoss",
    "WeightedEnergyLoss",
    "create_loss_function",
]
