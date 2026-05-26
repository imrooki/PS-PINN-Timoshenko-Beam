
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import time

try:
    from .nets import build_timoshenko_net
    from .data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from .bc import lifting, lifting_trig, get_lifting_function
    from .numerics import d_dx, sample_1d
    from .physics import EnergyLoss, WeightedEnergyLoss
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.nets import build_timoshenko_net
    from modules.data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from modules.bc import lifting, lifting_trig, get_lifting_function
    from modules.numerics import d_dx, sample_1d
    from modules.physics import EnergyLoss, WeightedEnergyLoss

class ConstantField:

    def __init__(self, value: float, name: str = "") -> None:
        self.value = float(value)
        self.__name__ = f"const_{name}" if name else "const_field"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)

def as_fun(val_or_fun, name: str = "") -> Callable[[torch.Tensor], torch.Tensor]:

    if callable(val_or_fun):
        return val_or_fun
    return ConstantField(val_or_fun, name)

class EnergyPINNStatic(nn.Module):

    def __init__(
        self,
        coeffs: MaterialCoeffs,
        params: PhysicalParams,
        bc: BoundaryConditions,
        *,
        device: Optional[torch.device] = None,
        bc_type: str = "C-C",
        bc_weight: float = 10.0,
        is_nonlinear: bool = False,
        encoder_dims_shared: Optional[list] = None,
        head_dims: Optional[list] = None,
        in_dim: int = 1,
        sampler: str = "uniform",
        sampler_reuse: bool = False,
        integrator: str = "mc",
        agq_rule: str = "G10K21",
        agq_abs_tol: float = 1e-6,
        agq_rel_tol: float = 1e-4,
        agq_max_points: int = 4096,
        agq_max_depth: int = 100,
        agq_refine_every: int = 0,
        agq_fail_policy: str = "use_partial",
        activation_type: str = "Tanh",
        siren_omega_0: float = 30.0,
        siren_omega_hidden: float = 30.0,
        lifting_basis: str = "poly",
    ) -> None:
        super().__init__()
        self.coeffs = coeffs
        self.params = params
        self.bc = BoundaryConditions(
            type=bc_type,
            u_left=bc.u_left,
            u_right=bc.u_right,
            w_left=bc.w_left,
            w_right=bc.w_right,
            phi_left=bc.phi_left,
            phi_right=bc.phi_right,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if in_dim != 1:
            raise ValueError(
                f"当前静力 EnergyPINNStatic 仅支持 x-only 输入 (in_dim=1)，收到 in_dim={in_dim}. "
                "若需 (x,t) 动力学，请使用相应的动态求解入口。"
            )
        self.net = build_timoshenko_net(
            in_dim=in_dim,
            encoder_dims_shared=encoder_dims_shared,
            head_dims=head_dims,
            activation_type=activation_type,
            siren_omega_0=siren_omega_0,
            siren_omega_hidden=siren_omega_hidden,
        ).to(self.device)

        self.lifting_basis = str(lifting_basis)
        self._lifting_fn = get_lifting_function(self.lifting_basis)

        self.bc_type = bc_type
        self.bc_weight = float(bc_weight)
        self.is_nonlinear = bool(is_nonlinear)
        self.sampler_type = str(sampler)
        self.sampler_reuse = bool(sampler_reuse)
        self._cached_samples: Optional[torch.Tensor] = None
        self._cached_N: Optional[int] = None
        self.integrator_type = str(integrator)
        self.agq_rule = str(agq_rule)
        self.agq_abs_tol = float(agq_abs_tol)
        self.agq_rel_tol = float(agq_rel_tol)
        self.agq_max_points = int(agq_max_points)
        self.agq_max_depth = int(agq_max_depth)
        self.agq_refine_every = int(agq_refine_every)
        self.agq_fail_policy = str(agq_fail_policy)

        if self.params.lambda_val <= 0:
            raise ValueError(f"lambda_val必须大于0，当前值：{self.params.lambda_val}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA specified but GPU not available, switching to CPU")
            self.device = torch.device("cpu")

        energy_loss = EnergyLoss(self.coeffs, self.params, self.bc, self.device, is_nonlinear=self.is_nonlinear)
        
        def _get_boundary_penalty_class():
            try:
                from .bc import BoundaryConditionPenalty
                return BoundaryConditionPenalty
            except ImportError:
                from modules.bc import BoundaryConditionPenalty
                return BoundaryConditionPenalty

        BoundaryConditionPenalty = _get_boundary_penalty_class()
        bc_penalty = BoundaryConditionPenalty(self.bc, self.coeffs, self.params, self.device, is_nonlinear=self.is_nonlinear)
        self.weighted_loss_func = WeightedEnergyLoss(energy_loss, bc_penalty, self.bc_weight)
        self.weighted_loss_func.integrator = (self.integrator_type or 'mc')
        self.weighted_loss_func.agq_rule = self.agq_rule
        self.weighted_loss_func.agq_abs_tol = self.agq_abs_tol
        self.weighted_loss_func.agq_rel_tol = self.agq_rel_tol
        self.weighted_loss_func.agq_max_points = self.agq_max_points
        self.weighted_loss_func.agq_max_depth = self.agq_max_depth
        self.weighted_loss_func.agq_refine_every = self.agq_refine_every
        self.weighted_loss_func.agq_fail_policy = self.agq_fail_policy

    def fields_and_grads(self, x_norm: torch.Tensor) -> Dict[str, torch.Tensor]:

        raw = self.net(x_norm)
        if raw.size(-1) < 3:
            raise ValueError(f"网络输出维度不足，期望至少3个输出，实际得到{raw.size(-1)}个")
        raw_u, raw_w, raw_phi = raw[:, [0]], raw[:, [1]], raw[:, [2]]
        u, w, phi = self._lifting_fn(x_norm, raw_u, raw_w, raw_phi, self.bc)

        ux = torch.autograd.grad(
            u, x_norm,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        wx = torch.autograd.grad(
            w, x_norm,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        phix = torch.autograd.grad(
            phi, x_norm,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return {"u": u, "w": w, "phi": phi, "ux": ux, "wx": wx, "phix": phix}

    def _train_samples(self, N_samples: int) -> torch.Tensor:
        if self.sampler_reuse and self._cached_samples is not None and self._cached_N == int(N_samples):
            return self._cached_samples
        x = sample_1d(N_samples, self.device, sampler=self.sampler_type, dtype=torch.float32)
        if self.sampler_reuse:
            self._cached_samples = x
            self._cached_N = int(N_samples)
        return x

    def energies(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        fields = self.fields_and_grads(x_norm)
        energy = self.weighted_loss_func.energy_loss.compute_total_energy(x_norm, fields)
        return energy["Pi_str"], energy["Pi_str_T"], energy["Pi_e"], energy["Pi_all"]

    def bc_loss(self) -> torch.Tensor:

        return self.weighted_loss_func.bc_penalty.compute(self.fields_and_grads)

    def loss(self, N_samples: int = 4096) -> Dict[str, torch.Tensor]:
        x_norm = self._train_samples(N_samples)
        loss_comp = self.weighted_loss_func.compute_total_loss(x_norm, self.fields_and_grads)
        return {
            "Pi_str": loss_comp["Pi_str"],
            "Pi_T": loss_comp["Pi_str_T"],
            "Pi_e": loss_comp["Pi_e"],
            "Pi_all": loss_comp["Pi_all"],
            "bc": loss_comp["bc"],
            "total": loss_comp["total"],
        }

class LBFGSClosure:

    def __init__(self, model: "EnergyPINNStatic", optimizer: torch.optim.Optimizer, x_samples: torch.Tensor) -> None:
        self.model = model
        self.optimizer = optimizer
        self.x_samples = x_samples

    def __call__(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss_comp = self.model.weighted_loss_func.compute_total_loss(self.x_samples, self.model.fields_and_grads)
        total_c = loss_comp["total"]
        if torch.isnan(total_c) or torch.isinf(total_c):
            return torch.tensor(float("inf"), device=total_c.device)
        total_c.backward()
        return total_c

def train_model(
    model: EnergyPINNStatic,
    epochs: int = 3000,
    lr: float = 1e-3,
    N_samples: int = 4096,
    print_every: int = 200,
    best_model_path: Optional[str] = None,
    optimizer_type: str = "Adam",
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    lbfgs_line_search_fn: Optional[str] = "strong_wolfe",
    adamw_weight_decay: float = 1e-4,
):

    opt_upper = optimizer_type.upper()
    if opt_upper == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_upper == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=adamw_weight_decay)
    elif opt_upper == "RADAM":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    elif opt_upper == "NADAM":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif opt_upper == "ADAMAX":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    elif opt_upper == "LBFGS":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn=lbfgs_line_search_fn,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                        f"Options: ['Adam', 'AdamW', 'RAdam', 'NAdam', 'Adamax', 'LBFGS']")

    log_list = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()

        if opt_upper in ("ADAM", "ADAMW", "RADAM", "NADAM", "ADAMAX"):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(N_samples=N_samples)
            total = losses["total"]
            if torch.isnan(total) or torch.isinf(total):
                print(f"[WARNING] epoch {epoch} NaN/Inf loss detected, skipping update")
                continue
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"[WARNING] epoch {epoch} NaN/Inf gradient detected, skipping update")
                continue
            optimizer.step()
            current_loss = float(total.item())
        else:
            x_samples = model._train_samples(N_samples)
            
            closure = LBFGSClosure(model, optimizer, x_samples)
            optimizer.step(closure)
            loss_comp = model.weighted_loss_func.compute_total_loss(x_samples, model.fields_and_grads)
            losses = {
                "Pi_str": loss_comp["Pi_str"],
                "Pi_T": loss_comp["Pi_str_T"],
                "Pi_e": loss_comp["Pi_e"],
                "Pi_all": loss_comp["Pi_all"],
                "bc": loss_comp["bc"],
                "total": loss_comp["total"],
            }
            current_loss = float(losses["total"].item())
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                print(f"[WARNING] epoch {epoch} NaN/Inf loss detected, skipping logging")
                continue

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            if best_model_path:
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"[WARNING] Failed to save model (epoch {epoch}): {e}")

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print("=" * 60)
            msg = (
                f"[{epoch:5d}/{epochs}] loss={current_loss:.4e} "
                f"(best: {best_loss:.4e} @ epoch: {best_epoch})  "
                f"Pi_str={losses['Pi_str'].item():.4e}  "
            )
            if "Pi_T" in losses and float(losses["Pi_T"].item()) != 0.0:
                msg += f"Pi_T={losses['Pi_T'].item():.4e}  "
            msg += (
                f"Pi_e={losses['Pi_e'].item():.4e}  "
                f"Pi_all={losses['Pi_all'].item():.4e}  "
                f"bc={losses['bc'].item():.4e}"
            )
            print(msg)
            
            if epoch % 1000 == 0 and epoch > 0:
                try:
                    try:
                        from ..utils.gpu_monitor import get_gpu_status_string
                    except ImportError:
                        import sys, os
                        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from utils.gpu_monitor import get_gpu_status_string
                    
                    gpu_status = get_gpu_status_string()
                    print(f"      [GPU Status] {gpu_status}")
                except Exception:
                    if torch.cuda.is_available():
                        gpu_id = torch.cuda.current_device()
                        gpu_name = torch.cuda.get_device_name(gpu_id)
                        gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        print(f"      [GPU Status] {gpu_name}: Allocated {gpu_memory_allocated:.2f}GB / Reserved {gpu_memory_reserved:.2f}GB")

        log_list.append([
            epoch,
            current_loss,
            float(losses.get("Pi_all").item()) if isinstance(losses.get("Pi_all"), torch.Tensor) else float(losses.get("Pi_all", float("nan"))),
            float(losses.get("bc").item()) if isinstance(losses.get("bc"), torch.Tensor) else float(losses.get("bc", float("nan"))),
            float(losses.get("Pi_str").item()) if isinstance(losses.get("Pi_str"), torch.Tensor) else float(losses.get("Pi_str", float("nan"))),
            float(losses.get("Pi_T").item()) if isinstance(losses.get("Pi_T"), torch.Tensor) else float(losses.get("Pi_T", 0.0)),
            float(losses.get("Pi_e").item()) if isinstance(losses.get("Pi_e"), torch.Tensor) else float(losses.get("Pi_e", float("nan"))),
        ])

    return np.array(log_list, dtype=np.float64), best_loss

def build_model(
    problem: str,
    *,
    coeffs: MaterialCoeffs,
    params: PhysicalParams,
    bc: BoundaryConditions,
    device: torch.device,
    bc_weight: float,
    encoder_dims_shared: Optional[list] = None,
    head_dims: Optional[list] = None,
    in_dim: int = 1,
    sampler: str = "uniform",
    sampler_reuse: bool = False,
    integrator: str = "mc",
    agq_rule: str = "G10K21",
    agq_abs_tol: float = 1e-6,
    agq_rel_tol: float = 1e-4,
    agq_max_points: int = 4096,
    agq_max_depth: int = 100,
    agq_refine_every: int = 0,
    agq_fail_policy: str = "use_partial",
    activation_type: str = "Tanh",
    siren_omega_0: float = 30.0,
    siren_omega_hidden: float = 30.0,
    lifting_basis: str = "poly",
) -> EnergyPINNStatic:

    is_nonlinear = problem.lower() == "nonlinear"
    model = EnergyPINNStatic(
        coeffs,
        params,
        bc,
        device=device,
        bc_type=bc.type,
        bc_weight=bc_weight,
        is_nonlinear=is_nonlinear,
        encoder_dims_shared=encoder_dims_shared,
        head_dims=head_dims,
        in_dim=in_dim,
        sampler=sampler,
        sampler_reuse=sampler_reuse,
        integrator=integrator,
        agq_rule=agq_rule,
        agq_abs_tol=agq_abs_tol,
        agq_rel_tol=agq_rel_tol,
        agq_max_points=agq_max_points,
        agq_max_depth=agq_max_depth,
        agq_refine_every=agq_refine_every,
        agq_fail_policy=agq_fail_policy,
        activation_type=activation_type,
        siren_omega_0=siren_omega_0,
        siren_omega_hidden=siren_omega_hidden,
        lifting_basis=lifting_basis,
    )
    return model

__all__ = ["EnergyPINNStatic", "as_fun", "train_model", "build_model"]
