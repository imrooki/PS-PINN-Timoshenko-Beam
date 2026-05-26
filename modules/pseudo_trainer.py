
from __future__ import annotations

from typing import Dict, Optional, Callable, Any
import math
import os

import torch

from modules.solver import build_model
from modules.data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
from modules.solver import EnergyPINNStatic
from modules.numerics import sample_1d
from utils.gpu_monitor import get_gpu_status_string

def _ps_weight(epoch: int, total: int, start_val: float, end_at_ratio: float = 0.8) -> float:
    if total <= 0:
        return 0.0
    t = epoch / float(total)
    if t >= end_at_ratio:
        return 0.0
    if end_at_ratio <= 0.0:
        return 0.0
    frac = t / end_at_ratio
    return float(start_val * max(0.0, 1.0 - frac))

def _encoder_soft_update(
    target: EnergyPINNStatic,
    source: EnergyPINNStatic,
    *,
    alpha: float = 0.3,
    ratio: float = 0.7,
) -> None:
    t_state = target.net.state_dict()
    s_state = source.net.state_dict()

    enc_keys = [k for k in t_state.keys() if k.startswith("encoder.")]
    layer_indices = {}
    for name in enc_keys:
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1])
            layer_indices.setdefault(idx, []).append(name)
    if not layer_indices:
        return
    sorted_idx = sorted(layer_indices.keys())
    num_layers = len(sorted_idx)
    take = max(1, int(math.ceil(num_layers * max(0.0, min(1.0, ratio)))))
    sel = sorted_idx[:take]
    a = float(max(0.0, min(1.0, alpha)))
    for idx in sel:
        for name in layer_indices[idx]:
            t_param = t_state[name]
            s_param = s_state.get(name, None)
            if s_param is None or t_param.shape != s_param.shape:
                continue
            t_state[name] = (1.0 - a) * t_param + a * s_param
    target.net.load_state_dict(t_state)

def run_dual_pseudo_transfer(
    *,
    coeffs: MaterialCoeffs,
    params_obj: PhysicalParams,
    bc: BoundaryConditions,
    device: torch.device,
    encoder_dims_shared: list,
    head_dims: list,
    in_dim: int,
    epochs: int,
    N_train: int,
    lr: float,
    print_every: int,
    bc_weight: float,
    optimizer_type: str,
    lbfgs_max_iter: int,
    lbfgs_history_size: int,
    lbfgs_line_search_fn: Optional[str],
    adamw_weight_decay: float = 1e-4,
    sampler: str,
    sampler_reuse: bool,
    integrator: str,
    agq_rule: str,
    agq_abs_tol: float,
    agq_rel_tol: float,
    agq_max_points: int,
    agq_max_depth: int,
    agq_refine_every: int,
    agq_fail_policy: str,
    use_adaptive_lr: bool,
    lr_early_max: float,
    lr_early_min: float,
    lr_mid_max: float,
    lr_mid_min: float,
    lr_late_fixed: float,
    lr_patience: int,
    lr_improvement_threshold: float,
    lr_decay_factor: float,
    lr_verbose: bool,
    lr_warmup_epochs: int = 100,
    lr_early_ratio: float = 0.6,
    lr_mid_ratio: float = 0.85,
    lr_min_early_epochs: int = 1000,
    lr_min_mid_epochs: int = 5000,
    use_pseudo_supervision: bool = True,
    ps_w_non_start: float = 1.0,
    ps_w_lin_start: float = 0.5,
    ps_cut_ratio: float = 0.8,
    ps_use_phi: bool = False,
    transfer_alpha: float = 0.3,
    transfer_ratio: float = 0.7,
    transfer_freq: int = 500,
    transfer_cut_ratio: float = 0.2,
    x_eval: Optional[torch.Tensor] = None,
    epoch_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    activation_type: str = "Tanh",
    siren_omega_0: float = 30.0,
    siren_omega_hidden: float = 30.0,
    lifting_basis: str = "poly",
) -> Dict[str, object]:
    print(f"[Pseudo-Supervision] Status: {'ENABLED' if use_pseudo_supervision else 'DISABLED'}")
    if use_pseudo_supervision:
        print(f"[Pseudo-Supervision] Parameters: ps_w_non_start={ps_w_non_start}, ps_w_lin_start={ps_w_lin_start}, ps_cut_ratio={ps_cut_ratio}")

    model_lin = build_model(
        'linear',
        coeffs=coeffs,
        params=params_obj,
        bc=bc,
        device=device,
        bc_weight=bc_weight,
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
    model_non = build_model(
        'nonlinear',
        coeffs=coeffs,
        params=params_obj,
        bc=bc,
        device=device,
        bc_weight=bc_weight,
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

    opt_upper = optimizer_type.upper()
    if opt_upper == "LBFGS":
        raise ValueError("LBFGS optimizer is not supported in dual pseudo-supervision training mode. "
                        "Use Adam, AdamW, RAdam, NAdam, or Adamax instead.")

    initial_lr = lr_early_max if use_adaptive_lr else lr

    if opt_upper == "ADAM":
        opt_lin = torch.optim.Adam(model_lin.parameters(), lr=initial_lr)
        opt_non = torch.optim.Adam(model_non.parameters(), lr=initial_lr)
    elif opt_upper == "ADAMW":
        opt_lin = torch.optim.AdamW(model_lin.parameters(), lr=initial_lr, weight_decay=adamw_weight_decay)
        opt_non = torch.optim.AdamW(model_non.parameters(), lr=initial_lr, weight_decay=adamw_weight_decay)
    elif opt_upper == "RADAM":
        opt_lin = torch.optim.RAdam(model_lin.parameters(), lr=initial_lr)
        opt_non = torch.optim.RAdam(model_non.parameters(), lr=initial_lr)
    elif opt_upper == "NADAM":
        opt_lin = torch.optim.NAdam(model_lin.parameters(), lr=initial_lr)
        opt_non = torch.optim.NAdam(model_non.parameters(), lr=initial_lr)
    elif opt_upper == "ADAMAX":
        opt_lin = torch.optim.Adamax(model_lin.parameters(), lr=initial_lr)
        opt_non = torch.optim.Adamax(model_non.parameters(), lr=initial_lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                        f"Options: ['Adam', 'AdamW', 'RAdam', 'NAdam', 'Adamax']")

    scheduler_lin = None
    scheduler_non = None
    if use_adaptive_lr:
        from utils.adaptive_lr_scheduler import LossBasedAdaptiveLRScheduler
        scheduler_lin = LossBasedAdaptiveLRScheduler(
            opt_lin,
            lr_early_max=lr_early_max,
            lr_early_min=lr_early_min,
            lr_mid_max=lr_mid_max,
            lr_mid_min=lr_mid_min,
            lr_late_fixed=lr_late_fixed,
            patience=lr_patience,
            improvement_threshold=lr_improvement_threshold,
            lr_decay_factor=lr_decay_factor,
            verbose=lr_verbose,
            warmup_epochs=lr_warmup_epochs,
            early_ratio=lr_early_ratio,
            mid_ratio=lr_mid_ratio,
            min_early_epochs=lr_min_early_epochs,
            min_mid_epochs=lr_min_mid_epochs
        )
        scheduler_non = LossBasedAdaptiveLRScheduler(
            opt_non,
            lr_early_max=lr_early_max,
            lr_early_min=lr_early_min,
            lr_mid_max=lr_mid_max,
            lr_mid_min=lr_mid_min,
            lr_late_fixed=lr_late_fixed,
            patience=lr_patience,
            improvement_threshold=lr_improvement_threshold,
            lr_decay_factor=lr_decay_factor,
            verbose=lr_verbose,
            warmup_epochs=lr_warmup_epochs,
            early_ratio=lr_early_ratio,
            mid_ratio=lr_mid_ratio,
            min_early_epochs=lr_min_early_epochs,
            min_mid_epochs=lr_min_mid_epochs
        )

    logs_lin = []
    logs_non = []
    best_lin = float('inf')
    best_non = float('inf')
    best_lin_state = None
    best_non_state = None
    best_lin_epoch = 0
    best_non_epoch = 0

    for epoch in range(1, epochs + 1):
        x = sample_1d(N_train, device, sampler=sampler, dtype=torch.float32)

        comp_lin = model_lin.weighted_loss_func.compute_total_loss(x, model_lin.fields_and_grads)
        comp_non = model_non.weighted_loss_func.compute_total_loss(x, model_non.fields_and_grads)

        if use_pseudo_supervision:
            fields_lin = model_lin.fields_and_grads(x)
            fields_non = model_non.fields_and_grads(x)
            w_lin = fields_lin['w']
            w_non = fields_non['w']
            w_lin_ps = w_non.detach()
            w_non_ps = w_lin.detach()

            ps_non_w = _ps_weight(epoch, epochs, ps_w_non_start, ps_cut_ratio)
            ps_lin_w = _ps_weight(epoch, epochs, ps_w_lin_start, ps_cut_ratio)

            ps_loss_lin = ps_lin_w * torch.mean((w_lin - w_lin_ps) ** 2)
            ps_loss_non = ps_non_w * torch.mean((w_non - w_non_ps) ** 2)

            if ps_use_phi:
                phi_lin = fields_lin['phi']
                phi_non = fields_non['phi']
                phi_lin_ps = phi_non.detach()
                phi_non_ps = phi_lin.detach()
                ps_loss_lin = ps_loss_lin + 0.2 * ps_lin_w * torch.mean((phi_lin - phi_lin_ps) ** 2)
                ps_loss_non = ps_loss_non + 0.2 * ps_non_w * torch.mean((phi_non - phi_non_ps) ** 2)

            L_lin = comp_lin['total'] + ps_loss_lin
            L_non = comp_non['total'] + ps_loss_non
        else:
            ps_non_w = 0.0
            ps_lin_w = 0.0
            L_lin = comp_lin['total']
            L_non = comp_non['total']

        opt_lin.zero_grad(set_to_none=True)
        opt_non.zero_grad(set_to_none=True)

        L_lin.backward()
        torch.nn.utils.clip_grad_norm_(model_lin.parameters(), 1.0)
        opt_lin.step()

        L_non.backward()
        torch.nn.utils.clip_grad_norm_(model_non.parameters(), 1.0)
        opt_non.step()

        tl = float(L_lin.detach().item())
        tn = float(L_non.detach().item())
        if tl < best_lin:
            best_lin = tl
            best_lin_state = {k: v.detach().cpu().clone() for k, v in model_lin.state_dict().items()}
            best_lin_epoch = epoch
        if tn < best_non:
            best_non = tn
            best_non_state = {k: v.detach().cpu().clone() for k, v in model_non.state_dict().items()}
            best_non_epoch = epoch

        if scheduler_lin is not None:
            scheduler_lin.step(tl)
        if scheduler_non is not None:
            scheduler_non.step(tn)

        def get_energy_val(comp_dict: Dict[str, torch.Tensor], key: str, default: float = 0.0) -> float:
            tensor_val = comp_dict.get(key)
            if tensor_val is None:
                return default
            return tensor_val.item()

        lin_piall = get_energy_val(comp_lin, 'Pi_all')
        lin_bc = get_energy_val(comp_lin, 'bc')
        lin_pistr = get_energy_val(comp_lin, 'Pi_str')
        lin_pistrt = get_energy_val(comp_lin, 'Pi_str_T')
        lin_piw = get_energy_val(comp_lin, 'Pi_w')
        lin_pie = get_energy_val(comp_lin, 'Pi_e')

        non_piall = get_energy_val(comp_non, 'Pi_all')
        non_bc = get_energy_val(comp_non, 'bc')
        non_pistr = get_energy_val(comp_non, 'Pi_str')
        non_pistrt = get_energy_val(comp_non, 'Pi_str_T')
        non_piw = get_energy_val(comp_non, 'Pi_w')
        non_pie = get_energy_val(comp_non, 'Pi_e')

        if use_pseudo_supervision:
            lin_pseudo = float(ps_loss_lin.detach().item())
            non_pseudo = float(ps_loss_non.detach().item())
        else:
            lin_pseudo = 0.0
            non_pseudo = 0.0

        logs_lin.append([epoch, tl, lin_piall, lin_bc, lin_pistr, lin_pistrt, lin_piw, lin_pie, lin_pseudo])
        logs_non.append([epoch, tn, non_piall, non_bc, non_pistr, non_pistrt, non_piw, non_pie, non_pseudo])

        if use_pseudo_supervision and transfer_freq > 0 and (epoch % transfer_freq == 0) and (epoch < transfer_cut_ratio * epochs):
            if tl <= tn:
                _encoder_soft_update(model_non, model_lin, alpha=transfer_alpha, ratio=transfer_ratio)
            else:
                _encoder_soft_update(model_lin, model_non, alpha=transfer_alpha, ratio=transfer_ratio)

        if (epoch % print_every == 0) or epoch == 1 or epoch == epochs:
            lr_lin = scheduler_lin.get_lr() if scheduler_lin is not None else initial_lr
            lr_non = scheduler_non.get_lr() if scheduler_non is not None else initial_lr
            phase_lin = scheduler_lin.get_phase() if scheduler_lin is not None else "fixed"
            phase_non = scheduler_non.get_phase() if scheduler_non is not None else "fixed"

            print("=" * 60)
            if use_adaptive_lr:
                print(f"[LR] Linear: {lr_lin:.2e} ({phase_lin})  |  Nonlinear: {lr_non:.2e} ({phase_non})")
            print(
                f"[Dual {epoch:5d}/{epochs}] Linear: L={tl:.4e} (lr={lr_lin:.2e}) (best_linear={best_lin:.4e} @ epoch: {best_lin_epoch})  "
                f"Pi_str={lin_pistr:.4e}  Pi_str_T={lin_pistrt:.4e}  Pi_w={lin_piw:.4e}  Pi_e={lin_pie:.4e}  Pi_all={lin_piall:.4e}  bc={lin_bc:.4e}"
            )
            print(
                f"               Nonlinear: L={tn:.4e} (lr={lr_non:.2e}) (best_nonlinear={best_non:.4e} @ epoch: {best_non_epoch})  "
                f"Pi_str={non_pistr:.4e}  Pi_str_T={non_pistrt:.4e}  Pi_w={non_piw:.4e}  Pi_e={non_pie:.4e}  Pi_all={non_piall:.4e}  bc={non_bc:.4e}  "
                f"(ps_w: {'disabled' if not use_pseudo_supervision else f'non={ps_non_w:.3f}, lin={ps_lin_w:.3f}'})"
            )

        if epoch_callback is not None:
            callback_context = {
                "epoch": epoch,
                "total_epochs": epochs,
                "model_linear": model_lin,
                "model_nonlinear": model_non,
                "optimizer_linear": opt_lin,
                "optimizer_nonlinear": opt_non,
                "loss_linear": tl,
                "loss_nonlinear": tn,
                "best_lin_loss": best_lin,
                "best_non_loss": best_non,
                "best_lin_epoch": best_lin_epoch,
                "best_non_epoch": best_non_epoch,
                "best_lin_state": best_lin_state,
                "best_non_state": best_non_state,
                "x_eval": x_eval,
                "device": device,
                "use_pseudo_supervision": use_pseudo_supervision,
                "initial_lr": initial_lr,
                "N_train": N_train,
                "encoder_dims_shared": encoder_dims_shared,
                "head_dims": head_dims,
                "in_dim": in_dim,
                "bc_weight": bc_weight,
                "bc": bc,
            }
            epoch_callback(epoch, callback_context)

        if epoch % 1000 == 0:
            try:
                print(f"      [GPU Status] {get_gpu_status_string()}")
            except Exception:
                pass

    if best_lin_state is not None:
        model_lin.load_state_dict(best_lin_state)
    if best_non_state is not None:
        model_non.load_state_dict(best_non_state)

    try:
        print(
            f"[Dual] Best Summary  linear: {best_lin:.6e} @ epoch {best_lin_epoch}; "
            f"nonlinear: {best_non:.6e} @ epoch {best_non_epoch}"
        )
    except Exception:
        pass

    out: Dict[str, object] = {}
    if x_eval is not None:
        model_lin.eval(); model_non.eval()
        fld_lin = model_lin.fields_and_grads(x_eval)
        fld_non = model_non.fields_and_grads(x_eval)
        out['fields_linear'] = fld_lin
        out['fields_nonlinear'] = fld_non

    import numpy as _np
    out['logs_linear'] = _np.array(logs_lin, dtype=_np.float64) if logs_lin else None
    out['logs_nonlinear'] = _np.array(logs_non, dtype=_np.float64) if logs_non else None
    out['best_linear_loss'] = float(best_lin)
    out['best_nonlinear_loss'] = float(best_non)
    out['model_linear'] = model_lin
    out['model_nonlinear'] = model_non
    out['best_linear_epoch'] = best_lin_epoch
    out['best_nonlinear_epoch'] = best_non_epoch
    return out

