#!/usr/bin/env python3
"""2D displacement comparison plot for the Open_Source release.

Layout mirrors Figs/Comparison_u_w_phi/k1=k2=0/comparison_u_w_phi_2d.py
with the middle column changed from PINN to FEM (N=1500).

  2x3 grid:
    columns:   DQ Method | FEM N=1500 | PS-PINN (with TL)
    row 0:     U(x,z)  = u + z * phi   (FSDT axial)
    row 1:     W(x,z)  = w             (lateral deflection)

Each panel:
  - jet pcolormesh on deformed (X, Z + deform_scale * W) mesh
  - beam outline (4 black edges, lw=1.2)
  - boundary supports (clamped rectangle / hinged triangle / simply-supported circle)
  - z/h y-tick labels (col 0 only)
  - per-panel colorbar (shrink=0.6, pad=0.02)
  - subplot labels (a)-(f) in upper-left
  - two-line title: <method> \n <field formula> (R^2=...)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Reuse loaders from plot_1d
from plot_1d import (  # noqa: E402
    _long, find_run_dir, load_dq, load_fem,
    build_and_load_model, forward_at,
)
from utils.figs_plotting import add_subplot_labels, set_matplotlib_style  # noqa: E402
import params as P  # noqa: E402

# ----------------------------- Matplotlib style -----------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
set_matplotlib_style(font_size=14)


# ----------------------------- FSDT 2D expansion ----------------------------
def expand_to_2d_grid(x_1d: np.ndarray, u_1d: np.ndarray, w_1d: np.ndarray,
                      phi_1d: np.ndarray, Nz: int = 21) -> Dict[str, np.ndarray]:
    """FSDT: U(x,z) = u(x) + z * phi(x);  W(x,z) = w(x);  z in [-0.5, 0.5]."""
    z_1d = np.linspace(-0.5, 0.5, Nz)
    X, Z = np.meshgrid(x_1d, z_1d)
    U_2d = u_1d[np.newaxis, :] + Z * phi_1d[np.newaxis, :]
    W_2d = np.tile(w_1d, (Nz, 1))
    return {"x": x_1d, "z": z_1d, "X": X, "Z": Z, "U": U_2d, "W": W_2d}


def interpolate_1d(x_src: np.ndarray, fields: Dict[str, np.ndarray],
                   x_target: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for k in ("u", "w", "phi"):
        if k not in fields:
            continue
        f = interp1d(x_src, fields[k], kind="cubic",
                     bounds_error=False, fill_value="extrapolate")
        out[k] = f(x_target)
    return out


# ----------------------------- Boundary supports ----------------------------
def _draw_clamped(ax, side: str, z_bot: float, z_top: float,
                  wall_width: float = 0.04, wall_extend: float = 0.15) -> None:
    x0 = -wall_width if side == "left" else 1.0
    ax.add_patch(Rectangle(
        (x0, z_bot - wall_extend),
        wall_width, (z_top - z_bot) + 2 * wall_extend,
        facecolor="0.6", edgecolor="0.3", linewidth=1.5, zorder=5,
    ))


def _draw_hinged(ax, side: str, z_bot: float,
                 tri_base: float = 0.06, tri_height: float = 0.12) -> None:
    x0 = 0.0 if side == "left" else 1.0
    ax.add_patch(Polygon(
        [(x0, z_bot), (x0 - tri_base / 2, z_bot - tri_height),
         (x0 + tri_base / 2, z_bot - tri_height)],
        closed=True, facecolor="0.6", edgecolor="0.3",
        linewidth=1.5, zorder=5,
    ))


def _draw_simply_supported(ax, side: str, z_bot: float, radius: float = 0.04) -> None:
    x0 = 0.0 if side == "left" else 1.0
    ax.add_patch(plt.Circle((x0, z_bot - radius), radius,
                            facecolor="0.6", edgecolor="0.3",
                            linewidth=1.5, zorder=5))


def draw_supports(ax, bc_type: str, Z_def: np.ndarray) -> None:
    parts = bc_type.split("-")
    if len(parts) != 2:
        return
    left_bc, right_bc = parts
    z_left_bot, z_left_top = Z_def[0, 0],  Z_def[-1, 0]
    z_right_bot, z_right_top = Z_def[0, -1], Z_def[-1, -1]
    if left_bc == "C":   _draw_clamped(ax, "left",  z_left_bot,  z_left_top)
    elif left_bc == "H": _draw_hinged(ax, "left",  z_left_bot)
    elif left_bc == "S": _draw_simply_supported(ax, "left",  z_left_bot)
    if right_bc == "C":  _draw_clamped(ax, "right", z_right_bot, z_right_top)
    elif right_bc == "H":_draw_hinged(ax, "right", z_right_bot)
    elif right_bc == "S":_draw_simply_supported(ax, "right", z_right_bot)


# ----------------------------- Plot helpers ---------------------------------
def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_field_r2(dq_2d: Optional[Dict], pred_2d: Optional[Dict], field: str) -> float:
    if dq_2d is None or pred_2d is None:
        return float("nan")
    return compute_r2(dq_2d[field].flatten(), pred_2d[field].flatten())


def plot_beam_2d_field(ax, data_2d: Dict, field_type: str, title: str,
                       deform_scale: float, cmap: str, vmin: float, vmax: float,
                       bc_type: str, visual_ratio: float = 5.0,
                       show_ylabel: bool = True):
    """One panel: pcolormesh on deformed mesh + boundary supports."""
    X, Z, W_2d = data_2d["X"], data_2d["Z"], data_2d["W"]
    field = data_2d["U"] if field_type == "U" else data_2d["W"]

    X_def = X.copy()
    Z_def = Z + deform_scale * W_2d

    pcm = ax.pcolormesh(X_def, Z_def, field, shading="gouraud",
                        cmap=cmap, vmin=vmin, vmax=vmax, zorder=1,
                        rasterized=True)  # CRITICAL: avoid 100MB+ SVGs from per-triangle paths
    # beam outline
    ax.plot(X_def[0, :],  Z_def[0, :],  "k-", lw=1.2, zorder=2)
    ax.plot(X_def[-1, :], Z_def[-1, :], "k-", lw=1.2, zorder=2)
    ax.plot(X_def[:, 0],  Z_def[:, 0],  "k-", lw=1.2, zorder=2)
    ax.plot(X_def[:, -1], Z_def[:, -1], "k-", lw=1.2, zorder=2)
    # supports
    draw_supports(ax, bc_type, Z_def)
    # cosmetics
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    if show_ylabel:
        ax.set_ylabel(r"$z/h$", fontsize=15)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    x_margin, z_margin = 0.08, 0.20
    ax.set_xlim(-x_margin, 1.0 + x_margin)
    ax.set_ylim(Z_def.min() - z_margin, Z_def.max() + z_margin)
    ax.set_aspect(1.0 / visual_ratio, adjustable="box")
    return pcm


def plot_displacement_2d_comparison(
    dq_2d: Optional[Dict],
    fem_2d: Optional[Dict],
    ps_pinn_2d: Optional[Dict],
    output_path: Path,
    problem_type: str = "nonlinear",
    bc_type: str = "C-C",
    deform_scale: float = 1.0,
    visual_ratio: float = 5.0,
    dpi: int = 300,
) -> None:
    """2x3 layout: DQ / FEM / PS-PINN(with TL) x U / W."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    col_labels = ["DQ Method", "FEM N=1500", "PS-PINN (with TL)"]
    row_labels = [r"$U = u + z\cdot\phi$", r"$W = w$"]
    field_types = ["U", "W"]
    data_list = [dq_2d, fem_2d, ps_pinn_2d]

    # Global color range
    U_all = [d["U"] for d in data_list if d is not None]
    W_all = [d["W"] for d in data_list if d is not None]
    U_min, U_max = (min(u.min() for u in U_all), max(u.max() for u in U_all)) if U_all else (-1, 1)
    W_min, W_max = (min(w.min() for w in W_all), max(w.max() for w in W_all)) if W_all else (-1, 1)

    # R^2 of FEM and PS-PINN vs DQ (per field, on the common grid built by the caller)
    r2_values = {}
    for ft in ("U", "W"):
        r2_values[("FEM",     ft)] = compute_field_r2(dq_2d, fem_2d,     ft)
        r2_values[("PS-PINN", ft)] = compute_field_r2(dq_2d, ps_pinn_2d, ft)

    cmap = "jet"
    for row, (ft, row_label) in enumerate(zip(field_types, row_labels)):
        for col, (data, col_label) in enumerate(zip(data_list, col_labels)):
            ax = axes[row, col]
            if data is None:
                ax.text(0.5, 0.5, f"No {col_label} data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=15, color="gray", style="italic")
                ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
                for s in ax.spines.values(): s.set_visible(False)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{col_label}\n{row_label}", fontsize=16, fontweight="bold")
                continue

            vmin = U_min if ft == "U" else W_min
            vmax = U_max if ft == "U" else W_max

            if col == 0:  # DQ reference -- no R^2
                title = f"{col_label}\n{row_label}"
            else:
                method_key = "FEM" if col == 1 else "PS-PINN"
                r2_val = r2_values.get((method_key, ft), float("nan"))
                title = (f"{col_label}\n{row_label} ($R^2$={r2_val:.6f})"
                         if not np.isnan(r2_val)
                         else f"{col_label}\n{row_label}")

            pcm = plot_beam_2d_field(ax, data, ft, title,
                                      deform_scale, cmap, vmin, vmax,
                                      bc_type, visual_ratio,
                                      show_ylabel=(col == 0))
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=12)

    add_subplot_labels(axes, fontsize=15, y=1.28)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_long(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(_long(svg_path), bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved: {output_path}")
    print(f"[OK] Saved: {svg_path}")
    plt.close(fig)


# ----------------------------- main -----------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Open_Source 2D comparison plot (DQ | FEM N=1500 | PS-PINN with TL)"
    )
    ap.add_argument("--run-dir", default=None,
                    help="Path to the main.py run directory (default: auto-detect)")
    ap.add_argument("--nz", type=int, default=21, help="Thickness mesh resolution")
    ap.add_argument("--nx", type=int, default=201, help="Streamwise common-grid size")
    ap.add_argument("--problem-type", choices=["nonlinear", "linear", "both"],
                    default="nonlinear",
                    help="Which problem to plot (default: nonlinear)")
    ap.add_argument("--deform-scale", type=float, default=1.0,
                    help="Deformation magnification factor (default: 1.0)")
    ap.add_argument("--visual-ratio", type=float, default=5.0,
                    help="Beam visual aspect ratio (default: 5.0)")
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    bc_type = str(getattr(P, "bc_type", "C-C"))
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] bc_type = {bc_type}")

    # ---- 1D fields from three sources (nonlinear branch is the primary target) ----
    x_dq,  dq_1d_nl  = load_dq()
    x_fem, fem_1d_nl = load_fem()
    model, device, pth_path = build_and_load_model(run_dir)
    print(f"[INFO] Loaded PS-PINN pth: {pth_path.name}")

    # Common dense grid for visualisation
    x_common = np.linspace(0.0, 1.0, args.nx)
    pinn_1d_nl = forward_at(model, device, x_common)

    # ---- Linear branch (optional, if requested and DQ/FEM/Lw_*.pth available) ----
    has_linear = False
    if args.problem_type in ("linear", "both"):
        dq_csv = HERE / "DQ" / "results_WGr0.025_q-0.080_k1_0.010_k2_0.001_C-C.csv"
        fem_csv = (HERE / "FEM" / "k1_0.01_k2_0.001" / "N1500"
                   / "w_W_0.025_T_300_H_0.8_q_n0.08_k1_0.01_k2_0.001_N1500.csv")
        if dq_csv.exists() and fem_csv.exists():
            df_dq  = pd.read_csv(_long(dq_csv))
            df_fem = pd.read_csv(_long(fem_csv))
            dq_1d_lin  = {k: df_dq[f"{k}_linear"].values  for k in ("u", "w", "phi")}
            fem_1d_lin = {k: df_fem[f"linear_{k}"].values for k in ("u", "w", "phi")}
            # PS-PINN linear branch -- look for Lw_*.pth
            lw_files = sorted((run_dir / "models").glob("Lw_*.pth"))
            if lw_files:
                from modules.solver import build_model, as_fun
                from modules.data_types import MaterialCoeffs, PhysicalParams
                from modules.bc import make_bc_spec
                from utils.material_properties import compute_material_params_for_solver
                h = float(P.h); L = float(P.L_factor) * h
                mat = compute_material_params_for_solver(
                    h=h, L=L, num_layers=int(P.num_layers),
                    W_Gr=float(P.W_Gr), H_Gr=float(P.H_Gr),
                    T=float(P.T), distribution_type=str(P.distribution),
                    q=float(P.q),
                )
                coeffs = MaterialCoeffs(
                    a11=as_fun(mat["a11"], "a11"), b11=as_fun(mat["b11"], "b11"),
                    d11=as_fun(mat["d11"], "d11"), a55=as_fun(mat["a55"], "a55"),
                )
                params_obj = PhysicalParams(
                    alpha_t=mat["alpha_effective"], DeltaT=mat["delta_T"],
                    lambda_val=mat["lambda_val"],
                    q=mat["q"], n_xT=mat["n_xT"], m_xT=mat["m_xT"],
                    k1=float(getattr(P, "k1", 0.0)), k2=float(getattr(P, "k2", 0.0)),
                    A0=float(getattr(P, "A0", 0.0)), a=float(getattr(P, "a", 0.0)),
                    b=float(getattr(P, "b", 0.0)), c=float(getattr(P, "c", 0.5)),
                )
                bc = make_bc_spec(bc_type)
                lin_model = build_model(
                    "linear", coeffs=coeffs, params=params_obj, bc=bc, device=device,
                    bc_weight=float(getattr(P, "bc_weight", 1000.0)),
                    encoder_dims_shared=getattr(P, "encoder_dims_shared", [1, 32, 64, 128]),
                    head_dims=getattr(P, "head_dims", [128, 64, 32, 1]), in_dim=1,
                    activation_type=str(getattr(P, "activation_type", "Tanh")),
                    siren_omega_0=float(getattr(P, "siren_omega_0", 30.0)),
                    siren_omega_hidden=float(getattr(P, "siren_omega_hidden", 30.0)),
                    lifting_basis=str(getattr(P, "lifting_basis", "poly")),
                )
                lin_model.load_state_dict(torch.load(_long(lw_files[0]), map_location=device))
                lin_model.eval()
                pinn_1d_lin = forward_at(lin_model, device, x_common)
                has_linear = True

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resample everything onto x_common, then expand to 2D ----
    def to_2d(x_src, fields_1d):
        f_common = interpolate_1d(x_src, fields_1d, x_common)
        return expand_to_2d_grid(x_common, f_common["u"], f_common["w"], f_common["phi"], args.nz)

    # Nonlinear figure
    if args.problem_type in ("nonlinear", "both"):
        dq_2d_nl   = to_2d(x_dq,  dq_1d_nl)
        fem_2d_nl  = to_2d(x_fem, fem_1d_nl)
        pinn_2d_nl = expand_to_2d_grid(x_common, pinn_1d_nl["u"], pinn_1d_nl["w"],
                                       pinn_1d_nl["phi"], args.nz)
        plot_displacement_2d_comparison(
            dq_2d_nl, fem_2d_nl, pinn_2d_nl,
            plots_dir / "comparison_nonlinear_2d.png",
            problem_type="nonlinear", bc_type=bc_type,
            deform_scale=args.deform_scale, visual_ratio=args.visual_ratio,
        )

    # Linear figure
    if args.problem_type in ("linear", "both") and has_linear:
        dq_2d_lin   = to_2d(x_dq,  dq_1d_lin)
        fem_2d_lin  = to_2d(x_fem, fem_1d_lin)
        pinn_2d_lin = expand_to_2d_grid(x_common, pinn_1d_lin["u"], pinn_1d_lin["w"],
                                        pinn_1d_lin["phi"], args.nz)
        plot_displacement_2d_comparison(
            dq_2d_lin, fem_2d_lin, pinn_2d_lin,
            plots_dir / "comparison_linear_2d.png",
            problem_type="linear", bc_type=bc_type,
            deform_scale=args.deform_scale, visual_ratio=args.visual_ratio,
        )
    elif args.problem_type in ("linear", "both"):
        print("[WARN] Linear plot skipped (Lw_*.pth or DQ/FEM linear data missing).")

    # ---- Print R^2 summary ----
    print()
    print("R^2 (FEM and PS-PINN vs DQ, common 201x21 grid):")
    if args.problem_type in ("nonlinear", "both"):
        print(f"  Nonlinear U:  FEM = {compute_field_r2(dq_2d_nl, fem_2d_nl, 'U'):.6f}   "
              f"PS-PINN = {compute_field_r2(dq_2d_nl, pinn_2d_nl, 'U'):.6f}")
        print(f"  Nonlinear W:  FEM = {compute_field_r2(dq_2d_nl, fem_2d_nl, 'W'):.6f}   "
              f"PS-PINN = {compute_field_r2(dq_2d_nl, pinn_2d_nl, 'W'):.6f}")
    if has_linear and args.problem_type in ("linear", "both"):
        print(f"  Linear    U:  FEM = {compute_field_r2(dq_2d_lin, fem_2d_lin, 'U'):.6f}   "
              f"PS-PINN = {compute_field_r2(dq_2d_lin, pinn_2d_lin, 'U'):.6f}")
        print(f"  Linear    W:  FEM = {compute_field_r2(dq_2d_lin, fem_2d_lin, 'W'):.6f}   "
              f"PS-PINN = {compute_field_r2(dq_2d_lin, pinn_2d_lin, 'W'):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
