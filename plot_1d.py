#!/usr/bin/env python3
"""1D displacement comparison plot for the Open_Source release.

Sources plotted (nonlinear branch only):
  - DQ spectral reference (23 Chebyshev-Gauss-Lobatto nodes)
  - FEM N=1500 reference (1501 uniform nodes)
  - PS-PINN (with TL): NLw_*.pth loaded from the most recent main.py run

Output: 1x3 figure (u, w, phi) saved as <run_dir>/plots/comparison_1d.png
        + a side-by-side data CSV for paper-figure reproduction.

Usage:
    python plot_1d.py                         # default DQ csv (with foundation)
    python plot_1d.py --run-dir <path>        # override PINN run directory
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from modules.data_types import MaterialCoeffs, PhysicalParams  # noqa: E402
from modules.bc import make_bc_spec  # noqa: E402
from modules.solver import build_model, as_fun  # noqa: E402
from utils.material_properties import compute_material_params_for_solver  # noqa: E402
import params as P  # noqa: E402

# ----------------------------- Matplotlib style -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"DQ": "#000000", "FEM": "#ff7f0e", "PS-PINN": "#1f77b4"}

# ----------------------------- helpers --------------------------------------
def _long(p: Path) -> str:
    s = str(p)
    return ("\\\\?\\" + s) if os.name == "nt" and len(s) >= 240 and not s.startswith("\\\\?\\") else s


def find_run_dir(run_dir_override: str = None) -> Path:
    """Find the run directory containing NLw_*.pth."""
    if run_dir_override:
        p = Path(run_dir_override)
        if (p / "models").exists():
            return p
        raise FileNotFoundError(f"--run-dir does not contain models/: {p}")
    case_subpath = ("results/main/C-C/X/"
                    "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh-k0.01_0.001")
    candidates = [HERE / case_subpath,                       # Open_Source/results/main/...
                  HERE.parent.parent / case_subpath]         # project_root/results/main/...
    for c in candidates:
        if (c / "models").exists() and list((c / "models").glob("NLw_*.pth")):
            return c
    raise FileNotFoundError(
        "No run dir with NLw_*.pth found. "
        "Run `python main.py` first, or pass --run-dir <path>."
    )


def load_dq() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    f = HERE / "DQ" / "results_WGr0.025_q-0.080_k1_0.010_k2_0.001_C-C.csv"
    if not f.exists():
        raise FileNotFoundError(f"DQ reference missing: {f}")
    df = pd.read_csv(_long(f))
    return df["x"].values, {k: df[f"{k}_nonlinear"].values for k in ("u", "w", "phi")}


def load_fem() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    f = (HERE / "FEM" / "k1_0.01_k2_0.001" / "N1500"
         / "w_W_0.025_T_300_H_0.8_q_n0.08_k1_0.01_k2_0.001_N1500.csv")
    if not f.exists():
        raise FileNotFoundError(f"FEM reference missing: {f}")
    df = pd.read_csv(_long(f))
    return df["x"].values, {k: df[f"nonlinear_{k}"].values for k in ("u", "w", "phi")}


def build_and_load_model(run_dir: Path):
    """Rebuild the nonlinear PS-PINN model and load the saved NLw_*.pth."""
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
        alpha_t=mat["alpha_effective"], DeltaT=mat["delta_T"], lambda_val=mat["lambda_val"],
        q=mat["q"], n_xT=mat["n_xT"], m_xT=mat["m_xT"],
        k1=float(getattr(P, "k1", 0.0)), k2=float(getattr(P, "k2", 0.0)),
        A0=float(getattr(P, "A0", 0.0)), a=float(getattr(P, "a", 0.0)),
        b=float(getattr(P, "b", 0.0)), c=float(getattr(P, "c", 0.5)),
    )
    bc = make_bc_spec(str(getattr(P, "bc_type", "C-C")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        "nonlinear", coeffs=coeffs, params=params_obj, bc=bc, device=device,
        bc_weight=float(getattr(P, "bc_weight", 1000.0)),
        encoder_dims_shared=getattr(P, "encoder_dims_shared", [1, 32, 64, 128]),
        head_dims=getattr(P, "head_dims", [128, 64, 32, 1]), in_dim=1,
        activation_type=str(getattr(P, "activation_type", "Tanh")),
        siren_omega_0=float(getattr(P, "siren_omega_0", 30.0)),
        siren_omega_hidden=float(getattr(P, "siren_omega_hidden", 30.0)),
        lifting_basis=str(getattr(P, "lifting_basis", "poly")),
    )
    pth = sorted((run_dir / "models").glob("NLw_*.pth"))[0]
    model.load_state_dict(torch.load(_long(pth), map_location=device))
    model.eval()
    return model, device, pth


def forward_at(model, device, x: np.ndarray) -> Dict[str, np.ndarray]:
    xt = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    xt.requires_grad_(True)
    f = model.fields_and_grads(xt)
    return {k: f[k].detach().cpu().numpy().flatten() for k in ("u", "w", "phi")}


def r2(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_ref - y_pred) ** 2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")


# ----------------------------- main plot ------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Open_Source 1D comparison plot")
    ap.add_argument("--run-dir", default=None,
                    help="Path to the main.py run directory (default: auto-detect)")
    ap.add_argument("--n-eval", type=int, default=201,
                    help="Number of uniform nodes for PINN forward (default: 201)")
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    print(f"[INFO] Run dir: {run_dir}")

    # Load three sources
    x_dq, dq = load_dq()
    x_fem, fem = load_fem()
    print(f"[INFO] DQ nodes: {len(x_dq)}, FEM nodes: {len(x_fem)}")

    model, device, pth_path = build_and_load_model(run_dir)
    print(f"[INFO] Loaded PS-PINN pth: {pth_path.name}")
    x_pinn = np.linspace(0.0, 1.0, args.n_eval)
    pinn = forward_at(model, device, x_pinn)

    # Compute R² against DQ at DQ nodes (interpolate FEM/PINN to DQ grid)
    fem_at_dq = {f: interp1d(x_fem, fem[f], kind="cubic")(x_dq) for f in ("u", "w", "phi")}
    pinn_at_dq = {f: interp1d(x_pinn, pinn[f], kind="cubic")(x_dq) for f in ("u", "w", "phi")}

    # Plot 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["u (Axial Displacement)", "w (Deflection)", r"$\phi$ (Rotation)"]
    ylabs  = ["u", "w", r"$\phi$"]
    keys   = ["u", "w", "phi"]

    for col, (k, title, ylab) in enumerate(zip(keys, titles, ylabs)):
        ax = axes[col]
        # DQ markers
        ax.plot(x_dq, dq[k], "o", color=COLORS["DQ"], markersize=6,
                markerfacecolor="white", markeredgewidth=1.5,
                label="DQ (reference)", zorder=3)
        # FEM dashed line
        r2_fem = r2(dq[k], fem_at_dq[k])
        ax.plot(x_fem, fem[k], "--", color=COLORS["FEM"], linewidth=2.0,
                label=f"FEM N=1500 (R$^2$={r2_fem:.4f})", zorder=2)
        # PS-PINN solid line
        r2_pinn = r2(dq[k], pinn_at_dq[k])
        ax.plot(x_pinn, pinn[k], "-", color=COLORS["PS-PINN"], linewidth=1.6,
                label=f"PS-PINN (with TL) (R$^2$={r2_pinn:.4f})", zorder=1)

        ax.set_xlabel("$x/L$", fontsize=15)
        ax.set_ylabel(ylab, fontsize=15)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=10)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    plt.tight_layout()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "comparison_1d.png"
    plt.savefig(_long(out_png), dpi=300, bbox_inches="tight")
    out_svg = plots_dir / "comparison_1d.svg"
    plt.savefig(_long(out_svg), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_svg}")

    # Companion data CSV (paper-figure reproducibility)
    df_out = pd.DataFrame({
        "x_dq": np.concatenate([x_dq, np.full(max(len(x_fem), len(x_pinn)) - len(x_dq), np.nan)]),
        "u_dq": np.concatenate([dq["u"], np.full(max(len(x_fem), len(x_pinn)) - len(x_dq), np.nan)]),
        "w_dq": np.concatenate([dq["w"], np.full(max(len(x_fem), len(x_pinn)) - len(x_dq), np.nan)]),
        "phi_dq": np.concatenate([dq["phi"], np.full(max(len(x_fem), len(x_pinn)) - len(x_dq), np.nan)]),
    })
    # Add fem/pinn columns aligned to max length
    n_max = max(len(x_dq), len(x_fem), len(x_pinn))
    def _pad(arr):
        return np.concatenate([arr, np.full(n_max - len(arr), np.nan)])
    df_out = pd.DataFrame({
        "x_dq":   _pad(x_dq),    "u_dq":   _pad(dq["u"]),    "w_dq":   _pad(dq["w"]),    "phi_dq":   _pad(dq["phi"]),
        "x_fem":  _pad(x_fem),   "u_fem":  _pad(fem["u"]),   "w_fem":  _pad(fem["w"]),   "phi_fem":  _pad(fem["phi"]),
        "x_pinn": _pad(x_pinn),  "u_pinn": _pad(pinn["u"]),  "w_pinn": _pad(pinn["w"]),  "phi_pinn": _pad(pinn["phi"]),
    })
    out_csv = plots_dir / "comparison_1d_data.csv"
    df_out.to_csv(_long(out_csv), index=False, float_format="%.10e")
    print(f"[OK] Saved: {out_csv}")

    # Print R² summary
    print()
    print("R^2 vs DQ:")
    for k in keys:
        print(f"  {k:<5}: FEM = {r2(dq[k], fem_at_dq[k]):.6f}   "
              f"PS-PINN (with TL) = {r2(dq[k], pinn_at_dq[k]):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
