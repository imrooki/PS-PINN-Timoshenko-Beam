
from __future__ import annotations

from typing import Optional,  Union, Callable

import torch

def d_dx(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=False,
        only_inputs=True
    )[0]

def compute_derivatives(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    grad = y
    for i in range(order):
        need_graph = (i < order - 1)
        grad = torch.autograd.grad(
            grad, x,
            grad_outputs=torch.ones_like(grad),
            create_graph=need_graph,
            retain_graph=False,
            only_inputs=True
        )[0]
    return grad

def safe_divide(numerator: torch.Tensor, denominator: Union[float, torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    import warnings
    
    if torch.isnan(numerator).any():
        raise ValueError("Numerator contains NaN values")
    if torch.isinf(numerator).any():
        warnings.warn("Numerator contains Inf values, results may be unstable")
    
    if isinstance(denominator, float):
        if abs(denominator) < eps:
            warnings.warn(f"Small denominator detected: {denominator}, using eps protection")
        denom = max(abs(denominator), eps)
        result = numerator / denom
    else:
        if torch.isnan(denominator).any():
            raise ValueError("Denominator contains NaN values")
        if torch.isinf(denominator).any():
            warnings.warn("Denominator contains Inf values, results may be unstable")
            
        safe_denom = torch.clamp(torch.abs(denominator), min=eps) * torch.sign(denominator)
        zero_mask = torch.abs(denominator) < eps
        safe_denom = torch.where(zero_mask, eps, safe_denom)
        result = numerator / safe_denom
    
    if torch.isnan(result).any():
        raise RuntimeError("Safe divide produced NaN results - numerical instability detected")
    if torch.isinf(result).any():
        warnings.warn("Safe divide produced Inf results - potential numerical issues")
    
    return result

def mean_integral(density: torch.Tensor, factor: float = 1.0, weights: Optional[torch.Tensor] = None) -> torch.Tensor:

    if weights is None:
        return torch.mean(density) * factor
    if density.dim() == 2 and density.size(1) == 1:
        density = density.view(-1)
    w = weights.view(-1).to(dtype=density.dtype, device=density.device)
    return torch.sum(w * density.view(-1))

__all__ = ["d_dx", "compute_derivatives", "safe_divide", "mean_integral"]

def as_shape(
    value: Union[float, int, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    like: torch.Tensor,
) -> torch.Tensor:

    if callable(value):
        out = value(like)
        return out.to(dtype=like.dtype, device=like.device)

    if isinstance(value, torch.Tensor):
        out = value.to(dtype=like.dtype, device=like.device)
        return out + torch.zeros_like(like)

    return torch.full_like(like, float(value))

__all__.append("as_shape")

def quad_nodes_weights(method: str, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    if n <= 0:
        raise ValueError("n must be positive")
    m = (method or "").lower()
    import numpy as np

    if m in ("gauss", "gauss_legendre", "legendre"):
        t, w = np.polynomial.legendre.leggauss(int(n))
        x = 0.5 * (t + 1.0)
        w = 0.5 * w
        x_t = torch.tensor(x, device=device, dtype=dtype).reshape(-1, 1)
        w_t = torch.tensor(w, device=device, dtype=dtype).reshape(-1)
        return x_t.requires_grad_(True), w_t

    if m in ("clenshaw", "clenshaw_curtis", "cc"):
        if n == 1:
            x = np.array([0.5], dtype=np.float64)
            w = np.array([1.0], dtype=np.float64)
        else:
            N = n - 1
            k = np.arange(0, n, dtype=np.float64)
            theta = np.pi * k / N
            x_std = np.cos(theta)
            w = np.ones(n, dtype=np.float64)
            jmax = N // 2
            j = np.arange(1, jmax + 1, dtype=np.float64)
            coeff = 2.0 / (4.0 * j * j - 1.0)
            cos_term = np.cos(np.outer(2.0 * j, theta))
            w -= (coeff[:, None] * cos_term).sum(axis=0)
            if N % 2 == 0 and N > 0:
                jN = N // 2
                w -= (1.0 / (4.0 * jN * jN - 1.0)) * np.cos(N * theta)
            w *= 2.0 / N
            x = 0.5 * (x_std + 1.0)
            w *= 0.5
        x_t = torch.tensor(x, device=device, dtype=dtype).reshape(-1, 1)
        w_t = torch.tensor(w, device=device, dtype=dtype).reshape(-1)
        return x_t.requires_grad_(True), w_t

    raise ValueError(f"Unknown quadrature method: {method}")

__all__.extend(["quad_nodes_weights"])

def sample_1d(N: int, device: torch.device, *, sampler: str = "uniform", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    s = (sampler or "uniform").lower()
    if s == "uniform":
        x = torch.rand((N, 1), device=device, dtype=dtype)
        return x.requires_grad_(True)
    elif s == "sobol":
        try:
            from torch.quasirandom import SobolEngine
        except Exception as e:
            raise ImportError("Sobol sampler requires torch.quasirandom.SobolEngine") from e
        seed = int(torch.initial_seed() & 0xFFFFFFFF)
        engine = SobolEngine(dimension=1, scramble=True, seed=seed)
        X = engine.draw(N)
        x = X.to(device=device, dtype=dtype)
        return x.requires_grad_(True)
    elif s == "lhs":
        try:
            from pyDOE2 import lhs as _lhs
            X = _lhs(1, samples=int(N), criterion="maximin", iterations=10)
        except Exception:
            try:
                from pyDOE import lhs as _lhs
                X = _lhs(1, samples=int(N), criterion="maximin", iterations=10)
            except Exception:
                import numpy as np

                def _lhs_1d_local(samples: int, iterations: int = 10) -> np.ndarray:
                    samples = int(samples)
                    if samples <= 0:
                        return np.zeros((0, 1), dtype=np.float64)
                    best = None
                    best_score = -1.0
                    bin_width = 1.0 / samples
                    starts = np.linspace(0.0, 1.0 - bin_width, samples)
                    rng = np.random
                    for _ in range(max(1, int(iterations))):
                        u = rng.rand(samples)
                        x = starts + u * bin_width
                        rng.shuffle(x)
                        xs = np.sort(x)
                        diffs = np.diff(np.concatenate(([0.0], xs, [1.0])))
                        score = float(diffs.min())
                        if score > best_score:
                            best_score = score
                            best = x.copy()
                    return best.reshape(-1, 1)

                X = _lhs_1d_local(int(N), iterations=10)
        x = torch.tensor(X, device=device, dtype=dtype)
        return x.requires_grad_(True)
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Choose from 'uniform', 'lhs', 'sobol'.")

__all__.append("sample_1d")
