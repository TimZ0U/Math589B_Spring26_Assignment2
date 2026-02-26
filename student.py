"""
student.py

Math 589B Programming Assignment 2 (Autograded)

Numerical quadrature + interpolation.

Rules / constraints (per assignment announcements):
- Allowed libs: numpy, mpmath, and Python stdlib.
- Do NOT use high-level integrators like mpmath.quad or scipy.integrate.quad.
- Do NOT print from your functions.
- f will be called with scalar floats.
- Use numerically stable interpolation (barycentric strongly recommended).

This file implements:
    composite_simpson
    gauss_legendre
    romberg
    equispaced_interpolant_values
    chebyshev_lobatto_interpolant_values
    poly_integral_from_values
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import mpmath as mp


# ============================================================
# Helpers: input validation
# ============================================================

def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.ravel()


def _check_interval(a: float, b: float) -> None:
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("a and b must be finite floats.")


# ============================================================
# Quadrature
# ============================================================

def composite_simpson(f: Callable[[float], float], a: float, b: float, n_panels: int) -> float:
    """Composite Simpson's rule with n_panels panels (2 subintervals per panel)."""
    _check_interval(a, b)
    if n_panels <= 0:
        raise ValueError("n_panels must be a positive integer.")

    # Handle reversed interval cleanly
    if b < a:
        return -composite_simpson(f, b, a, n_panels)

    m = 2 * n_panels  # number of subintervals
    h = (b - a) / m

    # Nodes x0..xm
    x = a + h * np.arange(m + 1, dtype=float)
    fx = np.array([f(float(xi)) for xi in x], dtype=float)

    # Simpson:
    # integral ≈ (h/3) * [f0 + fm + 4 * sum(f_odd) + 2 * sum(f_even, excluding endpoints)]
    odd_sum = float(np.sum(fx[1:m:2]))     # indices 1,3,5,...
    even_sum = float(np.sum(fx[2:m:2]))    # indices 2,4,6,..., m-2
    return (h / 3.0) * (float(fx[0]) + float(fx[m]) + 4.0 * odd_sum + 2.0 * even_sum)


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """Gauss-Legendre quadrature with n_nodes on [a,b]."""
    _check_interval(a, b)
    if n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer.")

    if b < a:
        return -gauss_legendre(f, b, a, n_nodes)

    # Nodes/weights for [-1,1]
    t, w = np.polynomial.legendre.leggauss(n_nodes)

    # Map to [a,b]: x = (b-a)/2 * t + (a+b)/2; dx = (b-a)/2 dt
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * t

    fx = np.array([f(float(xi)) for xi in x], dtype=float)
    return float(half * np.dot(w, fx))


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Romberg integration; return R[n,n]."""
    _check_interval(a, b)
    if n < 0:
        raise ValueError("n must be >= 0.")

    if b < a:
        return -romberg(f, b, a, n)

    # Use higher precision accumulation to reduce round-off (allowed: mpmath).
    # IMPORTANT: we are NOT calling mp.quad or any high-level integrator.
    # We only use mp.mpf for arithmetic.
    mp.mp.dps = max(50, mp.mp.dps)  # be conservative; grader can override if needed

    R = [[mp.mpf("0") for _ in range(n + 1)] for __ in range(n + 1)]

    fa = mp.mpf(f(float(a)))
    fb = mp.mpf(f(float(b)))
    R[0][0] = mp.mpf(b - a) * (fa + fb) / 2

    # Romberg trapezoid refinements:
    # T_k = 1/2 T_{k-1} + h_k * sum_{i=1}^{2^{k-1}} f(a + (2i-1)h_k)
    # where h_k = (b-a)/2^k
    for k in range(1, n + 1):
        h_k = mp.mpf(b - a) / (2 ** k)
        # new midpoints (odd indices)
        s = mp.mpf("0")
        # i = 1..2^{k-1}
        for i in range(1, 2 ** (k - 1) + 1):
            x = mp.mpf(a) + (2 * i - 1) * h_k
            s += mp.mpf(f(float(x)))
        R[k][0] = mp.mpf("0.5") * R[k - 1][0] + h_k * s

        # Richardson extrapolation:
        # R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1])/(4^j - 1)
        for j in range(1, k + 1):
            factor = mp.mpf(4) ** j
            R[k][j] = R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / (factor - 1)

    return float(R[n][n])


# ============================================================
# Interpolation (Barycentric)
# ============================================================

def _barycentric_weights_generic(x_nodes: np.ndarray) -> np.ndarray:
    """Generic O(n^2) barycentric weights for distinct nodes."""
    x = np.asarray(x_nodes, dtype=float)
    n = x.size
    w = np.ones(n, dtype=float)
    # w_j = 1 / prod_{m!=j} (x_j - x_m)
    for j in range(n):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / float(np.prod(diff))
    return w


def _barycentric_weights_cheb_lobatto(n: int) -> np.ndarray:
    """Barycentric weights for Chebyshev–Lobatto nodes x_k = cos(pi*k/n), k=0..n.

    Standard stable weights:
        w_k = (-1)^k * c_k
    where c_0 = c_n = 1/2 and c_k = 1 otherwise.
    """
    if n == 0:
        return np.array([1.0], dtype=float)
    k = np.arange(n + 1, dtype=int)
    c = np.ones(n + 1, dtype=float)
    c[0] = 0.5
    c[-1] = 0.5
    w = c * ((-1.0) ** k)
    return w


def _barycentric_eval(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_eval: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate barycentric interpolant at x_eval.

    Uses first-form barycentric formula:
        p(x) = sum_j (w_j y_j / (x - x_j)) / sum_j (w_j / (x - x_j))
    with exact hit handling when x equals a node.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = _as_1d_float_array(x_eval)

    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be 1D arrays.")
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length.")
    if x_nodes.size == 0:
        raise ValueError("Need at least one node.")

    if w is None:
        w = _barycentric_weights_generic(x_nodes)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != x_nodes.shape:
            raise ValueError("Weights must have the same shape as x_nodes.")

    out = np.empty_like(x_eval, dtype=float)

    # Evaluate pointwise (n <= ~50; simple loop is fine and robust).
    tol = 1e-14
    for i, x in enumerate(x_eval):
        diff = x - x_nodes
        hit = np.where(np.abs(diff) < tol)[0]
        if hit.size:
            out[i] = float(y_nodes[hit[0]])
            continue

        tmp = w / diff
        num = float(np.sum(tmp * y_nodes))
        den = float(np.sum(tmp))
        out[i] = num / den

    return out


def equispaced_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant at equispaced nodes on [-1,1] at x_eval."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    x_nodes = np.linspace(-1.0, 1.0, n + 1, dtype=float)
    y_nodes = np.array([f(float(x)) for x in x_nodes], dtype=float)
    return _barycentric_eval(x_nodes, y_nodes, x_eval)


def chebyshev_lobatto_interpolant_values(f: Callable[[float], float], n: int, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant at Chebyshev-Lobatto nodes on [-1,1] at x_eval."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    if n == 0:
        # single node at x=1 (or -1); conventionally x_0 = cos(0)=1
        x_nodes = np.array([1.0], dtype=float)
        y_nodes = np.array([f(1.0)], dtype=float)
        return _barycentric_eval(x_nodes, y_nodes, x_eval, w=np.array([1.0], dtype=float))

    k = np.arange(n + 1, dtype=float)
    x_nodes = np.cos(np.pi * k / n)  # includes endpoints +1, -1
    y_nodes = np.array([f(float(x)) for x in x_nodes], dtype=float)
    w = _barycentric_weights_cheb_lobatto(n)
    return _barycentric_eval(x_nodes, y_nodes, x_eval, w=w)


# ============================================================
# Integral of interpolating polynomial
# ============================================================

def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """Return integral over [-1,1] of interpolating polynomial through (x_nodes, y_nodes).

    Robust approach:
    - Treat interpolant as a polynomial of degree <= m-1 where m=len(x_nodes).
    - Integrate it using Gauss–Legendre with enough nodes to be (theoretically) exact for polynomials:
        GL with M nodes is exact up to degree 2M-1.
      So choose M >= ceil((deg+1)/2).
    - Accumulate in mpmath high precision to reduce cancellation/roundoff in the sum.
    """
    x_nodes = _as_1d_float_array(x_nodes)
    y_nodes = _as_1d_float_array(y_nodes)
    if x_nodes.size != y_nodes.size:
        raise ValueError("x_nodes and y_nodes must have the same length.")
    if x_nodes.size == 0:
        raise ValueError("Need at least one node.")

    deg = int(x_nodes.size - 1)

    # Minimum GL nodes for exactness on degree deg polynomials (in exact arithmetic):
    # 2M - 1 >= deg  =>  M >= (deg+1)/2
    M = int(math.ceil((deg + 1) / 2))
    # Add a small safety margin for floating evaluation error of the interpolant
    M = max(1, M + 2)

    # GL nodes/weights on [-1,1] in double, cast to mp for summation
    t, w = np.polynomial.legendre.leggauss(M)

    # Use barycentric evaluation of the interpolant at GL nodes
    # (generic weights are fine; nodes here may be arbitrary).
    bw = _barycentric_weights_generic(x_nodes)
    p_at_t = _barycentric_eval(x_nodes, y_nodes, t, w=bw)

    mp.mp.dps = max(60, mp.mp.dps)
    s = mp.mpf("0")
    for wi, pi in zip(w, p_at_t):
        s += mp.mpf(wi) * mp.mpf(pi)

    # On [-1,1], integral = sum w_i p(t_i)
    return float(s)
