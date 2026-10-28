"""The corrected plug-in ("corrected code" version, POST HOC, round 4 of the reviews).

The frozen `pipeline.increments` reads the reference shape F0 at the rescaled ages s * a. Beyond the reference's
horizon H the pooled F0 is flat (mature cohorts are cut at H), so a cohort with speed s > 1, which needs F0(sH), is
estimated too low. Here F0 beyond H is the generalized gamma that the frozen code already fits for the model
selection, scaled to continue the pooled reference (`maturity.onestage.extended_reference`).
Nothing else changes: speeds, speed model, calendar step and every baseline are the frozen ones. A cohort whose
rescaled ages stay within H (s * h <= H) gets the frozen estimate exactly.
Use: `corrected.patch(pl)` replaces `pl.increments` of a loaded pipeline module (0011 or 0012).
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from maturity import onestage


def increments(
    curve: np.ndarray, a: int, speed: float, ugrid: np.ndarray, f0: np.ndarray, horizon: int
) -> np.ndarray:
    """inc[k], k = 0..horizon: observed adjusted increments up to age a, fitted ones (shape F0, speed s) after.
    The frozen function with one change: F0 is continued beyond the horizon of the reference."""
    inc = np.zeros(horizon + 1)
    a = min(a, horizon)
    inc[1 : a + 1] = np.diff(curve[: a + 1])
    f = onestage.extended_reference(ugrid, f0)(speed * np.arange(a, horizon + 1))
    inc[a + 1 :] = curve[a] * np.diff(f) / max(f[0], 1e-9)
    return inc


def patch(pl: ModuleType) -> None:
    pl.increments = increments


def demo() -> None:
    """Within the horizon the corrected function is the frozen one; beyond it the fitted increments are larger."""
    grid = np.linspace(0.0, 4.0 * 84, onestage.N_GRID)
    shape = 1 - np.exp(-((grid / 60.0) ** 1.5))
    f0 = np.minimum(1.0, shape / np.interp(84.0, grid, shape))
    curve = 0.05 * np.interp(np.arange(201.0), grid, f0)

    def frozen(speed: float) -> np.ndarray:
        inc = np.zeros(85)
        inc[1:25] = np.diff(curve[:25])
        f = np.interp(speed * np.arange(24, 85), grid, f0, right=1.0)
        inc[25:] = curve[24] * np.diff(f) / max(f[0], 1e-9)
        return inc

    assert np.allclose(increments(curve, 24, 1.0, grid, f0, 84), frozen(1.0))
    assert np.allclose(increments(curve, 24, 0.8, grid, f0, 84), frozen(0.8))
    assert increments(curve, 24, 1.3, grid, f0, 84).sum() > 1.02 * frozen(1.3).sum()


if __name__ == "__main__":
    demo()
