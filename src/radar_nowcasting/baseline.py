"""
Eulerian persistence baseline.

The future rainfall field is assumed to be identical to the last observed
frame at every grid point. It is simple and physically interpretable, yet a
surprisingly strong reference in short-range nowcasting - any learned model
has to beat it to demonstrate genuine skill.
"""

from __future__ import annotations

import numpy as np


def persistence_forecast(input_sequence: np.ndarray, nsteps: int = 3) -> np.ndarray:
    """Repeat the last input frame ``nsteps`` times.

    Parameters
    ----------
    input_sequence : (T_in, H, W, C)
    nsteps : forecast horizon ``T_out``

    Returns
    -------
    (nsteps, H, W, C)
    """
    last_frame = input_sequence[-1]
    return np.repeat(last_frame[np.newaxis, ...], nsteps, axis=0)


def persistence_predictions(X: np.ndarray, nsteps: int = 3) -> np.ndarray:
    """Apply :func:`persistence_forecast` to every sample of ``X`` ``(N, T_in, H, W, C)``."""
    return np.array([persistence_forecast(X[i], nsteps=nsteps) for i in range(len(X))])
