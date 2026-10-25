"""Common state and gate quality metrics."""

import numpy as np


def bell_fid(rho, target=None):
    """Return the fidelity with |Phi+> on the computational subspace."""
    if target is None:
        target = np.zeros((4,), dtype=complex)
        target[[0, 3]] = 1.0 / np.sqrt(2.0)
    state = np.asarray(rho)
    if state.shape == (9, 9):
        state = state[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
    return float(np.real(np.conj(target) @ state @ target))


def fidelity(rho, target):
    """Return pure-state overlap for a density matrix or state vector."""
    state = np.asarray(rho)
    target = np.asarray(target)
    if state.ndim == 1:
        return float(abs(np.vdot(target, state)) ** 2)
    return float(np.real(np.conj(target) @ state @ target))


def leakage(rho, computational_indices=(0, 1, 3, 4)):
    """Return population outside the computational subspace."""
    state = np.asarray(rho)
    if state.ndim == 1:
        population = np.sum(np.abs(state[list(computational_indices)]) ** 2)
    else:
        population = np.real(np.trace(state[np.ix_(computational_indices, computational_indices)]))
    return float(1.0 - population)
