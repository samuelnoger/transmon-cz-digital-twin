"""Common state and gate quality metrics."""

import numpy as np
import numpy as np
from scipy.optimize import minimize
from src.physics.propagators import propagate_flux


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


def cz_process_fidelity(U_comp, n_grid=72):
    """Gate fidelity of a 4x4 computational-subspace block against CZ, up to local Z phases.

    U_comp: 4x4 block of the full unitary in the basis 00, 01, 10, 11. It may be
    sub-unitary: population that left the qubit subspace counts as loss.
    Returns F_avg (average gate fidelity), F_pro (process fidelity), leakage and the
    best local Z phases (p1, p2).
    """
    U = np.asarray(U_comp, dtype=complex)
    d = 4
    diag = np.diag(U)

    def overlap(p):
        # |Tr(V^dagger U)|^2 with V = CZ * Z(p1) x Z(p2) = diag(1, e^{i p2}, e^{i p1}, -e^{i(p1+p2)})
        v = np.array([1, np.exp(1j * p[1]), np.exp(1j * p[0]), -np.exp(1j * (p[0] + p[1]))])
        return abs(np.vdot(v, diag)) ** 2

    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    start = max(((overlap((a, b)), a, b) for a in grid for b in grid))[1:]
    res = minimize(lambda p: -overlap(p), x0=start, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14})
    ov = -res.fun
    tr_uu = float(np.real(np.trace(U.conj().T @ U)))
    return {
        "F_avg": (tr_uu + ov) / (d * (d + 1)),   # Pedersen et al. 2007
        "F_pro": ov / d**2,
        "leakage": 1.0 - tr_uu / d,
        "p1": float(res.x[0]), "p2": float(res.x[1]),
    }

COMP_IDX = np.array([0, 1, 3, 4])

def cz_comp_block(pulse_rad, dt, H0, Hz_control):
    """4x4 computational-subspace block of the unitary produced by a flux pulse."""
    U = np.asarray(propagate_flux(pulse_rad, dt, H0, Hz_control))
    return U[np.ix_(COMP_IDX, COMP_IDX)]


def bell_from_cz_block(U, n=721):
    """Bell fidelity if the single-qubit gates were ideal and only this CZ block is real."""
    s = np.ones(4, complex) / 2                                   # Ry(pi/2) x Ry(pi/2) on |00>
    c, sn = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    R = np.kron(np.eye(2), np.array([[c, -sn], [sn, c]]))         # Ry(-pi/2) on Q2
    best = 0.0
    for p2 in np.linspace(-np.pi, np.pi, n):                      # virtual Z on Q2 before the last rotation
        Z = np.diag([1, np.exp(-1j * p2), 1, np.exp(-1j * p2)])
        psi = R @ (Z @ (np.asarray(U, complex) @ s))
        best = max(best, (abs(psi[0]) + abs(psi[3])) ** 2 / 2)
    return best