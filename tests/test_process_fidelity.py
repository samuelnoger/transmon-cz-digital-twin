"""Tests for cz_process_fidelity (paste the function into src/analysis/metrics.py)."""
import numpy as np
import pytest

from src.analysis.metrics import cz_process_fidelity
from src.analysis.metrics import bell_from_cz_block

CZ = np.diag([1, 1, 1, -1]).astype(complex)


def local_z(p1, p2):
    return np.diag([1, np.exp(1j * p2), np.exp(1j * p1), np.exp(1j * (p1 + p2))])


def test_ideal_cz_with_local_z_is_perfect():
    r = cz_process_fidelity(CZ @ local_z(0.7, -1.9))
    assert r["F_avg"] == pytest.approx(1.0, abs=1e-8)
    assert r["leakage"] == pytest.approx(0.0, abs=1e-12)


def test_identity_has_no_entangling_power():
    r = cz_process_fidelity(np.eye(4, dtype=complex))
    assert r["F_pro"] == pytest.approx(0.5, abs=1e-6)
    assert r["F_avg"] == pytest.approx(0.6, abs=1e-6)


def test_pure_leakage_scales_fidelity():
    r = cz_process_fidelity(np.sqrt(0.98) * CZ)
    assert r["leakage"] == pytest.approx(0.02, abs=1e-9)
    assert r["F_avg"] == pytest.approx(0.98, abs=1e-8)

def test_bell_from_cz_block():
    assert bell_from_cz_block(CZ) == pytest.approx(1.0, abs=1e-6)
    d = 0.25
    assert bell_from_cz_block(np.diag([1, 1, 1, -np.exp(1j * d)])) == pytest.approx(1 - d**2 / 16, abs=1e-3)


def _swap(e):
    S = np.eye(4, dtype=complex)
    S[1, 1] = S[2, 2] = np.cos(e)
    S[1, 2] = S[2, 1] = -1j * np.sin(e)
    return S


@pytest.mark.parametrize("U", [
    CZ @ _swap(0.2),                                                         # residual |01>-|10> swap
    0.99 * CZ @ np.diag([1, 1, 1, np.exp(0.4j)]) @ _swap(0.2),               # leaky + phase error + swap
])
def test_average_fidelity_matches_haar_average(U):
    """Independent check of the formula: average |<psi|V^dagger U|psi>|^2 over random states."""
    r = cz_process_fidelity(U)
    V = CZ @ local_z(r["p1"], r["p2"])
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(40000, 4)) + 1j * rng.normal(size=(40000, 4))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    M = V.conj().T @ U
    mc = np.mean(np.abs(np.einsum("ni,ij,nj->n", psi.conj(), M, psi)) ** 2)
    assert mc == pytest.approx(r["F_avg"], abs=6e-3)
