"""Sequence generators for 1Q Reference RB and CZ Interleaved RB."""

import numpy as np
from src.experiments.rb_cliffords import CLIFFORDS_1Q, _rx_2x2, _rz_2x2, find_1q_clifford_index

# Precompute all 576 two-qubit product Cliffords (C1 x C1) for instant state-recovery lookup
_U_1Q_PAIRS = np.array(
    [
        np.kron(CLIFFORDS_1Q[i]["U"], CLIFFORDS_1Q[j]["U"])
        for i in range(24)
        for j in range(24)
    ],
    dtype=complex,
)

# |+> = Ry(pi/2)|0>, and CZ(|+> x |+>) is the canonical 2Q entangled stabilizer state
_KET_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
_PHI_BELL = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
# Single-qubit Clifford that maps |+> -> |0>, i.e. Ry(-pi/2) = Rz(-pi/2) Rx(pi/2) Rz(pi/2)
_PLUS_TO_ZERO_IDX = find_1q_clifford_index(
    _rz_2x2(-0.5 * np.pi) @ _rx_2x2(0.5 * np.pi) @ _rz_2x2(0.5 * np.pi)
)


def generate_1q_rb_sequence(m, rng):
    """Generate a simultaneous 1Q Clifford RB sequence of length m + 1 recovery gate."""
    if m == 0:
        return [("1Q", 0, 0)]

    ops = []
    U1 = np.eye(2, dtype=complex)
    U2 = np.eye(2, dtype=complex)
    for _ in range(m):
        c1 = int(rng.integers(0, 24))
        c2 = int(rng.integers(0, 24))
        ops.append(("1Q", c1, c2))
        U1 = CLIFFORDS_1Q[c1]["U"] @ U1
        U2 = CLIFFORDS_1Q[c2]["U"] @ U2

    inv1 = find_1q_clifford_index(U1.conj().T)
    inv2 = find_1q_clifford_index(U2.conj().T)
    ops.append(("1Q", inv1, inv2))
    return ops


def generate_cz_rb_sequence(m, rng):
    """Generate an interleaved CZ RB sequence of m (1Q Clifford + CZ) layers + exact recovery."""
    if m == 0:
        return [("1Q", 0, 0)]

    CZ_4x4 = np.diag([1.0, 1.0, 1.0, -1.0])
    ops = []
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    for _ in range(m):
        c1 = int(rng.integers(0, 24))
        c2 = int(rng.integers(0, 24))
        ops.append(("1Q", c1, c2))
        ops.append(("CZ",))
        U_1q = np.kron(CLIFFORDS_1Q[c1]["U"], CLIFFORDS_1Q[c2]["U"])
        psi = CZ_4x4 @ (U_1q @ psi)

    # Check if psi is a product stabilizer state (entanglement = 0)
    # or maximally entangled stabilizer state (entanglement = 1 ebit)
    candidates = _U_1Q_PAIRS @ psi  # shape (576, 4)

    # Case 1: Product state -> 1 single-qubit Clifford layer returns psi to |00>
    prod_hits = np.where(np.abs(candidates[:, 0]) > 0.999)[0]
    if len(prod_hits) > 0:
        k = int(prod_hits[0])
        ops.append(("1Q", k // 24, k % 24))
        return ops

    # Case 2: Maximally entangled state -> map to CZ|++>, apply 1 CZ, and map |++> to |00>
    bell_overlaps = np.abs(candidates @ _PHI_BELL.conj())
    bell_hits = np.where(bell_overlaps > 0.999)[0]
    assert len(bell_hits) > 0, "Failed to find stabilizer recovery gate"
    k = int(bell_hits[0])
    ops.append(("1Q", k // 24, k % 24))
    ops.append(("CZ",))
    ops.append(("1Q", _PLUS_TO_ZERO_IDX, _PLUS_TO_ZERO_IDX))
    return ops