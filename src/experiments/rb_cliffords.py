"""Single-qubit Clifford group utilities for randomized benchmarking."""

import numpy as np

def _rz_2x2(theta):
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex
    )


def _rx_2x2(theta):
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_2x2(theta):
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _canonical_u2(U):
    """Strip global phase so U in SU(2) has a unique canonical sign."""
    U = np.asarray(U, dtype=complex)
    det = np.linalg.det(U)
    U = U / np.sqrt(det)
    # Fix sign (+/-) by first non-zero element
    for x in U.ravel():
        if abs(x) > 1e-6:
            if x.real < -1e-6 or (abs(x.real) <= 1e-6 and x.imag < -1e-6):
                U = -U
            break
    return U


def build_1q_clifford_table():
    """Build all 24 single-qubit Cliffords decomposed into (z1, p1, z2, p2, z3).

    Each Clifford is represented as:
        U = Rz(z3) @ P(p2) @ Rz(z2) @ P(p1) @ Rz(z1)
    where p1, p2 in {0, 1} indicate either Identity (0) or Rx(pi/2) (1),
    and z1, z2, z3 are Virtual-Z angles in {0, pi/2, pi, 3*pi/2}.
    """
    angles = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    Rx90 = _rx_2x2(0.5 * np.pi)
    I2 = np.eye(2, dtype=complex)

    cliffords = []
    # Prefer decompositions with minimal number of physical Rx(pi/2) pulses
    candidates = []
    for p1 in [0, 1]:
        for p2 in [0, 1]:
            for z1 in angles:
                for z2 in angles:
                    for z3 in angles:
                        if p1 == 0 and z1 != 0.0:
                            continue
                        if p2 == 0 and z2 != 0.0:
                            continue
                        M1 = Rx90 if p1 else I2
                        M2 = Rx90 if p2 else I2
                        U = _rz_2x2(z3) @ M2 @ _rz_2x2(z2) @ M1 @ _rz_2x2(z1)
                        U_can = _canonical_u2(U)
                        n_pulses = p1 + p2
                        candidates.append((n_pulses, (z1, p1, z2, p2, z3), U_can))

    candidates.sort(key=lambda item: item[0])
    for _, decomp, U_can in candidates:
        if not any(
            np.allclose(U_can, c["U"], atol=1e-5) or np.allclose(-U_can, c["U"], atol=1e-5)
            for c in cliffords
        ):
            cliffords.append({"decomp": decomp, "U": U_can})
        if len(cliffords) == 24:
            break

    assert len(cliffords) == 24, f"Expected 24 1Q Cliffords, found {len(cliffords)}"
    return cliffords


CLIFFORDS_1Q = build_1q_clifford_table()


def find_1q_clifford_index(U_target):
    """Find which of the 24 single-qubit Cliffords matches U_target up to global phase."""
    U_t = _canonical_u2(U_target)
    for idx, c in enumerate(CLIFFORDS_1Q):
        if abs(np.trace(U_t.conj().T @ c["U"])) > 1.999:
            return idx
    raise ValueError("Matrix is not in the 1Q Clifford group.")


_PAULI_1Q_INDICES = [
    find_1q_clifford_index(matrix)
    for matrix in [
        np.eye(2),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
]


_DIHEDRAL_1Q_INDICES = [
    idx
    for idx, clifford in enumerate(CLIFFORDS_1Q)
    if abs(clifford["U"][0, 0]) > 0.99 or abs(clifford["U"][0, 1]) > 0.99
]
