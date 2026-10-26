"""Two-qubit quantum process tomography (QPT) on the computational subspace."""

import numpy as np
import qutip as qt
from scipy.optimize import minimize

from src.analysis.metrics import COMP_IDX


# --- Single-qubit input states in the 3-level qutrit basis: |0>, |1>, |+>, |+i> ---
def _input_kets_3d():
    k0 = qt.basis(3, 0)
    k1 = qt.basis(3, 1)
    kp = (k0 + k1).unit()
    ki = (k0 + 1j * k1).unit()
    return [k0, k1, kp, ki]


# --- Single-qubit 2x2 density matrices for the same 4 states ---
def _input_dms_2d():
    k0 = np.array([1.0, 0.0], dtype=complex)
    k1 = np.array([0.0, 1.0], dtype=complex)
    kp = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    ki = np.array([1.0, 1j], dtype=complex) / np.sqrt(2.0)
    kets = [k0, k1, kp, ki]
    return [np.outer(k, k.conj()) for k in kets]


# --- Single-qubit readout basis rotations (3x3 qutrit operators, acting on {|0>, |1>}) ---
def _readout_rotations_3d():
    """Return 3x3 unitaries that rotate Z, X, Y eigenstates into the Z measurement basis."""
    # Z basis: Identity
    Uz = qt.qeye(3)

    # X basis: Ry(-pi/2) on {|0>, |1>}, maps |+> -> |0>, |-> -> |1>
    c, s = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
    Ux_mat = np.eye(3, dtype=complex)
    Ux_mat[:2, :2] = [[c, s], [-s, c]]
    Ux = qt.Qobj(Ux_mat)

    # Y basis: Rx(pi/2) on {|0>, |1>}, maps |+i> -> |0>, |-i> -> |1>
    Uy_mat = np.eye(3, dtype=complex)
    Uy_mat[:2, :2] = [[c, -1j * s], [-1j * s, c]]
    Uy = qt.Qobj(Uy_mat)

    return {"X": Ux, "Y": Uy, "Z": Uz}


# --- Two-qubit Pauli matrices (4x4) ---
_PAULIS_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_LABELS = ["I", "X", "Y", "Z"]
_BASIS_LABELS = ["X", "Y", "Z"]


def reconstruct_state_from_9_bases(probs_9x4):
    """Reconstruct a 4x4 computational-subspace density matrix from 9 Pauli-basis readouts.

    Parameters
    ----------
    probs_9x4 : array of shape (3, 3, 4)
        probs_9x4[i, j] holds the 4 computational probabilities [p00, p01, p10, p11]
        measured in basis (_BASIS_LABELS[i], _BASIS_LABELS[j]).
        Note: The sum p00 + p01 + p10 + p11 equals 1 - leakage (not normalized to 1).
    """
    exp_vals = {}
    # <I x I> is the total population in the computational subspace; average over all 9 settings
    exp_vals[("I", "I")] = float(np.mean(np.sum(probs_9x4, axis=-1)))

    # Single-qubit marginals <B1 x I> and <I x B2>: average over the 3 settings of the other qubit
    for i, b1 in enumerate(_BASIS_LABELS):
        # p00 + p01 - p10 - p11 averaged over j in {X, Y, Z}
        p = probs_9x4[i, :, :]
        exp_vals[(b1, "I")] = float(np.mean((p[:, 0] + p[:, 1]) - (p[:, 2] + p[:, 3])))

    for j, b2 in enumerate(_BASIS_LABELS):
        # p00 + p10 - p01 - p11 averaged over i in {X, Y, Z}
        p = probs_9x4[:, j, :]
        exp_vals[("I", b2)] = float(np.mean((p[:, 0] + p[:, 2]) - (p[:, 1] + p[:, 3])))

    # Two-qubit correlators <B1 x B2>
    for i, b1 in enumerate(_BASIS_LABELS):
        for j, b2 in enumerate(_BASIS_LABELS):
            p00, p01, p10, p11 = probs_9x4[i, j]
            exp_vals[(b1, b2)] = float((p00 + p11) - (p01 + p10))

    # Linear inversion: rho = (1/4) * sum_{a,b} <P_a x P_b> (P_a x P_b)
    rho_rec = np.zeros((4, 4), dtype=complex)
    for a in _PAULI_LABELS:
        for b in _PAULI_LABELS:
            P_ab = np.kron(_PAULIS_1Q[a], _PAULIS_1Q[b])
            rho_rec += exp_vals[(a, b)] * P_ab
    return 0.25 * rho_rec


def measure_9_bases(rho_9x9, n_shots=None, rng=None):
    """Apply the 9 two-qubit readout rotations to a 9x9 state and read out [p00, p01, p10, p11]."""
    rots = _readout_rotations_3d()
    probs_9x4 = np.zeros((3, 3, 4), dtype=float)

    for i, b1 in enumerate(_BASIS_LABELS):
        for j, b2 in enumerate(_BASIS_LABELS):
            U_rot = qt.tensor(rots[b1], rots[b2])
            rho_rot = U_rot * rho_9x9 * U_rot.dag()

            # Extract all 9 qutrit level probabilities so leakage is properly accounted for
            exact_probs = np.clip(rho_rot.diag().real, 0.0, 1.0)
            exact_probs /= np.sum(exact_probs)

            if n_shots is None:
                probs_9x4[i, j] = exact_probs[COMP_IDX]
            else:
                if rng is None:
                    counts = np.random.multinomial(n_shots, exact_probs)
                else:
                    counts = rng.multinomial(n_shots, exact_probs)
                probs_9x4[i, j] = (counts / float(n_shots))[COMP_IDX]

    return probs_9x4


def reconstruct_choi_from_io(input_dms_4x4, output_dms_4x4):
    """Reconstruct the 16x16 Choi matrix Lambda = sum_{i,j} |i><j| x E(|i><j|) from 16 I/O pairs."""
    # Each input_dm is 4x4; stack their flattened row vectors into a 16x16 matrix M
    M = np.array([rho_in.reshape(16) for rho_in in input_dms_4x4], dtype=complex)  # (16, 16)
    Y = np.array([rho_out.reshape(16) for rho_out in output_dms_4x4], dtype=complex)  # (16, 16)

    # Solve M @ T = Y, where row (4*i + j) of T is E(|i><j|).reshape(16)
    T = np.linalg.solve(M, Y)  # (16, 16)

    # Assemble the 16x16 Choi matrix: block (i, j) is E(|i><j|)
    choi = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            E_ij = T[4 * i + j].reshape(4, 4)
            choi[4 * i : 4 * (i + 1), 4 * j : 4 * (j + 1)] = E_ij
    return choi


def run_process_tomography(device, pulse_rad, n_shots=None, seed=42):
    """Run 16-input x 9-basis process tomography of a flux pulse on HiddenDevice.

    Returns
    -------
    choi : ndarray of shape (16, 16)
        Reconstructed Choi matrix on the 4D computational subspace.
    raw_probs : ndarray of shape (16, 3, 3, 4)
        All 144 measured probability vectors [p00, p01, p10, p11].
    """
    rng = np.random.default_rng(seed) if n_shots is not None else None
    kets_3d = _input_kets_3d()
    dms_2d = _input_dms_2d()

    input_dms_4x4 = []
    output_dms_4x4 = []
    raw_probs = np.zeros((16, 3, 3, 4), dtype=float)

    idx = 0
    for a in range(4):
        for b in range(4):
            rho_in_3x3 = qt.ket2dm(qt.tensor(kets_3d[a], kets_3d[b]))
            rho_in_4x4 = np.kron(dms_2d[a], dms_2d[b])

            # 1 pulse simulation per input state (16 total)
            rho_out_9x9 = device.play_pulse(pulse_rad, rho_in_3x3)

            # 9 basis readouts on the output state
            probs_9x4 = measure_9_bases(rho_out_9x9, n_shots=n_shots, rng=rng)
            rho_out_4x4 = reconstruct_state_from_9_bases(probs_9x4)

            input_dms_4x4.append(rho_in_4x4)
            output_dms_4x4.append(rho_out_4x4)
            raw_probs[idx] = probs_9x4
            idx += 1

    choi = reconstruct_choi_from_io(input_dms_4x4, output_dms_4x4)
    return choi, raw_probs