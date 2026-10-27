"""Waveform compiler for 1Q Clifford and CZ Randomized Benchmarking sequences."""

import numpy as np
from src.calibration.rb_calibration import _modulate_iq
from src.experiments.rb_cliffords import CLIFFORDS_1Q


def compile_rb_waveform(primitives, ops):
    """Compile a sequence of ('1Q', c1, c2) and ('CZ',) operations into device waveforms.

    Returns
    -------
    wf : dict with keys 'I1', 'Q1', 'I2', 'Q2', 'flux'
    frame1, frame2 : final Virtual-Z angles in the `virtual_z.py` convention
        `Z = diag(1, exp(-1j * frame))`, so `apply_frame(rho, frame1, frame2)`
        or `np.diag(np.exp(-1j * frame * np.arange(2)))` rotates back to the logical frame.
    """
    I_env1 = primitives.get("I_env1", primitives["I_env"])
    Q_drag1 = primitives.get("Q_drag1", primitives["Q_drag"])
    I_env2 = primitives.get("I_env2", primitives["I_env"])
    Q_drag2 = primitives.get("Q_drag2", primitives["Q_drag"])
    t_drag = primitives["t_drag"]
    w10, w01 = primitives["w10"], primitives["w01"]

    za1_rx1, za2_rx1, zb1_rx1, zb2_rx1 = primitives["p_rx1"]
    za1_rx2, za2_rx2, zb1_rx2, zb2_rx2 = primitives["p_rx2"]
    za1_both, za2_both, zb1_both, zb2_both = primitives["p_both"]

    n_drag = len(t_drag)
    zeros_drag = np.zeros(n_drag, dtype=float)

    I1_chunks, Q1_chunks = [], []
    I2_chunks, Q2_chunks = [], []
    flux_chunks = []

    # Internal frame tracks Rz(theta) = diag(e^{-i*theta/2}, e^{+i*theta/2}) ~ diag(1, e^{+i*theta})
    frame1 = 0.0
    frame2 = 0.0

    def _append_rx_slot(do_q1, do_q2):
        nonlocal frame1, frame2
        if not do_q1 and not do_q2:
            return

        if do_q1 and not do_q2:
            frame1 += za1_rx1
            frame2 += za2_rx1
            I1, Q1 = _modulate_iq(I_env1, Q_drag1, w10, t_drag, frame1)
            I2, Q2 = zeros_drag, zeros_drag
            frame1 += zb1_rx1
            frame2 += zb2_rx1
        elif do_q2 and not do_q1:
            frame1 += za1_rx2
            frame2 += za2_rx2
            I1, Q1 = zeros_drag, zeros_drag
            I2, Q2 = _modulate_iq(I_env2, Q_drag2, w01, t_drag, frame2)
            frame1 += zb1_rx2
            frame2 += zb2_rx2
        else:
            frame1 += za1_both
            frame2 += za2_both
            I1, Q1 = _modulate_iq(I_env1, Q_drag1, w10, t_drag, frame1)
            I2, Q2 = _modulate_iq(I_env2, Q_drag2, w01, t_drag, frame2)
            frame1 += zb1_both
            frame2 += zb2_both

        I1_chunks.append(I1)
        Q1_chunks.append(Q1)
        I2_chunks.append(I2)
        Q2_chunks.append(Q2)
        flux_chunks.append(zeros_drag)

    for op in ops:
        if op[0] == "1Q":
            _, c1_idx, c2_idx = op
            z1_a, p1_a, z2_a, p2_a, z3_a = CLIFFORDS_1Q[c1_idx]["decomp"]
            z1_b, p1_b, z2_b, p2_b, z3_b = CLIFFORDS_1Q[c2_idx]["decomp"]

            frame1 += z1_a
            frame2 += z1_b
            _append_rx_slot(bool(p1_a), bool(p1_b))

            frame1 += z2_a
            frame2 += z2_b
            _append_rx_slot(bool(p2_a), bool(p2_b))

            frame1 += z3_a
            frame2 += z3_b

        elif op[0] == "CZ":
            cz_pulse = primitives["cz_pulse"]
            m_cz = len(cz_pulse)
            z_cz = np.zeros(m_cz, dtype=float)
            I1_chunks.append(z_cz)
            Q1_chunks.append(z_cz)
            I2_chunks.append(z_cz)
            Q2_chunks.append(z_cz)
            flux_chunks.append(cz_pulse)
            frame1 += primitives["cz_vz1"]
            frame2 += primitives["cz_vz2"]
        else:
            raise ValueError(f"Unknown op: {op}")

    if not flux_chunks:
        z_min = np.zeros(2, dtype=float)
        wf = {"I1": z_min, "Q1": z_min, "I2": z_min, "Q2": z_min, "flux": z_min}
    else:
        wf = {
            "I1": np.concatenate(I1_chunks),
            "Q1": np.concatenate(Q1_chunks),
            "I2": np.concatenate(I2_chunks),
            "Q2": np.concatenate(Q2_chunks),
            "flux": np.concatenate(flux_chunks),
        }

    # Wrap to [-pi, pi) and return in the virtual_z.py convention diag(1, exp(-1j * frame))
    f1_out = float((frame1 + np.pi) % (2.0 * np.pi) - np.pi)
    f2_out = float((frame2 + np.pi) % (2.0 * np.pi) - np.pi)
    return wf, f1_out, f2_out