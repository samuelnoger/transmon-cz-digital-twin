"""Digital-Twin calibration of elementary 1Q DRAG and CZ primitives for RB."""

import numpy as np
from src.analysis.metrics import COMP_IDX, cz_comp_block, cz_process_fidelity
from src.control.pulses import generate_analytic_pulse
from src.physics.propagators import propagate_parallel


def _modulate_iq(I_env, Q_drag, w, t_drag, frame):
    """Modulate rotating-frame (I_env, Q_drag) onto carrier w with virtual-Z angle `frame`."""
    arg = w * t_drag + frame    # ADD frame, don't subtract it!
    I = I_env * np.cos(arg) + Q_drag * np.sin(arg)
    Q = -I_env * np.sin(arg) + Q_drag * np.cos(arg)
    return I, Q


def _extract_rx90_phases_1q(U_comp, driven_qubit):
    """Closed-form (za1, za2, zb1, zb2) aligning U_comp with Rx(pi/2) on `driven_qubit`."""
    if driven_qubit == 1:
        # Q1 driven (0 <-> 2 and 1 <-> 3), Q2 idle
        za1 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[0, 2]) + U_comp[1, 1] / (1j * U_comp[1, 3]))))
        zb1 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[2, 0]) + U_comp[1, 1] / (1j * U_comp[3, 1]))))
        za2 = 0.0
        zb2 = float(np.angle(0.5 * (U_comp[0, 0] / U_comp[1, 1] + U_comp[2, 2] / U_comp[3, 3])))
    else:
        # Q1 idle, Q2 driven (0 <-> 1 and 2 <-> 3)
        za1 = 0.0
        zb1 = float(np.angle(0.5 * (U_comp[0, 0] / U_comp[2, 2] + U_comp[1, 1] / U_comp[3, 3])))
        za2 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[0, 1]) + U_comp[2, 2] / (1j * U_comp[2, 3]))))
        zb2 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[1, 0]) + U_comp[2, 2] / (1j * U_comp[3, 2]))))
    return np.array([za1, za2, zb1, zb2], dtype=float)


def _extract_rx90_phases_both(U_comp):
    """Closed-form (za1, za2, zb1, zb2) aligning U_comp with Rx(pi/2) x Rx(pi/2)."""
    za1 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[0, 2]) + U_comp[1, 1] / (1j * U_comp[1, 3]))))
    zb1 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[2, 0]) + U_comp[1, 1] / (1j * U_comp[3, 1]))))
    za2 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[0, 1]) + U_comp[2, 2] / (1j * U_comp[2, 3]))))
    zb2 = float(np.angle(0.5 * (U_comp[0, 0] / (1j * U_comp[1, 0]) + U_comp[2, 2] / (1j * U_comp[3, 2]))))
    return np.array([za1, za2, zb1, zb2], dtype=float)


def calibrate_rb_primitives(
    config,
    H0_cal,
    Hx1,
    Hy1,
    Hx2,
    Hy2,
    Hz_control,
    cal_a,
    cal_d,
    cal_pulse_rad,
    drag_duration_ns=20.0,
):
    """Calibrate 1Q DRAG amplitudes and Virtual-Z frame shifts on the Digital Twin (H0_cal)."""
    dt = config.pulse.dt_ns
    I_env_raw, Q_drag_raw = generate_analytic_pulse(
        duration_ns=drag_duration_ns,
        dt_ns=dt,
        anharmonicity_mhz=cal_a,
        pulse_type="drag",
        target_angle=0.5 * np.pi,
    )
    I_env_raw = np.asarray(I_env_raw, dtype=float)
    Q_drag_raw = np.asarray(Q_drag_raw, dtype=float)
    n_drag = len(I_env_raw)
    t_drag = np.arange(n_drag) * dt
    zeros = np.zeros(n_drag, dtype=float)

    # Dressed single-qubit frequencies from H0_cal
    evals, evecs = np.linalg.eigh(np.asarray(H0_cal))

    def _dom_energy(bare_idx):
        overlaps = np.abs(evecs[bare_idx, :]) ** 2
        return float(evals[int(np.argmax(overlaps))])

    w00 = _dom_energy(0)
    w01 = _dom_energy(1) - w00
    w10 = _dom_energy(3) - w00

    # Calibrate fine amplitude scale factors on H0_cal so Rx(pi/2) excites P(|1>) = 0.500000
    I1_init, Q1_init = _modulate_iq(I_env_raw, Q_drag_raw, w10, t_drag, 0.0)
    U1_init = np.asarray(
        propagate_parallel(I1_init, Q1_init, zeros, zeros, dt, H0_cal, Hx1, Hy1, Hx2, Hy2)
    )[np.ix_(COMP_IDX, COMP_IDX)]
    theta1 = 2.0 * np.arctan2(abs(U1_init[2, 0]), abs(U1_init[0, 0]))
    scale1 = (0.5 * np.pi) / theta1

    I2_init, Q2_init = _modulate_iq(I_env_raw, Q_drag_raw, w01, t_drag, 0.0)
    U2_init = np.asarray(
        propagate_parallel(zeros, zeros, I2_init, Q2_init, dt, H0_cal, Hx1, Hy1, Hx2, Hy2)
    )[np.ix_(COMP_IDX, COMP_IDX)]
    theta2 = 2.0 * np.arctan2(abs(U2_init[1, 0]), abs(U2_init[0, 0]))
    scale2 = (0.5 * np.pi) / theta2

    I_env1, Q_drag1 = I_env_raw * scale1, Q_drag_raw * scale1
    I_env2, Q_drag2 = I_env_raw * scale2, Q_drag_raw * scale2

    # Propagate the amplitude-calibrated elementary slots (Q1-only, Q2-only, Both)
    I1_0, Q1_0 = _modulate_iq(I_env1, Q_drag1, w10, t_drag, 0.0)
    I2_0, Q2_0 = _modulate_iq(I_env2, Q_drag2, w01, t_drag, 0.0)

    U_rx1_0 = np.asarray(
        propagate_parallel(I1_0, Q1_0, zeros, zeros, dt, H0_cal, Hx1, Hy1, Hx2, Hy2)
    )[np.ix_(COMP_IDX, COMP_IDX)]
    U_rx2_0 = np.asarray(
        propagate_parallel(zeros, zeros, I2_0, Q2_0, dt, H0_cal, Hx1, Hy1, Hx2, Hy2)
    )[np.ix_(COMP_IDX, COMP_IDX)]
    U_both_0 = np.asarray(
        propagate_parallel(I1_0, Q1_0, I2_0, Q2_0, dt, H0_cal, Hx1, Hy1, Hx2, Hy2)
    )[np.ix_(COMP_IDX, COMP_IDX)]

    p_rx1 = _extract_rx90_phases_1q(U_rx1_0, driven_qubit=1)
    p_rx2 = _extract_rx90_phases_1q(U_rx2_0, driven_qubit=2)
    p_both = _extract_rx90_phases_both(U_both_0)

    # Calibrate CZ Virtual-Z phases on H0_cal:
    # U_cz_comp ~ CZ @ (Rz(p1) x Rz(p2)), so Virtual-Z(-p1, -p2) cancels the local Z phases.
    U_cz_comp = cz_comp_block(cal_pulse_rad, dt, H0_cal, Hz_control)
    cz_cal = cz_process_fidelity(U_cz_comp)
    cz_vz1 = -float(cz_cal["p1"])
    cz_vz2 = -float(cz_cal["p2"])

    return {
        "dt": dt,
        "I_env": I_env1,
        "Q_drag": Q_drag1,
        "I_env1": I_env1,
        "Q_drag1": Q_drag1,
        "I_env2": I_env2,
        "Q_drag2": Q_drag2,
        "t_drag": t_drag,
        "w10": w10,
        "w01": w01,
        "p_rx1": p_rx1,
        "p_rx2": p_rx2,
        "p_both": p_both,
        "cz_pulse": np.asarray(cal_pulse_rad, dtype=float),
        "cz_vz1": cz_vz1,
        "cz_vz2": cz_vz2,
    }