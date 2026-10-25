"""Baseline regression tests for the digital-twin and QuTiP device pipeline."""

import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import qutip as qt

from src.config import ExperimentConfig
from src.physics.operators import (
    build_two_qubit_operators,
    build_two_qubit_base,
    assemble_H0,
    build_true_H0,
)
from src.physics.propagators import propagate_flux
from src.devices.qutip_device import HiddenDevice
from src.control.waveforms import (
    build_local_drive_operators,
    build_bell_waveforms,
    generate_bell_state,
    pack_sequence,
)
from src.control.virtual_z import apply_frame
from src.calibration.virtual_z_cal import calibrate_virtual_z
from src.analysis.metrics import bell_fid, leakage

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"
BASELINE_PATH = TESTS_DIR / "baseline.json"


def test_twin_and_device_expose_matching_flux_execution():
    """Fast consistency check on a short zero-flux pulse."""
    device = HiddenDevice(t1_ns=0, t2_ns=0)
    H0, hz = build_two_qubit_operators(
        levels=3,
        anharm_1=-305.0,
        anharm_2=-305.0,
        coupling_mhz=21.5,
        idle_detuning_mhz=595.0,
    )
    pulse = np.zeros(4)
    twin_unitary = np.asarray(propagate_flux(pulse, device.dt, H0, hz))

    initial = qt.ket2dm(qt.tensor(qt.basis(3, 1), qt.basis(3, 0)))
    observed = device.play_pulse(pulse, initial)

    expected = twin_unitary @ np.asarray(initial.full()) @ twin_unitary.conj().T
    np.testing.assert_allclose(observed.full(), expected, atol=2e-5)


def test_twin_and_noiseless_device_agree_on_full_bell_sequence():
    """Verify JAX twin and noiseless QuTiP device agree on the full Bell circuit."""
    baseline = json.loads(BASELINE_PATH.read_text())
    config = ExperimentConfig.load(str(PROJECT_ROOT / "configs" / "two_qubit_CZ.yaml"))
    dt = config.pulse.dt_ns
    sq_gate_ns = baseline["single_qubit_gate_ns"]

    cal_g, cal_d, cal_a = np.load(DATA_DIR / "cal_params.npy")
    cal_pulse_rad = np.load(DATA_DIR / "cal_pulse_rad.npy")
    assert np.isclose(len(cal_pulse_rad) * dt, baseline["cz_duration_ns"])

    H_anharm, H_int_base, Hz_control = build_two_qubit_base(levels=config.hardware.levels)
    Hx1, Hy1, Hx2, Hy2 = build_local_drive_operators(levels=config.hardware.levels)
    H0_cal = assemble_H0(H_anharm, H_int_base, Hz_control, cal_g, cal_d, cal_a)

    c1, c2 = -0.93, 0.77

    # JAX Digital Twin 4x4 computational density matrix
    rho_twin_4x4 = np.asarray(
        generate_bell_state(
            sq_gate_ns, dt, cal_a, cal_d, cal_pulse_rad,
            H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control,
            v_phase1=c1, v_phase2=c2,
        )
    )

    # Noiseless QuTiP HiddenDevice with identical Hamiltonian parameters
    noiseless_device = HiddenDevice(
        levels=config.hardware.levels,
        true_coupling=float(cal_g),
        true_detuning=float(cal_d),
        true_anharm=float(cal_a),
        t1_ns=0.0,
        t2_ns=0.0,
    )
    rho00 = qt.ket2dm(qt.tensor(qt.basis(3, 0), qt.basis(3, 0)))
    wf = build_bell_waveforms(sq_gate_ns, dt, cal_a, cal_d, cal_pulse_rad, v_phase2=c2)
    rho_dev_9x9 = apply_frame(noiseless_device.play_sequence(pack_sequence(wf), rho00), c1, c2).full()

    idx_comp = np.array([0, 1, 3, 4])
    rho_dev_4x4 = rho_dev_9x9[np.ix_(idx_comp, idx_comp)]

    # Elementwise agreement and state overlap check
    np.testing.assert_allclose(rho_dev_4x4, rho_twin_4x4, atol=1e-3)
    overlap = float(np.real(np.trace(rho_twin_4x4.conj().T @ rho_dev_4x4)) / np.real(np.trace(rho_twin_4x4 @ rho_twin_4x4)))
    assert overlap >= baseline["noiseless_overlap_min"]


def test_bell_state_baseline_fidelities():
    """Reproduce the frozen 98.29% twin and 98.11% noisy device Bell fidelities."""
    baseline = json.loads(BASELINE_PATH.read_text())
    config = ExperimentConfig.load(str(PROJECT_ROOT / "configs" / "two_qubit_CZ.yaml"))
    dt = config.pulse.dt_ns
    sq_gate_ns = baseline["single_qubit_gate_ns"]

    cal_g, cal_d, cal_a = np.load(DATA_DIR / "cal_params.npy")
    cal_pulse_rad = np.load(DATA_DIR / "cal_pulse_rad.npy")

    H_anharm, H_int_base, Hz_control = build_two_qubit_base(levels=config.hardware.levels)
    Hx1, Hy1, Hx2, Hy2 = build_local_drive_operators(levels=config.hardware.levels)
    H0_cal = assemble_H0(H_anharm, H_int_base, Hz_control, cal_g, cal_d, cal_a)

    # Calibrate Virtual-Z phases on the Digital Twin
    vz_cal = calibrate_virtual_z(
        sq_gate_ns, dt, cal_a, cal_d, cal_pulse_rad,
        H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control,
    )
    c1, c2 = vz_cal["v_phase1"], vz_cal["v_phase2"]

    # Evaluate Digital Twin Bell fidelity
    rho_corr = generate_bell_state(
        sq_gate_ns, dt, cal_a, cal_d, cal_pulse_rad,
        H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control,
        v_phase1=c1, v_phase2=c2,
    )
    fidelity_twin = float(bell_fid(rho_corr))

    # Evaluate noisy QuTiP HiddenDevice Bell fidelity and leakage
    fridge = HiddenDevice(
        levels=config.hardware.levels,
        true_coupling=config.hardware.coupling_mhz,
        true_detuning=config.hardware.idle_detuning_mhz,
        true_anharm=config.hardware.anharmonicity_mhz,
        t1_ns=config.decoherence.t1_ns,
        t2_ns=config.decoherence.t2_ns,
    )
    rho00 = qt.ket2dm(qt.tensor(qt.basis(3, 0), qt.basis(3, 0)))
    wf = build_bell_waveforms(sq_gate_ns, dt, cal_a, cal_d, cal_pulse_rad, v_phase2=c2)
    rho_f = fridge.play_sequence(pack_sequence(wf), rho00)
    rho_framed = apply_frame(rho_f, c1, c2)

    fidelity_device = float(bell_fid(rho_framed.full()))
    leakage_device = float(leakage(rho_f.full()))

    np.testing.assert_allclose(
        fidelity_twin, baseline["twin_bell_fidelity"], atol=baseline["fidelity_atol"]
    )
    np.testing.assert_allclose(
        fidelity_device, baseline["device_bell_fidelity"], atol=baseline["fidelity_atol"]
    )
    np.testing.assert_allclose(
        leakage_device, baseline["device_leakage"], atol=baseline["leakage_atol"]
    )

def test_oracle_hamiltonian_matches_device():
    config = ExperimentConfig.load("configs/two_qubit_CZ_zz.yaml")
    H0_true, Hz = build_true_H0(config)
    device = HiddenDevice.from_config(config)
    np.testing.assert_allclose(np.asarray(H0_true), device.H0.full(), atol=1e-10)
    np.testing.assert_allclose(np.asarray(Hz), device.Hz_control.full(), atol=1e-10)