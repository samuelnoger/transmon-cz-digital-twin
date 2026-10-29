import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt

from src.control.pulses import apply_low_pass_filter
from src.physics.propagators import propagate_flux


def reconstruct_pulse_rad(opt_params, basis, kernel, f_max):
    """Reconstructs the filtered flux pulse in rad/ns from Fourier coefficients."""
    raw = basis @ opt_params['flux_coeffs']
    flux = f_max * jnp.tanh(raw / f_max)
    return np.array(apply_low_pass_filter(flux, kernel))


def evaluate_held_out_probes(
    fridge,
    H0_naive,
    H0_cal,
    Hz_control,
    dt,
    basis,
    kernel,
    f_max,
    n_test_pulses=50,
    seed=999,
):
    """
    Cross-validates the uncalibrated and calibrated JAX twins against the QuTiP
    HiddenDevice across random pulses from the same filtered design family used
    during calibration. The HiddenDevice then applies its hidden hardware
    response to that commanded pulse.
    """
    @jax.jit
    def predict_populations(H0_model, pulse_rad):
        U = propagate_flux(pulse_rad, dt, H0_model, Hz_control)
        idx = jnp.array([0, 1, 3, 4])
        pops_10 = jnp.abs(U[idx, 3]) ** 2
        pops_11 = jnp.abs(U[idx, 4]) ** 2
        return pops_10, pops_11

    rho_10 = qt.ket2dm(qt.tensor(qt.basis(3, 1), qt.basis(3, 0)))
    rho_11 = qt.ket2dm(qt.tensor(qt.basis(3, 1), qt.basis(3, 1)))

    rng = np.random.default_rng(seed)
    naive_errors = []
    cal_errors = []

    for _ in range(n_test_pulses):
        rand_coeffs = rng.uniform(-2.5, -1.2, size=(basis.shape[1],))
        raw_flux = basis @ rand_coeffs
        test_pulse_rad = np.asarray(
            apply_low_pass_filter(f_max * jnp.tanh(raw_flux / f_max), kernel)
        )

        true_p10 = fridge.measure_populations(test_pulse_rad, rho_10, n_shots=None)
        true_p11 = fridge.measure_populations(test_pulse_rad, rho_11, n_shots=None)

        naive_p10, naive_p11 = predict_populations(H0_naive, jnp.array(test_pulse_rad))
        cal_p10, cal_p11 = predict_populations(H0_cal, jnp.array(test_pulse_rad))

        naive_err = 0.5 * (np.mean(np.abs(naive_p10 - true_p10)) + np.mean(np.abs(naive_p11 - true_p11)))
        cal_err = 0.5 * (np.mean(np.abs(cal_p10 - true_p10)) + np.mean(np.abs(cal_p11 - true_p11)))

        naive_errors.append(float(naive_err) * 100.0)
        cal_errors.append(float(cal_err) * 100.0)

    return np.array(naive_errors), np.array(cal_errors)


def run_cz_ramsey_experiment(fridge, flux_pulse_rad, opt_phases, n_phase_pts=41, n_shots=1000, seed=42):
    """
    Runs a Conditional Ramsey experiment on the QuTiP HiddenDevice to measure
    the conditional phase shift of the CZ gate on Qubit 2 when Qubit 1 is in |0> vs |1>.
    """
    phases_sweep = np.linspace(0, 2 * np.pi, n_phase_pts)
    p2_virtual = float(opt_phases[1])

    probs_q1_in_0 = []
    probs_q1_in_1 = []
    rng = np.random.default_rng(seed)

    ket_plus = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    rho_0_plus = qt.ket2dm(qt.tensor(qt.basis(3, 0), ket_plus))
    rho_1_plus = qt.ket2dm(qt.tensor(qt.basis(3, 1), ket_plus))

    rho_out_0 = fridge.play_pulse(flux_pulse_rad, rho_0_plus)
    rho_out_1 = fridge.play_pulse(flux_pulse_rad, rho_1_plus)

    for phi in phases_sweep:
        ket_proj = (qt.basis(3, 0) + np.exp(1j * (phi + p2_virtual)) * qt.basis(3, 1)).unit()
        P_0 = qt.ket2dm(qt.tensor(qt.basis(3, 0), ket_proj))
        P_1 = qt.ket2dm(qt.tensor(qt.basis(3, 1), ket_proj))

        p0_exact = np.clip(qt.expect(P_0, rho_out_0), 0.0, 1.0)
        p1_exact = np.clip(qt.expect(P_1, rho_out_1), 0.0, 1.0)

        if n_shots is None:
            probs_q1_in_0.append(p0_exact)
            probs_q1_in_1.append(p1_exact)
        else:
            probs_q1_in_0.append(rng.binomial(n_shots, p0_exact) / float(n_shots))
            probs_q1_in_1.append(rng.binomial(n_shots, p1_exact) / float(n_shots))

    return phases_sweep, np.array(probs_q1_in_0), np.array(probs_q1_in_1)
