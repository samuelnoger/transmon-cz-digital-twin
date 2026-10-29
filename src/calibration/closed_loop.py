"""Closed-loop hardware calibration using only noisy device measurements."""

import numpy as np
import qutip as qt
from scipy.optimize import minimize

from src.analysis.metrics import cz_comp_block, cz_process_fidelity


def build_scaled_pulse(base_pulse_rad, amp_scale, length_scale, dt=0.1):
    """Scale a reference flux pulse in amplitude and time-stretch (duration)."""
    base = np.asarray(base_pulse_rad, dtype=float)
    n_base = len(base)
    n_new = max(10, int(round(n_base * float(length_scale))))

    t_old = np.linspace(0.0, 1.0, n_base)
    t_new = np.linspace(0.0, 1.0, n_new)
    resampled = np.interp(t_new, t_old, base)
    return float(amp_scale) * resampled


def _run_4point_cz_ramsey(fridge, pulse_rad, n_shots=2000, rng=None):
    """Run exact 4-quadrature Ramsey (0, pi/2, pi, 3*pi/2) for Q1 in |0>, |1>.
    Total settings = 8 (4 phases x 2 control states).
    """
    if rng is None:
        rng = np.random.default_rng()

    plus = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    psi_0 = qt.tensor(qt.basis(3, 0), plus)
    psi_1 = qt.tensor(qt.basis(3, 1), plus)

    rho_out_0 = fridge.play_pulse(pulse_rad, qt.ket2dm(psi_0))
    rho_out_1 = fridge.play_pulse(pulse_rad, qt.ket2dm(psi_1))

    quad_phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    p0 = np.zeros(4)
    p1 = np.zeros(4)

    for idx, phi in enumerate(quad_phases):
        ket_phi = (qt.basis(3, 0) + np.exp(1j * phi) * qt.basis(3, 1)).unit()
        P_plus_phi = qt.tensor(qt.qeye(3), ket_phi * ket_phi.dag())

        prob0 = float(np.clip(qt.expect(P_plus_phi, rho_out_0).real, 0.0, 1.0))
        prob1 = float(np.clip(qt.expect(P_plus_phi, rho_out_1).real, 0.0, 1.0))

        if n_shots is not None:
            p0[idx] = rng.binomial(n_shots, prob0) / float(n_shots)
            p1[idx] = rng.binomial(n_shots, prob1) / float(n_shots)
        else:
            p0[idx] = prob0
            p1[idx] = prob1

    return p0, p1


def measure_cz_hardware_cost(
    fridge,
    pulse_rad,
    n_shots=2000,
    rng=None,
):
    """Evaluate CZ gate infidelity estimate using exactly 9 hardware settings:
      - 8 Ramsey settings: Q2 at analysis phases (0, pi/2, pi, 3*pi/2) for Q1 in |0>, |1>
      - 1 Leakage/survival setting: start in |11> and measure 1 - P_comp (and 1 - P_11)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Leakage setting on |11> (1 setting)
    rho_11 = qt.ket2dm(qt.tensor(qt.basis(3, 1), qt.basis(3, 1)))
    pops_11 = fridge.measure_populations(pulse_rad, rho_11, n_shots=n_shots, rng=rng)
    leak_11 = float(np.clip(1.0 - np.sum(pops_11), 0.0, 1.0))
    non_11_pop = float(np.clip(1.0 - pops_11[3], 0.0, 1.0))

    # 2. Conditional phase via 4 quadratures: 0, pi/2, pi, 3*pi/2 (8 settings)
    p0, p1 = _run_4point_cz_ramsey(fridge, pulse_rad, n_shots=n_shots, rng=rng)

    c0, s0 = p0[0] - p0[2], p0[1] - p0[3]
    c1, s1 = p1[0] - p1[2], p1[1] - p1[3]
    eta_0 = float(np.arctan2(-s0, c0))
    eta_1 = float(np.arctan2(-s1, c1))

    cond_phase = float((eta_1 - eta_0) % (2.0 * np.pi))
    delta = cond_phase - np.pi  # phase error from pi

    # Fix #3: No double-counting. non_11_pop = 1 - P(|11>) already includes leak_11
    # plus any residual |20> <-> |11> incomplete swap error.
    cost = (delta ** 2) / 16.0 + 0.25 * non_11_pop

    return float(cost), {
        "leak_11": leak_11,
        "p11_surv": float(pops_11[3]),
        "cond_phase_rad": cond_phase,
        "phase_err_rad": float(delta),
        "n_experiments": 9,  # Strictly 4 phases x 2 control states + 1 leakage setting
    }


def evaluate_oracle_fidelity(pulse_rad, dt, H0_true, Hz_control, fridge=None):
    """Passive judge: exact gate fidelity on the true hardware Hamiltonian."""
    pulse_phys = fridge.filter_flux_waveform(pulse_rad) if fridge is not None else pulse_rad
    U_block = cz_comp_block(pulse_phys, dt, H0_true, Hz_control)
    return cz_process_fidelity(U_block)


def run_closed_loop_calibration(
    fridge,
    base_pulse_rad,
    H0_true,
    Hz_control,
    dt=0.1,
    init_scale=(1.0, 1.0),
    simplex_step=(0.02, 0.02),
    target_fidelity=0.996,
    max_iter=30,
    n_shots=4000,
    prior_experiments=0,
    seed=42,
):
    """Tune (amp_scale, length_scale) on `fridge` with Nelder-Mead.
    Fixes:
      1. `selected_F_avg` tracks the oracle fidelity of `best_pulse` chosen by
         the noisy measured cost, never peeking at the oracle.
      2. Re-evaluates challenger vs incumbent on a tie/new-best to suppress
         the shot-noise winner's curse.
    """
    rng = np.random.default_rng(seed)

    history = {
        "experiments": [],
        "cost": [],
        "probe_F_avg": [],
        "selected_F_avg": [],  # Oracle fidelity of the pulse chosen by measured cost
        "F_pro": [],
        "leakage": [],
        "cond_phase_rad": [],
        "params": [],
    }

    cum_experiments = int(prior_experiments)
    best_cost = np.inf
    selected_f_avg = 0.0
    best_pulse = np.asarray(base_pulse_rad, dtype=float).copy()
    best_info = {}
    experiments_to_target = None

    def objective(x):
        nonlocal cum_experiments, best_cost, selected_f_avg, best_pulse, best_info, experiments_to_target
        amp_scale, len_scale = float(x[0]), float(x[1])

        if not (0.5 <= amp_scale <= 1.5 and 0.5 <= len_scale <= 1.5):
            return 10.0

        candidate_pulse = build_scaled_pulse(base_pulse_rad, amp_scale, len_scale, dt=dt)
        cost, info = measure_cz_hardware_cost(
            fridge, candidate_pulse, n_shots=n_shots, rng=rng
        )
        cum_experiments += info["n_experiments"]

        oracle_res = evaluate_oracle_fidelity(
            candidate_pulse, dt, H0_true, Hz_control, fridge=fridge
        )
        probe_f_avg = float(oracle_res["F_avg"])

        # Fix #1 & #2: Update best_pulse ONLY using measured hardware cost.
        # If candidate beats best_cost and n_shots is finite, re-verify candidate once
        # and average the two cost readings to avoid picking a pure shot-noise outlier.
        if cost < best_cost:
            if n_shots is not None and best_cost < np.inf:
                cost_verify, info_v = measure_cz_hardware_cost(
                    fridge, candidate_pulse, n_shots=n_shots, rng=rng
                )
                cum_experiments += info_v["n_experiments"]
                cost_avg = 0.5 * (cost + cost_verify)
                if cost_avg < best_cost:
                    best_cost = cost_avg
                    best_pulse = candidate_pulse.copy()
                    best_info = {**info, **oracle_res, "amp_scale": amp_scale, "len_scale": len_scale}
                    selected_f_avg = probe_f_avg
                cost = cost_avg
            else:
                best_cost = cost
                best_pulse = candidate_pulse.copy()
                best_info = {**info, **oracle_res, "amp_scale": amp_scale, "len_scale": len_scale}
                selected_f_avg = probe_f_avg

        history["experiments"].append(cum_experiments)
        history["cost"].append(cost)
        history["probe_F_avg"].append(probe_f_avg)
        history["selected_F_avg"].append(selected_f_avg)
        history["F_pro"].append(float(oracle_res["F_pro"]))
        history["leakage"].append(float(oracle_res["leakage"]))
        history["cond_phase_rad"].append(info["cond_phase_rad"])
        history["params"].append((amp_scale, len_scale))

        # Fix #1: Trigger target ONLY when the experimenter's selected pulse >= target
        if experiments_to_target is None and selected_f_avg >= target_fidelity:
            experiments_to_target = cum_experiments

        return cost

    x0 = np.array(init_scale, dtype=float)
    da, dl = float(simplex_step[0]), float(simplex_step[1])
    simplex = np.array([
        x0,
        x0 + np.array([da, 0.0]),
        x0 + np.array([0.0, dl]),
    ])

    opt_res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "maxfev": max_iter,
            "xatol": 1e-3,
            "fatol": 1e-4,
        },
    )

    for k in ["experiments", "cost", "probe_F_avg", "selected_F_avg", "F_pro", "leakage", "cond_phase_rad"]:
        history[k] = np.array(history[k])

    return {
        "best_pulse_rad": best_pulse,
        "best_info": best_info,
        "history": history,
        "experiments_to_target": experiments_to_target,
        "total_experiments": cum_experiments,
        "opt_result": opt_res,
    }


def run_multi_seed_calibration(
    fridge,
    base_pulse_rad,
    H0_true,
    Hz_control,
    dt=0.1,
    init_scale=(1.0, 1.0),
    simplex_step=(0.02, 0.02),
    target_fidelity=0.996,
    max_iter=30,
    n_shots=4000,
    prior_experiments=0,
    seeds=range(42, 52),
):
    """Run closed-loop calibration across multiple noise seeds and compute median trajectories."""
    runs = []
    for s in seeds:
        res = run_closed_loop_calibration(
            fridge=fridge,
            base_pulse_rad=base_pulse_rad,
            H0_true=H0_true,
            Hz_control=Hz_control,
            dt=dt,
            init_scale=init_scale,
            simplex_step=simplex_step,
            target_fidelity=target_fidelity,
            max_iter=max_iter,
            n_shots=n_shots,
            prior_experiments=prior_experiments,
            seed=int(s),
        )
        runs.append(res)

    # Interpolate selected_F_avg onto a common experiment-count grid
    max_exp = max(r["history"]["experiments"][-1] for r in runs)
    min_exp = min(r["history"]["experiments"][0] for r in runs)
    exp_grid = np.arange(min_exp, max_exp + 1, 9)

    interp_f = np.zeros((len(runs), len(exp_grid)))
    for i, r in enumerate(runs):
        exps = r["history"]["experiments"]
        vals = r["history"]["selected_F_avg"]
        # Stepwise-constant interpolation (hold previous selected fidelity until next update)
        idx = np.searchsorted(exps, exp_grid, side="right") - 1
        idx = np.clip(idx, 0, len(vals) - 1)
        interp_f[i] = vals[idx]

    exps_to_target = [r["experiments_to_target"] for r in runs if r["experiments_to_target"] is not None]
    median_to_target = float(np.median(exps_to_target)) if exps_to_target else None

    return {
        "runs": runs,
        "exp_grid": exp_grid,
        "median_F_avg": np.median(interp_f, axis=0),
        "q25_F_avg": np.percentile(interp_f, 25, axis=0),
        "q75_F_avg": np.percentile(interp_f, 75, axis=0),
        "median_experiments_to_target": median_to_target,
        "success_rate": f"{len(exps_to_target)}/{len(runs)}",
    }