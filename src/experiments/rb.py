"""Two-qubit and interleaved CZ randomized benchmarking orchestration."""

import numpy as np
import qutip as qt
from joblib import Parallel, delayed

from src.analysis.metrics import COMP_IDX
from src.analysis.rb_metrics import fit_rb_decay
from src.calibration.rb_calibration import calibrate_rb_primitives
from src.control.rb_waveforms import compile_rb_waveform
from src.experiments.rb_sequences import (
    generate_1q_rb_sequence,
    generate_cz_rb_sequence,
)
from src.experiments.rb_cliffords import (
    CLIFFORDS_1Q,
    find_1q_clifford_index,
)


def _eval_single_sequence(device, primitives, ops, rho00):
    """Worker helper to compile and simulate one RB sequence."""
    waveform, _, _ = compile_rb_waveform(primitives, ops)
    rho_out = device.play_sequence(waveform, rho00)
    probabilities = np.real(rho_out.diag())
    surv = float(probabilities[0])
    leak = float(1.0 - np.sum(probabilities[COMP_IDX]))
    return surv, leak


def run_rb_experiment(
    device,
    primitives,
    lengths=(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128),
    n_seeds=8,
    mode="cz",
    seed=1234,
    fit=True,
    n_jobs=-1,
):
    """Run randomized benchmarking and fit the survival-probability decay."""
    rng = np.random.default_rng(seed)
    rho00 = qt.ket2dm(qt.tensor(qt.basis(3, 0), qt.basis(3, 0)))

    lengths = list(lengths)

    # Pre-generate all sequences deterministically on the main thread
    tasks = []
    for length_index, sequence_length in enumerate(lengths):
        for seed_index in range(n_seeds):
            if mode == "1q":
                ops = generate_1q_rb_sequence(sequence_length, rng)
            elif mode == "cz":
                ops = generate_cz_rb_sequence(sequence_length, rng)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            tasks.append((length_index, seed_index, ops))

    # Execute QuTiP simulations in parallel across CPU cores
    outputs = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_eval_single_sequence)(device, primitives, ops, rho00)
        for _, _, ops in tasks
    )

    survival = np.zeros((len(lengths), n_seeds), dtype=float)
    leakage_values = np.zeros((len(lengths), n_seeds), dtype=float)
    for (length_index, seed_index, _), (surv, leak) in zip(tasks, outputs):
        survival[length_index, seed_index] = surv
        leakage_values[length_index, seed_index] = leak

    mean_survival = np.mean(survival, axis=1)
    std_survival = np.std(survival, axis=1)
    mean_leakage = np.mean(leakage_values, axis=1)

    result = {
        "lengths": np.array(lengths),
        "surv_matrix": survival,
        "mean_surv": mean_survival,
        "std_surv": std_survival,
        "mean_leak": mean_leakage,
    }
    if fit:
        fit_res = fit_rb_decay(lengths, mean_survival, d=4, mean_leak=mean_leakage)
        result.update(fit_res)
    return result


__all__ = [
    "calibrate_rb_primitives",
    "compile_rb_waveform",
    "fit_rb_decay",
    "generate_1q_rb_sequence",
    "generate_cz_rb_sequence",
    "run_rb_experiment",
]