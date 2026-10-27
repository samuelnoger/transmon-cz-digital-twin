"""Randomized-benchmarking decay analysis."""

import numpy as np
from scipy.optimize import curve_fit

def fit_rb_decay(lengths, mean_surv, d=4, mean_leak=None):
    """Fit P(|00> | m) = A * p^m + B. If mean_leak is provided, fit B dynamically."""
    m_arr = np.asarray(lengths, dtype=float)
    y_arr = np.asarray(mean_surv, dtype=float)

    def model(m, A, p, B):
        return A * (p ** m) + B

    # Use the empirical asymptote from the longest sequence if available
    B_guess = y_arr[-1] if len(y_arr) > 0 else 0.25
    y_shifted = np.clip(y_arr - B_guess, 1e-4, 1.0)
    
    # Estimate initial decay p
    if len(m_arr) >= 2:
        slope = np.polyfit(m_arr[:3], np.log(y_shifted[:3]), 1)[0]
        p_init = float(np.clip(np.exp(slope), 0.80, 0.9999))
    else:
        p_init = 0.95

    # If we have severe leakage (asymptote is being pulled up), allow B to float freely
    try:
        popt, pcov = curve_fit(
            model,
            m_arr,
            y_arr,
            p0=[1.0 - B_guess, p_init, B_guess],
            bounds=([0.0, 0.5, 0.0], [1.0, 1.0, 1.0]),
            maxfev=10000,
        )
        A, p, B = [float(v) for v in popt]
        p_std = float(np.sqrt(np.diag(pcov))[1])
    except Exception:
        # Fallback to fixed empirical asymptote
        popt, pcov = curve_fit(
            lambda m, A, p: A * (p ** m) + B_guess,
            m_arr,
            y_arr,
            p0=[1.0 - B_guess, p_init],
            bounds=([0.0, 0.5], [1.0, 1.0]),
        )
        A, p = float(popt[0]), float(popt[1])
        B = B_guess
        p_std = float(np.sqrt(np.diag(pcov))[1])

    F_avg = ((d - 1.0) * p + 1.0) / d
    F_std = ((d - 1.0) / d) * p_std
    return {
        "A": A,
        "p": p,
        "B": B,
        "p_std": p_std,
        "F_avg": F_avg,
        "F_std": F_std,
        "epc": 1.0 - F_avg,
    }