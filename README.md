# Superconducting Two-Qubit CZ Gate: Calibrate-Then-Control Pipeline

A simulation and optimization project for superconducting transmon qutrits. The pipeline characterizes an unknown device with a vectorized swap-chevron fitter, designs a leakage-suppressed, bandwidth-limited CZ gate on the fitted digital twin, and uses that gate in an end-to-end Bell state ($\Phi^+$) circuit.

A JAX digital twin is used for fitting and pulse design. Everything is then checked against a QuTiP `HiddenDevice`, a simulated device with its own "true" parameters, $T_1/T_2$ decoherence and projective-measurement shot noise. All "device" results below are from this simulator, not real hardware.

## Methods

**1. Digital twin and calibration fitter**
- **Vectorized swap-chevron fitter:** fits exchange coupling ($g$), detuning ($\Delta$) and anharmonicity ($\alpha$) simultaneously from $|10\rangle$ and $|11\rangle$ probes.
- **JAX-jitted propagators:** piecewise-constant matrix exponentials (`expm`) per time step, optimized with Optax (`Adam`).

**2. Bandwidth-limited CZ optimization**
- **Fourier-basis pulse shaping:** smooth flux pulses built from sine harmonics and filtered with a Gaussian kernel to respect AWG bandwidth.
- **Leakage suppression:** the loss penalizes population in non-computational states (e.g. $|20\rangle$).

**3. Verification**
- **Held-out cross-validation:** the calibrated twin is tested on 200 random waveforms it was not fitted on.
- **Conditional Ramsey:** phase fringes on the QuTiP device verify the $\pi$ conditional phase of the CZ.

**4. Bell state synthesis and virtual-Z calibration**
- **Full circuit in the 9D two-qutrit space:** parallel DRAG $\pi/2$ pulses (with detuning modulation on Q1), the flux CZ, and a final $-\pi/2$ rotation on Q2.
- **Calibrate once, then freeze:** virtual-Z phases are found on the twin with a precomputed 2D grid search plus Nelder-Mead, by maximizing $\Phi^+$ overlap. This absorbs always-on coupling and local phase accumulation.
- **Hardware-style execution:** the Q2 phase is folded into the phase of the final drive (`build_bell_waveforms`), the whole timeline is packed with `pack_sequence` and played with `HiddenDevice.play_sequence`, and the Q1 phase is applied as a tomography-frame rotation (`apply_frame`).

## Parameter assumption

The twin uses the **fitted** parameters from the calibration notebook (fit errors: [TODO: insert from calibration run]). The `HiddenDevice` uses the **true** parameters from the config plus $T_1/T_2$. The device comparison therefore includes both fit error and decoherence.

## Results

**Calibration and CZ**
- **Parameter recovery:** fitted $g$, $\Delta$, $\alpha$ within < 0.01 MHz, of the true values with 2,000-shot measurements.
- **Generalization:** on held-out random waveforms, calibration reduces mean population prediction error from ~9.3% (uncalibrated) to ~0.09%, close to the $T_1/T_2$ floor.
- **Calibrated vs. uncalibrated CZ:** on the device model, the calibrated pulse substantially reduces leakage and gives the intended $\pi$ conditional phase.

**Bell state** (all rows use the same drive-rotated circuit, so they are directly comparable)

| Simulator | Parameters | Noise | $\Phi^+$ fidelity |
|---|---|---|---|
| JAX twin | fitted | none | 98.28% |
| QuTiP device | fitted | none | 98.31% |
| QuTiP device | true | $T_1/T_2$ | **98.10%** |

- The fitted-parameter noiseless device matches the twin to ~3e-4 in state overlap, which confirms the two implementations are consistent.
- Going from the twin to the full device model costs about 0.2%, combining fit error and decoherence. Phases calibrated on the twin were not re-tuned on the device.
- 99.48% of the population stays in the computational subspace (0.52% leakage). The remaining infidelity comes mostly from the gate itself (conditional phase, residual ZZ during the single-qubit pulses, leakage), not from the virtual-Z calibration.
- Fidelity is $\langle\Phi^+|\rho_{\text{comp}}|\Phi^+\rangle$ on the unnormalized computational block, so leakage counts as loss.
- Sequence: two 20 ns single-qubit pulses plus a 40 ns CZ, so $T_1/T_2$ contribute a small share of the error.
- Note: a variant that applies the virtual Z as a matrix between CZ and the last pulse scores 97.53% at the same phases. This is a slightly different circuit (the identity between the two holds only approximately because of the always-on coupling).

## Notebooks

Run in this order (the Bell notebook loads `artifacts/cal_pulse_rad.npy`, which the calibration notebook saves).

1. **`01_calibrate_twin.ipynb`:** swap-chevron calibration, CZ training, cross-validation, and Ramsey checks.
2. **`02_design_cz.ipynb`:** CZ optimization and visualization of the $|11\rangle \leftrightarrow |20\rangle$ avoided crossing.
3. **`03_bell_state.ipynb`:** virtual-Z calibration, implementation checks, and Bell-state execution.
4. **`04_chsh.ipynb`:** CHSH measurement protocol.

## Setup

Python 3.11, JAX (with `jax_enable_x64` enabled in the notebooks), Optax, QuTiP, NumPy, SciPy, Matplotlib, Plotly.

## Limitations

- Three levels per transmon, no crosstalk, no flux-line distortion, no readout error beyond shot noise.
- The twin is noiseless and the device is a simulator, so these results show that the calibrate-then-control workflow is sound, not that it would reach the same fidelity on a real chip.
- Short sequence relative to $T_1/T_2$, so decoherence is a minor error source here.

## Project Structure

```text
configs/                                Hardware and optimizer configuration
artifacts/                              Saved calibration outputs (gitignored)
src/config.py                           Configuration parser
src/physics/operators.py                Hamiltonian builders
src/physics/propagators.py              Time-evolution propagators
src/devices/qutip_device.py             HiddenDevice simulator
src/control/pulses.py                   DRAG pulses and filters
src/control/waveforms.py                Bell waveforms and sequence packing
src/control/virtual_z.py                Virtual-Z matrices and frame application
src/calibration/fitters.py              Digital twin fitting
src/calibration/cz_training.py          CZ losses and training loops
src/calibration/virtual_z_cal.py        Virtual-Z calibration entry point
src/experiments/ramsey.py               Ramsey and held-out probe experiments
src/experiments/chsh.py                 CHSH experiment
src/experiments/single_qubit_check.py   QuTiP pulse verification
src/experiments/rb.py                   RB experiment orchestration
src/experiments/rb_cliffords.py         Single-qubit Clifford utilities
src/experiments/rb_sequences.py         RB sequence generation
src/control/rb_waveforms.py             RB waveform compilation
src/calibration/rb_calibration.py       RB primitive calibration
src/analysis/metrics.py                 Fidelity and leakage metrics
src/analysis/rb_metrics.py              RB decay fitting
src/analysis/plots.py                   Visualization helpers
notebooks/                              Calibration, CZ, Bell-state, and CHSH notebooks
tests/                                  Automated twin/device regression tests
```

The domain packages are the only supported import paths. The old
`src.optimization`, `src.verification`, and `src.control.sequences` modules
were removed during the reorganization.

## Acknowledgements

Built as a personal learning project on superconducting-qubit simulation and optimal control, with AI coding assistants used for code structuring and debugging.