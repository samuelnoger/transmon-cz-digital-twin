# Superconducting Two-Qubit CZ Gate: Calibrate-Then-Control Pipeline

A simulation and optimization project for superconducting transmon qutrits. The pipeline characterizes an unknown device with a vectorized swap-chevron fitter, designs a leakage-suppressed, bandwidth-limited CZ gate on the fitted digital twin, and uses that gate in an end-to-end Bell state ($\Phi^+$) circuit and CHSH test. It then benchmarks the gate under unmodeled hardware defects (parasitic $ZZ$ crosstalk and cryogenic flux-line filtering), characterizes it via Quantum Process Tomography (QPT) and Interleaved Randomized Benchmarking (IRB), and closes the reality gap on the simulated hardware using closed-loop Nelder-Mead calibration.

A JAX digital twin is used for fitting and open-loop pulse design. Everything is checked against a QuTiP `HiddenDevice`, a simulated physical fridge with its own "true" parameters, $T_1/T_2$ decoherence, projective-measurement shot noise, and optional unmodeled physics ($ZZ$ crosstalk and RC low-pass flux filtering). All "device" results below are from this simulator, not real hardware.

## Methods

**1. Digital twin and calibration fitter**
- **Vectorized swap-chevron fitter:** fits exchange coupling ($g$), detuning ($\Delta$), and anharmonicity ($\alpha$) simultaneously from $\vert{}10\rangle$ and $\vert{}11\rangle$ probes.
- **JAX-jitted propagators:** piecewise-constant matrix exponentials (`expm`) per time step, optimized with Optax (`Adam`).

**2. Bandwidth-limited CZ optimization**
- **Fourier-basis pulse shaping:** smooth 40 ns flux pulses built from 8 sine harmonics and filtered with a Gaussian kernel to respect AWG bandwidth.
- **Leakage suppression:** the loss penalizes population in non-computational states (e.g. $\vert{}20\rangle$).

**3. Verification**
- **Held-out cross-validation:** the calibrated twin is tested on 100 random waveforms it was not fitted on.
- **Conditional Ramsey:** phase fringes on the QuTiP device verify the $\pi$ conditional phase of the CZ.

**4. Bell state synthesis, virtual-Z calibration, and CHSH**
- **Full circuit in the 9D two-qutrit space:** parallel 20 ns DRAG $\pi/2$ pulses (with detuning modulation on Q1), the 40 ns flux CZ, and a final $-\pi/2$ rotation on Q2.
- **Calibrate once, then freeze:** virtual-Z phases are found on the twin with a precomputed 2D grid search plus Nelder-Mead, maximizing $\Phi^+$ overlap to absorb always-on coupling and local phase accumulation.
- **Hardware-style execution:** the Q2 phase is folded into the phase of the final drive (`build_bell_waveforms`), the whole timeline is packed with `pack_sequence` and played with `HiddenDevice.play_sequence`, and the Q1 phase is applied as a tomography-frame rotation (`apply_frame`).
- **CHSH inequality:** correlator measurements across four Pauli basis settings verify entanglement against the classical bound ($S \le 2$).

**5. Unmodeled physics & reality gap analysis**
- **Parasitic static $ZZ$ crosstalk:** sweeps unmodeled $H_{ZZ} = 2\pi \zeta_{ZZ} (n_1 \otimes n_2)$ from $0$ to $2.5\text{ MHz}$ to isolate coherent phase errors.
- **Cryogenic flux-line filtering:** sweeps causal first-order RC low-pass filter 3 dB bandwidths from $500\text{ MHz}$ down to $50\text{ MHz}$ to isolate non-adiabatic trajectory distortion and leakage.

**6. Process Tomography (QPT) and Interleaved Randomized Benchmarking (IRB)**
- **Quantum Process Tomography:** reconstructs the $4 \times 4$ computational Choi matrix across 16 preparation and Pauli measurement settings (with and without $T_1/T_2$ decoherence and shot noise) and validates against the noiseless hardware oracle.
- **Interleaved RB:** benchmarks 1Q DRAG Clifford layers ($p_{\text{1Q}}$) against interleaved 1Q + CZ sequences ($p_{\text{IRB}}$) up to $m = 384$ Cliffords to extract the isolated CZ gate fidelity $p_{\text{CZ}} = p_{\text{IRB}} / p_{\text{1Q}}$.

**7. Closed-loop hardware calibration**
- **2D pulse scaling:** parameterizes the twin's waveform via amplitude and duration/symmetry scale factors and optimizes directly on the perturbed QuTiP device using a 9-setting hardware cost estimator ($\delta^2/16 + (1 - P_{11})/4$) and Nelder-Mead.
- **Multi-seed calibration race:** compares twin-seeded initialization (Arm A, including the 70-experiment chevron overhead) against an uncalibrated nominal guess (Arm B) across 10 independent shot-noise seeds (4,000 shots/setting) toward a target gate fidelity of $F_{\text{avg}} \ge 99.5\%$.

## Parameter assumption

The twin uses the **fitted** parameters from the calibration notebook (`cal_params.npy`). Under the perturbed hardware config (`configs/two_qubit_CZ_zz.yaml`, with $1.5\text{ MHz}$ unmodeled $ZZ$ crosstalk and $400\text{ MHz}$ flux bandwidth), the fitter absorbs part of the unmodeled shift into effective parameters (fit errors: $\delta g = 0.138\text{ MHz}$, $\delta \Delta = 0.132\text{ MHz}$, $\delta \alpha = 0.185\text{ MHz}$ at 4,000 shots). The `HiddenDevice` uses the **true** config parameters plus $T_1/T_2$ decoherence, static $ZZ$, and flux-line filtering.

## Results

**Calibration and CZ (`01_calibrate_twin.ipynb`)**
- **Parameter recovery:** fitted $g = 25.762\text{ MHz}$ (true $25.900\text{ MHz}$), $\Delta = 592.632\text{ MHz}$ (true $592.500\text{ MHz}$), and $\alpha = -305.385\text{ MHz}$ (true $-305.200\text{ MHz}$).
- **Generalization:** on 100 held-out random waveforms, calibration reduces mean population prediction error from $\sim 9.3\%$ (uncalibrated) to $< 0.1\%$.
- **Open-loop CZ performance:** achieves $99.89\%$ twin fidelity ($0.09\%$ leakage) on $H_{0,\text{cal}}$, and $F_{\text{avg}} = 99.23\%$ ($F_{\text{pro}} = 99.12\%$, $0.30\%$ leakage) on the true unitary hardware oracle with cryogenic bandwidth filtering (vs. $79.72\%$ for the uncalibrated pulse).

**Bell state and CHSH (`03_bell_state.ipynb`, `04_chsh.ipynb`)**

| Simulator | Parameters | Unmodeled Physics & Noise | $\Phi^+$ fidelity | Leakage |
|---|---|---|---|---|
| JAX twin | fitted (`zz1p5_bw400`) | none | 98.32% | 0.48% |
| QuTiP device | true (`zz1p5_bw400`) | $T_1/T_2$ + 1.5 MHz $ZZ$ + 400 MHz BW | **96.14%** | 0.56% |
| QuTiP device | true (`ideal`) | $T_1/T_2$ only | **98.10%** | 0.52% |

- Fidelity is $\langle\Phi^+\vert{}\rho_{\text{comp}}\vert{}\Phi^+\rangle$ on the unnormalized $4 \times 4$ computational block, so leakage counts directly as loss.
- **CHSH violation:** on the perturbed fridge with calibrated Virtual-Z phases ($c_1 = -0.96\text{ rad}$, $c_2 = 0.76\text{ rad}$), the Bell state achieves $S = 2.711 \pm 0.021$ at 5,000 shots—a $34.4\sigma$ violation of the classical bound ($S = 2.0$, vs. ideal quantum maximum $2.828$), dropping to $S = 0.956 \pm 0.024$ when the tomography frame is detuned by $+\pi/2$.

**Unmodeled physics sweeps (`05_unmodeled_physics.ipynb`)**
- **Static $ZZ$ crosstalk ($0 \to 2.5\text{ MHz}$):** primarily induces coherent phase errors during the sequence, degrading Bell fidelity while leaving qutrit leakage nearly flat ($\sim 0.5\%$).
- **Flux-line bandwidth ($500 \to 50\text{ MHz}$):** slurs the rising/falling edges of the 40 ns flux pulse, disrupting the adiabatic $\vert{}11\rangle \leftrightarrow \vert{}20\rangle$ trajectory and driving computational leakage up to $\sim 1.8\%$.

**QPT and Randomized Benchmarking (`06_benchmarks.ipynb`)**
- **Unitary & QPT hierarchy:** Twin unitary ($F_{\text{avg}} = 99.895\%$, $F_{\text{pro}} = 99.890\%$) $\to$ Unfiltered $H_{0,\text{true}}$ oracle ($F_{\text{avg}} = 99.453\%$, $F_{\text{pro}} = 99.336\%$) $\to$ Noiseless filtered QPT ($F_{\text{avg}} = 99.240\%$, $F_{\text{pro}} = 99.127\%$) $\to$ Open-system $T_1/T_2$ QPT ($F_{\text{avg}} = 99.091\%$, $F_{\text{pro}} = 98.943\%$, decoherence penalty $\Delta F_{\text{avg}} = 0.149\%$).
- **Interleaved RB:** 1Q Reference RB yields $p_{\text{1Q}} = 0.98554 \pm 0.00085$ ($F_{\text{avg}} = 98.915\%$), and Interleaved CZ RB yields $p_{\text{IRB}} = 0.95662 \pm 0.00405$ ($F_{\text{avg}} = 96.747\%$), isolating an open-system CZ gate fidelity of $F_{\text{avg}}(\text{CZ, RB}) = 97.800\%$ ($p_{\text{CZ}} = 0.97066$).

**Closed-loop calibration (`07_closed_loop.ipynb`)**
- **Twin-seeded (Arm A):** starts at $F_{\text{avg}} = 99.23\%$ on the perturbed fridge and reaches the $F_{\text{avg}} \ge 99.5\%$ target in **10/10 seeds** with a median of **124 cumulative hardware experiments** (70 chevron + 54 closed-loop), polishing to a final median $F_{\text{avg}} = 99.60\%$.
- **Uncalibrated guess (Arm B):** starts at $F_{\text{avg}} = 89.62\%$ ($3.15\%$ leakage) and reaches the $99.5\%$ target in **0/10 seeds** within the evaluation budget (stalling at a median $F_{\text{avg}} = 99.33\%$).

## Notebooks

Run in this order (downstream notebooks load `cal_params.npy`, `cal_pulse_rad.npy`, and `bell_run.npz` from `artifacts/<config_name>/`):

1. **`calibrate_twin.ipynb`:** swap-chevron calibration, CZ training, cross-validation, and conditional Ramsey checks.
2. **`design_cz.ipynb`:** CZ optimization and visualization of the $\vert{}11\rangle \leftrightarrow \vert{}20\rangle$ avoided crossing.
3. **`bell_state.ipynb`:** virtual-Z calibration, density-matrix cityscape, and Bell-state execution.
4. **`chsh.ipynb`:** CHSH entanglement verification across calibrated and detuned frames.
5. **`unmodeled_physics.ipynb`:** isolated sensitivity sweeps for parasitic $ZZ$ crosstalk and finite flux-line bandwidth.
6. **`benchmarks.ipynb`:** Quantum Process Tomography (QPT), shot-noise convergence, and Interleaved Randomized Benchmarking (IRB).
7. **`closed_loop.ipynb`:** closed-loop hardware cost validation and multi-seed Nelder-Mead calibration race.

## Setup

Python 3.11, JAX (with `jax_enable_x64` enabled in the notebooks), Optax, QuTiP, NumPy, SciPy, Matplotlib, Plotly.

## Limitations

- Three levels per transmon, first-order RC filter model for flux lines, and no readout assignment error beyond projective multinomial shot noise.
- The twin is noiseless and the device is a QuTiP Lindblad simulator, so these results show that the calibrate-then-control + closed-loop polishing workflow is sound in simulation, not that it would reach the exact same fidelity on a physical cryostat without hardware-specific predistortion.
- Short gate sequences (40 ns CZ, 80 ns Bell circuit) relative to $T_1 = 50\ \mu\text{s}$ and $T_2 = 30\ \mu\text{s}$, so coherent unmodeled errors dominate over decoherence in single-gate metrics.

## Project Structure

```text
configs/                Hardware and optimizer YAML configurations (ideal & perturbed)
artifacts/              Saved calibration parameters, pulses, and state runs
notebooks/              End-to-end experimental pipeline (Notebooks 01 through 07)
src/
  ├── physics/          Two-qutrit Hamiltonians and JAX matrix-exponential propagators
  ├── devices/          QuTiP HiddenDevice simulator (Lindblad, ZZ, RC flux filter)
  ├── control/          DRAG pulses, Virtual-Z frames, and Bell/RB waveform compilers
  ├── calibration/      Swap-chevron fitter, CZ Optax loops, and closed-loop Nelder-Mead
  ├── experiments/      Ramsey, CHSH, Process Tomography (QPT), and Interleaved RB
  └── analysis/         Process/Bell fidelity metrics, RB decay fitting, and plotting

```

## Acknowledgements

Built as a personal learning project on superconducting-qubit simulation and optimal control, with AI coding assistants used for code structuring and debugging.