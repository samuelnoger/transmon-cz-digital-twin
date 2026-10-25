import jax.numpy as jnp

def build_transmon_operators(levels: int, anharmonicity_mhz: float):
    # Convert MHz to GHz (so the unit is radians/ns)
    alpha = 2 * jnp.pi * (anharmonicity_mhz / 1000.0)
    
    # Energy levels in the rotating frame: E_n = -0.5 * alpha * n * (n - 1)
    # Level 0: 0, Level 1: 0, Level 2: -alpha, Level 3: -3*alpha
    energies = [-0.5 * alpha * n * (n - 1) for n in range(levels)]
    H0 = jnp.diag(jnp.array(energies, dtype=jnp.complex64))
    
    # Truncated annihilation operator
    a = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), k=1).astype(jnp.complex64)
    
    Hx = (a + a.conj().T) / 2
    Hy = (a - a.conj().T) / (2j)
    
    return H0, Hx, Hy

def build_two_qubit_operators(levels=3, anharm_1=-300.0, anharm_2=-300.0, coupling_mhz=20.0, idle_detuning_mhz=600.0):
    b = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), 1)
    bdag = b.T
    eye = jnp.eye(levels)
    num = bdag @ b
 
    H1 = 0.5 * 2 * jnp.pi * (anharm_1 / 1000.0) * num @ (num - eye)
    H2 = 0.5 * 2 * jnp.pi * (anharm_2 / 1000.0) * num @ (num - eye)
    H_int = 2 * jnp.pi * (coupling_mhz / 1000.0) * (jnp.kron(b, bdag) + jnp.kron(bdag, b))
    Hz_control = jnp.kron(num, eye)                            # shifts qubit 1; the flux value is in rad/ns
 
    # Qubit 1 idles `idle_detuning_mhz` above qubit 2. With the same frequency (idle_detuning_mhz = 0) and g = 20 MHz
    # the qubits swap |01> <-> |10> in 25 ns even with NO pulse (measured: 90% swap after 40 ns).
    # |11> meets |20> when qubit 1 is lowered to (qubit 2 + |anharmonicity|), i.e. by (idle - 300 MHz) from idle.
    H0 = jnp.kron(H1, eye) + jnp.kron(eye, H2) + H_int + 2 * jnp.pi * (idle_detuning_mhz / 1000.0) * Hz_control
    return H0, Hz_control

def build_two_qubit_base(levels=3, anharm_1=-300.0, anharm_2=-300.0):
    """Builds the static base matrices that do not need calibration."""
    b = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), 1)
    bdag = b.T
    eye = jnp.eye(levels)
    num = bdag @ b
 
    H1 = 0.5 * 2 * jnp.pi * (anharm_1 / 1000.0) * num @ (num - eye)
    H2 = 0.5 * 2 * jnp.pi * (anharm_2 / 1000.0) * num @ (num - eye)
    
    H_anharm_base = jnp.kron(H1, eye) + jnp.kron(eye, H2)
    H_int_base = jnp.kron(b, bdag) + jnp.kron(bdag, b)
    Hz_control = jnp.kron(num, eye)
    
    return H_anharm_base, H_int_base, Hz_control

def assemble_H0(H_anharm_base, H_int_base, Hz_control, coupling_mhz, idle_detuning_mhz, alpha_mhz=-300.0):
    # Scale the base anharmonicity matrix linearly relative to the default -300.0 MHz
    H_anharm = (alpha_mhz / -300.0) * H_anharm_base
    H_int = 2 * jnp.pi * (coupling_mhz / 1000.0) * H_int_base
    H_detuning = 2 * jnp.pi * (idle_detuning_mhz / 1000.0) * Hz_control
    
    H0 = H_anharm + H_int + H_detuning
    return H0

def build_true_H0(config):
    """Exact device Hamiltonian: true parameters plus the unmodeled ZZ term (JAX twin of HiddenDevice.H0)."""
    hw = config.hardware
    H0, Hz = build_two_qubit_operators(hw.levels, hw.anharmonicity_mhz, hw.anharmonicity_mhz,
                                       hw.coupling_mhz, hw.idle_detuning_mhz)
    num = jnp.diag(jnp.arange(hw.levels, dtype=float))
    H0 = H0 + 2 * jnp.pi * (hw.zz_crosstalk_mhz / 1000.0) * jnp.kron(num, num)
    return H0, Hz