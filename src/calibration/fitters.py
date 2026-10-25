import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from src.calibration.cz_training import calibration_loss

class DigitalTwinFitter:
    def __init__(self, dt_step, H_anharm_base, H_int_base, Hz_control, learning_rate=0.1, steps=250):
        """
        dt_step: step size between Chevron duration checkpoints in ns (e.g., 8.0)
        """
        self.dt_step = dt_step
        self.H_anharm_base = H_anharm_base
        self.H_int_base = H_int_base
        self.Hz_control = Hz_control
        self.steps = steps
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=1e-4
        )
        self.optimizer = optax.adam(scheduler)

        @jax.jit
        def _step_fn(params, state, jax_amps, jax_targets_10, jax_targets_11):
            def loss_wrapper(p):
                return calibration_loss(
                    p, 
                    self.dt_step, 
                    self.H_anharm_base, 
                    self.H_int_base, 
                    self.Hz_control, 
                    jax_amps, 
                    jax_targets_10, 
                    jax_targets_11
                )

            loss, grads = jax.value_and_grad(loss_wrapper)(params)
            updates, state = self.optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, loss

        self._step_fn = _step_fn

    def fit(self, amplitudes_rad, target_dataset_10, target_dataset_11, init_coupling=20.0, init_detuning=600.0, init_alpha=-305.0):
        calib_params = {
            'coupling_mhz': jnp.array(init_coupling, dtype=jnp.float64),
            'detuning_mhz': jnp.array(init_detuning, dtype=jnp.float64),
            'anharmonicity_mhz': jnp.array(init_alpha, dtype=jnp.float64)
        }
        opt_state = self.optimizer.init(calib_params)
        
        jax_amps = jnp.array(amplitudes_rad, dtype=jnp.float64)
        jax_targets_10 = jnp.array(target_dataset_10, dtype=jnp.float64)
        jax_targets_11 = jnp.array(target_dataset_11, dtype=jnp.float64)

        history = {"loss": [], "coupling": [], "detuning": [], "anharmonicity": []}

        for step in tqdm(range(self.steps), desc="Calibrating Digital Twin"):
            calib_params, opt_state, loss = self._step_fn(
                calib_params, opt_state, jax_amps, jax_targets_10, jax_targets_11
            )
            
            history["loss"].append(float(loss))
            history["coupling"].append(float(calib_params['coupling_mhz']))
            history["detuning"].append(float(calib_params['detuning_mhz']))
            history["anharmonicity"].append(float(calib_params['anharmonicity_mhz']))

        return calib_params, history