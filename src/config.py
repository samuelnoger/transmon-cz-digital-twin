from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareConfig:
    anharmonicity_mhz: float
    levels: int
    coupling_mhz: Optional[float] = None
    idle_detuning_mhz: Optional[float] = None
    n_shots: Optional[int] = None
    zz_crosstalk_mhz: float = 0.0               
    flux_bandwidth_mhz: Optional[float] = None  

@dataclass
class DecoherenceConfig:
    t1_ns: float
    t2_ns: float

@dataclass
class PulseConfig:
    duration_ns: float
    dt_ns: float
    max_amp_mhz: Optional[float] = None
    max_flux_amp_ghz: Optional[float] = None
    cutoff_freq_ghz: Optional[float] = None

@dataclass
class OptimizerConfig:
    learning_rate: float
    steps: int
    leakage_weight: float

@dataclass
class ExperimentConfig:
    hardware: HardwareConfig
    pulse: PulseConfig
    optimizer: OptimizerConfig
    decoherence: Optional[DecoherenceConfig] = None
    name: str = "ideal"

    def artifact_path(self, filename: str) -> Path:
        folder = Path(__file__).parent.parent / "artifacts" / self.name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / filename

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        # Resolve the absolute path from the project root
        project_root = Path(__file__).parent.parent
        absolute_path = project_root / filepath
        
        with open(absolute_path, 'r') as f:
            raw = yaml.safe_load(f)
            
        # Safely handle the decoherence block if it exists
        decoherence_data = raw.get('decoherence')
        decoherence_config = DecoherenceConfig(**decoherence_data) if decoherence_data else None
            
        return cls(
            hardware=HardwareConfig(**raw['hardware']),
            pulse=PulseConfig(**raw['pulse']),
            optimizer=OptimizerConfig(**raw['optimizer']),
            decoherence=decoherence_config,
            name=raw.get("name", "ideal")
        )