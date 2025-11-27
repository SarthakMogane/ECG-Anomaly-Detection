from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    ptbdb_normal_data_path: Path
    ptbdb_abnormal_data_path : Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_input_dimension: int
    params_learning_rate: float
    params_latent_dimension: int
    