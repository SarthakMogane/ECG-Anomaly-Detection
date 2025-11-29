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
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    trained_model_path: Path
    base_model_path: Path
    training_data : Path
    X_train_path: Path
    X_test_path: Path
    params_epochs : int
    params_batch_size : int
    params_patience: int
    params_min_delta: float
    params_monitor: str
    params_restore_best_weights: bool

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir:Path
    trained_model_path:Path
    training_data:Path
    evaluation_data:Path
    anomaly_path : Path
    X_train_path :Path
    X_test_path :Path
    all_params: dict
    ml_flow_URI:str