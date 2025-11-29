import numpy as np
import mlflow
import mlflow.keras
import pandas as pd 
import sys
from AutoencoderClassifier.exception import CustomException

import tensorflow as tf
from tensorflow import data
from tensorflow.keras.metrics import mae

import pandas as pd
import numpy as np

import json
from pathlib import Path
from AutoencoderClassifier.utils.common import save_json
from AutoencoderClassifier.entity.config_entity import EvaluationConfig
from AutoencoderClassifier.components.model_training import AutoEncoder


def predict(model, data: np.ndarray):
    pred = model.predict(data, verbose=False)
    loss = mae(pred, data)
    return pred, loss
class Evaluation():
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_datasets(self):
        # TODO: replace with your real loading logic
        # X_train, X_test, anomaly must be numpy arrays of shape (n_samples, 187)
        X_train = np.load(self.config.X_train_path)
        X_test  = np.load(self.config.X_test_path)
        
        path_str =str(self.config.evaluation_data)

        self.data = pd.read_csv(
            path_str,
            header=None ).iloc[:, :-1]
        
        anomaly = self.data.to_numpy()
        self.save_data(self.config.anomaly_path,anomaly)

        return X_train, X_test, anomaly

    @staticmethod
    def save_data(path,data):
        np.save(path,data)
   
    def _load_model(self):
        model = tf.keras.models.load_model(
            self.config.trained_model_path,
            custom_objects={"AutoEncoder": AutoEncoder}
        )
        return model
    def predict(model, data):
        pred = model.predict(data, verbose=0)
        loss = np.mean(np.abs(pred - data), axis=1)  # MAE per sample
        return pred, loss

    
    def flatten_dict(self,d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
            return dict(items)

    
    def run_evaluation(self):
    
        """
        main evaluation function with MLflow
        """
        try:

            # 1) MLflow tracking to DagsHub
            print(self.config.ml_flow_URI)
            mlflow.set_tracking_uri(self.config.ml_flow_URI)
            mlflow.set_experiment("ecg_anomaly_detection")

            # 2) Load data
            X_train, X_test, anomaly = self.load_datasets()

            # 3) Load model
            model = self._load_model()

            # 4) Start main MLflow run
            with mlflow.start_run(run_name="autoencoder_eval"):

                # Log hyperparams from config (flattened YAML)
                flat_params = self.flatten_dict(self.config.all_params)
                mlflow.log_params(flat_params)

                
                train_pred, train_loss = predict(model, X_train)
                test_pred,  test_loss  = predict(model, X_test)
                anom_pred,  anom_loss  = predict(model, anomaly)

                mlflow.log_metric("train_loss_mean", float(np.mean(train_loss)))
                mlflow.log_metric("test_loss_mean",  float(np.mean(test_loss)))
                mlflow.log_metric("anom_loss_mean",  float(np.mean(anom_loss)))

                # -------- MULTIPLE THRESHOLDS PART --------
                percentiles = self.config.threshold_percentile 

                best_thr = None
                best_anom_acc = -1
                best_metrics = None

                for perc in percentiles:
                    threshold = np.percentile(train_loss, perc)

                    train_acc = np.mean(train_loss <= threshold)
                    test_acc  = np.mean(test_loss  <= threshold)
                    anom_acc  = np.mean(anom_loss  >  threshold)

                    print(
                        f"P={perc}: thr={threshold:.5f} | "
                        f"train={train_acc:.2%}, test={test_acc:.2%}, anom={anom_acc:.2%}"
                    )

                    # Nested run for this particular threshold
                    with mlflow.start_run(run_name=f"th_{perc}", nested=True):
                        mlflow.log_param("threshold_percentile", perc)
                        mlflow.log_metric("threshold_value", float(threshold))
                        mlflow.log_metric("train_accuracy", float(train_acc))
                        mlflow.log_metric("test_accuracy", float(test_acc))
                        mlflow.log_metric("anomaly_accuracy", float(anom_acc))

                    # Track the best anomaly accuracy
                    if anom_acc > best_anom_acc:
                        best_anom_acc = anom_acc
                        best_thr = threshold
                        best_metrics = {
                            "percentile": perc,
                            "threshold": float(threshold),
                            "train_accuracy": float(train_acc),
                            "test_accuracy": float(test_acc),
                            "anomaly_accuracy": float(anom_acc),
                        }

                mlflow.keras.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="ecg_autoencoder"
                )

                metrics_dir = Path("artifacts/metrics")
                metrics_dir.mkdir(parents=True, exist_ok=True)
                save_json(metrics_dir / "metrics.json", best_metrics)

                print("\nBest threshold summary:")
                print(f"Percentile:       {best_metrics['percentile']}")
                print(f"Threshold:        {best_metrics['threshold']:.5f}")
                print(f"Train accuracy:   {best_metrics['train_accuracy']:.2%}")
                print(f"Test accuracy:    {best_metrics['test_accuracy']:.2%}")
                print(f"Anomaly accuracy: {best_metrics['anomaly_accuracy']:.2%}")

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    from pathlib import Path

    config = EvaluationConfig(
        root_dir=Path("artifacts/model"),
        trained_model_path=Path("artifacts/training/model.keras"),
        training_data=Path("data/ptbdb_normal.csv"),
        evaluation_data= Path("data/ptbdb_abnormal.csv"),
        anomaly_path = Path("artifacts/data_ingestion/anomaly.npy"),
        X_train_path = Path("artifacts/data_ingestion/X_train.npy"),
        X_test_path = Path("artifacts/data_ingestion/X_test.npy"),
        all_params={"input_dimension" : 187,
                "latent_dimension" : 32,
                "learning_rate" : 0.001,
                "epochs" : 100,
                "batch_size" : 128,
                "patience": 10,
                "min_delta": 1e-3,
                "monitor": "val_loss",
                "restore_best_weights": True},
        ml_flow_URI=str(),
        threshold_percentile=[90, 92, 95, 97, 99]
    )
    run = Evaluation(config)
    run.run_evaluation()
