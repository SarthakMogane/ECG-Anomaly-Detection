import tensorflow as tf
from tensorflow import data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
from AutoencoderClassifier.entity.config_entity import PrepareBaseModelConfig

gpus = tf.config.list_physical_devices('GPU')
gpus

tf.keras.utils.set_random_seed(1024)

class AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((input_dim, 1)),  # Reshape to 3D for Conv1D
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
        ])
        # Previously, I was using UpSampling. I am trying Transposed Convolution this time around.
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
#             layers.UpSampling1D(2),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(input_dim)
        ])

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super(AutoEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        })
        return config
    
    # *** ADDED CLASSMETHOD TO ALLOW LOADING ***
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BuildModel():

    def __init__(self,config: PrepareBaseModelConfig):
        self.config= config

    def get_model(self) -> tf.keras.Model:
        try:
            model = AutoEncoder(
                input_dim=self.config.params_input_dimension,
                latent_dim=self.config.params_latent_dimension,
            )
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.params_learning_rate
                ),
                loss="mae"
            )
        
            self.save_model(self,model=model)
            
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def save_model(self, model: tf.keras.Model):
            """
            Saves model to the configured path.
            """
            model_path = self.config.base_model_path
            model.save(model_path)     # saves full model (architecture + weights + optimizer)
            print(f"✅ Model saved at: {model_path}")  


if __name__ == "__main__":
    # TEMPORARY TEST CONFIG
    from pathlib import Path
    test_config = PrepareBaseModelConfig(
        root_dir=Path("."),
        base_model_path=Path("model.keras"),
        params_input_dimension=187,
        params_learning_rate=0.001,
        params_latent_dimension=32
    )

    builder = BuildModel(test_config)
    model = builder.get_model()
    model.summary()