import tensorflow as tf
from tensorflow import data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import mae
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from AutoencoderClassifier.entity.config_entity import TrainingConfig

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


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_path,
        custom_objects={'AutoEncoder': AutoEncoder})
    
    def get_data(self):
        path_str =str(self.config.training_data)

        self.data = pd.read_csv(
        path_str,
        header=None ).iloc[:, :-1]
        

    def split_data(self):
        data = self.data.to_numpy()
        self.X_train, self.X_test = train_test_split(data, test_size=0.15, random_state=45, shuffle=True)
        print(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        self.save_data(self.config.X_train_path,self.X_train)
        self.save_data(self.config.X_test_path,self.X_test)
        
    def fit_model(self):
        epochs = 100
        batch_size = 128
        early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)


        history = self.model.fit(self.X_train, self.X_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.1, callbacks=[early_stopping])
        
        self.save_model(self,model=self.model)
        self.model.summary()

        return history

    @staticmethod
    def save_data(path,data):
        np.save(path,data)
    @staticmethod
    def save_model(self, model: tf.keras.Model):
            """
            Saves model to the configured path.
            """
            model_path = self.config.trained_model_path
            
            model.save(model_path)     # saves full model (architecture + weights + optimizer)
            print(f"✅ Model saved at: {model_path}")
