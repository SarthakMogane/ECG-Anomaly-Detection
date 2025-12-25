# 🫀 ECG Anomaly Detection using Deep Autoencoders

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)
![DVC](https://img.shields.io/badge/DVC-Pipeline-9cf)

## 📌 Project Overview

This project implements an end-to-end **ECG anomaly detection system** using a CNN-based Autoencoder for unsupervised learning. The system is designed to identify abnormal heartbeats by learning the reconstruction pattern of normal ECG signals and flagging deviations as anomalies.

Beyond model training, this project emphasizes **reproducibility, experiment tracking, and production-oriented ML system design** using **DVC** and **MLflow (DagsHub)**.

## 🎯 Key Objectives

* **Unsupervised Detection:** Detect anomalous ECG beats without relying on labeled anomaly data during training.
* **MLOps Best Practices:** Build a reproducible ML pipeline using industry-standard tools (DVC, MLflow).
* **Adaptive Thresholding:** Evaluate anomaly detection using dynamic percentile-based strategies.
* **Production Readiness:** Prepare the system architecture for real-time (online) ECG monitoring via sliding-window inference.

---

## 🧠 Model Architecture

The core of the system is a **CNN-based Autoencoder** optimized for time-series data.

* **Input:** Single ECG heartbeat (187 samples).
* **Encoder:** * Composed of `Conv1D` layers, `Batch Normalization`, and `MaxPooling`.
    * Learns a compressed latent representation of *normal* ECG patterns.
* **Decoder:**
    * Composed of `Conv1DTranspose` layers.
    * Reconstructs the ECG signal from the latent space.
* **Loss Function:** Mean Absolute Error (MAE).

> **Logic:** The model is trained *only* on normal heartbeats. When it encounters an anomaly, the reconstruction error (MAE) spikes, signaling a defect.

---

## ⚙️ System Architecture & Pipeline

The project follows a modular, config-driven ML pipeline orchestrated by DVC:

```mermaid
graph LR
    A[Data Ingestion] --> B[Base Model Prep]
    B --> C[Model Training]
    C --> D[Model Evaluation]


Each stage is:

Config-driven (config.yaml)

Reproducible via DVC

Logged and tracked via MLflow
Data Ingestion: Loading and cleaning data from the source.

Base Model Preparation: Defining architecture parameters via config.yaml.

Model Training: Training the Autoencoder and logging metrics to MLflow.

Model Evaluation: Testing against unseen anomalies and calculating threshold statistics.

All stages are reproducible via the dvc repro command.

📊 Data & Preprocessing
Dataset: PTB Diagnostic ECG Database.

Training Set: Contains Normal ECGs only (to teach the model "normality").

Test/Anomaly Set: Contains both Normal and Abnormal ECGs (for evaluation).

Preprocessing: Signals are truncated to a fixed length of 187 samples and stored as NumPy arrays for efficient I/O.

🔍 Anomaly Detection Strategy
We utilize Reconstruction Loss as the anomaly score.

Inference: Pass the ECG signal through the Autoencoder.

Calculation: Compute MAE between Input and Reconstructed Output.

Thresholding: - If MAE > Threshold → Anomaly 🚨

If MAE <= Threshold → Normal ✅

We evaluated multiple percentile-based thresholds (90th, 95th, 99th) to balance precision and recall. These metrics are tracked in MLflow to visualize the trade-offs.

🔄 Real-Time (Online) Monitoring Design
The system supports a Sliding Window Inference approach for streaming data:

A rolling window of 187 samples captures the live signal.

Reconstruction error is computed in near real-time.

Simulates real-world scenarios like wearable devices or ICU monitoring.

📈 Experiment Tracking & Reproducibility
MLflow (DagsHub)

Model parameters

Threshold values

Train / Test / Anomaly accuracy

Reconstruction loss statistics

Model versions via MLflow Registry

DVC

Data versioning

Pipeline orchestration

Reproducible experiments using dvc repro


🧪 Evaluation Metrics

Reconstruction Loss Distribution

Accuracy under different thresholds

Anomaly detection rate

Comparative analysis of threshold strategies

Metrics are saved as artifacts and tracked across runs.

🧩 Extensibility & Future Enhancements

The project is designed for easy extension, including:

🔍 Explainability via reconstruction-error visualization

📉 Data drift detection using statistical tests

⚡ Real-time inference via FastAPI or streaming frameworks

🏷 Automated model promotion using MLflow Registry

📊 Live monitoring dashboards

🚀 Tech Stack

Deep Learning: TensorFlow / Keras

MLOps: DVC, MLflow, DagsHub

Data: NumPy, Pandas, Scikit-learn

Visualization: Matplotlib

Deployment Ready: Model serialization (.keras format)

📌 How to Run
# Reproduce full pipeline
dvc repro

# Track experiments
mlflow ui

🧠 What This Project Demonstrates

Strong understanding of unsupervised anomaly detection

End-to-end ML system design, not just model training

Practical MLOps skills (DVC + MLflow)

Readiness for real-world deployment scenarios

Clear separation of concerns and scalable architecture

📬 Author

Sarthak Mogane
Machine Learning & AI Engineer
Focused on scalable ML systems and production-ready AI solutions