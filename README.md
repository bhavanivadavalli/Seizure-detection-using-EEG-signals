# Seizure-detection-using-EEG-signals
Automated epileptic seizure detection using EEG signals, feature extraction, and XGBoost.
# Automated Epileptic Seizure Detection Using EEG Signals and XGBoost

## Project Overview

This project is an automated EEG-based epileptic seizure detection system developed using machine learning. The system analyzes EEG signals and classifies them into seizure and non-seizure categories.

## Features

- EEG signal file upload
- Manual EEG signal input
- EEG signal visualization
- Feature extraction
- XGBoost-based classification
- Seizure / non-seizure prediction
- User-friendly Streamlit interface

## Dataset

The project uses the Bonn University EEG Dataset.

The dataset contains five groups (A, B, C, D, and E), with each group containing EEG signal segments.

For this project:
- A–D → Non-Seizure
- E → Seizure

## Feature Extraction

The system extracts multiple features from EEG signals, including:

- Mean
- Standard deviation
- Maximum
- Minimum
- Signal energy
- Hjorth activity
- Hjorth mobility
- Hjorth complexity
- Entropy
- Wavelet energy
- FFT-based frequency features

## Machine Learning Model

XGBoost is used as the primary classification algorithm.

The extracted EEG features are provided to the trained XGBoost model, which predicts whether the input signal belongs to the seizure or non-seizure class.

## Technology Stack

- Python
- Streamlit
- XGBoost
- NumPy
- SciPy
- Scikit-learn
- PyWavelets
- Matplotlib
- MySQL

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/eeg-seizure-detection-xgboost.git
