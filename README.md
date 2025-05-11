# Gamma Telescope Readings Classification Model

## Overview

This repository contains a machine learning model designed to classify cosmic ray events detected by gamma telescopes as either gamma rays or hadrons. The model utilizes data from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml), specifically the "Magic Gamma Telescope Data Set." By analyzing various features extracted from telescope readings, the model can effectively distinguish between these two types of high-energy particles.

The goal of this project is to provide a robust and accurate classification model that can be used in astrophysical research for event categorization and analysis.

## Dataset

The model was trained and evaluated using the publicly available "Magic Gamma Telescope Data Set" from the UCI Machine Learning Repository. This dataset contains features derived from the signals recorded by the MAGIC (Major Atmospheric Gamma-ray Imaging Cherenkov) telescope system.

**Citation:**

> Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
>
> Donated by: P. Savicky Institute of Computer Science, AS of CR Czech Republic savicky '@' cs.cas.cz

## Model and Techniques

This model employs multiple machine learning techniques to achieve an accuracy of **88%** in classifying gamma and hadron events. While the specific techniques used to obtain this best accuracy are not explicitly stated here, the development process likely involved experimentation with various classification algorithms and hyperparameter tuning.

The following libraries were utilized in the development of this model:

* **NumPy:** For numerical computations and array manipulation (`import numpy as np`).
* **Pandas:** For data manipulation and analysis using DataFrames (`import pandas as pd`).
* **Matplotlib:** For creating visualizations and plots (`import matplotlib.pyplot as plt`).
* **Scikit-learn (sklearn):** For machine learning tasks, specifically:
    * `StandardScaler` for feature scaling (`from sklearn.preprocessing import StandardScaler`).
* **Imbalanced-learn (imblearn):** For addressing potential class imbalance:
    * `RandomOverSampler` for oversampling the minority class (`from imblearn.over_sampling import RandomOverSampler`).

## Usage

While the specific implementation details and model file are not provided here, the general workflow for using such a model would typically involve the following steps:

1.  **Loading the Model:** Load the trained machine learning model from a saved file (e.g., using libraries like `pickle` or `joblib`).
2.  **Data Preparation:** Ensure that the input data (new telescope readings) is preprocessed in the same way as the training data. This typically includes:
    * Loading the data into a Pandas DataFrame.
    * Scaling the features using the same `StandardScaler` object fitted on the training data.
3.  **Prediction:** Use the loaded model's `predict()` method to classify the new data points as either gamma (likely represented by one class label) or hadron (represented by another class label).

**Example (Conceptual):**

```python
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('gamma_hadron_classifier.pkl') # Replace with your actual model file

# Load new data (replace with your actual data loading)
new_data = pd.read_csv('new_telescope_readings.csv')

# Assuming the features in new_data are in the same columns as the training data
features = ['feature1', 'feature2', ..., 'feature10'] # Replace with your actual feature names
X_new = new_data[features]

# Load the scaler fitted on the training data
scaler = joblib.load('scaler.pkl') # Replace with your actual scaler file
X_scaled_new = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_scaled_new)

# Interpret the predictions (assuming 0 is hadron and 1 is gamma)
predicted_classes = ['gamma' if p == 1 else 'hadron' for p in predictions]

print(predicted_classes)
