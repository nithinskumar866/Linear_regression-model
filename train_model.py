# train_model.py

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Create dataset directory
os.makedirs("dataset", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Generate realistic productivity dataset
np.random.seed(42)
data_size = 150

data = pd.DataFrame({
    'Work_Hours': np.random.normal(8, 1.5, data_size).clip(4, 12),
    'Break_Time': np.random.normal(1, 0.3, data_size).clip(0.3, 2),
    'Tasks_Completed': np.random.normal(5, 2, data_size).clip(1, 10),
})

# Target variable: Productivity Score (out of 100)
data['Productivity'] = (
    data['Work_Hours'] * 10
    - data['Break_Time'] * 5
    + data['Tasks_Completed'] * 7
    + np.random.normal(0, 5, data_size)  # noise
).clip(30, 100)

# Save dataset
data.to_csv("dataset/productivity_data.csv", index=False)

# Split and scale
X = data.drop("Productivity", axis=1)
y = data["Productivity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
with open("model/lr_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved.")
