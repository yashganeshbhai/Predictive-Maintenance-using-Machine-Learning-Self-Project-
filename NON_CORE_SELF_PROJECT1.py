import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# --- Step 1: Load and Preprocess Data ---
print("⚙️ Step 1: Loading and preprocessing data...")

# Define column names based on the dataset's structure
column_names = ['unit', 'time', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]

# Load the dataset, skipping the extra whitespace columns at the end
try:
    data = pd.read_csv('C:\\Users\\dell\\Downloads\\Self_project_non_core\\train_FD001.txt', sep=' ', header=None)
    data.dropna(axis=1, inplace=True) # Drops the empty columns
    data.columns = column_names
except FileNotFoundError:
    print("Error: 'train_FD001.txt' not found. Please ensure the dataset is in the same folder.")
    exit()

# --- Step 2: Calculate Remaining Useful Life (RUL) ---
print("⚙️ Step 2: Calculating Remaining Useful Life (RUL)...")

# Group data by engine unit and find the maximum cycle for each
max_cycles = data.groupby('unit')['time'].max().reset_index()
max_cycles.columns = ['unit', 'max_cycle']

# Merge the max cycle back into the main dataframe
data = data.merge(max_cycles, on='unit', how='left')

# Calculate RUL: RUL = total cycles - current cycle
data['RUL'] = data['max_cycle'] - data['time']
data.drop(columns='max_cycle', inplace=True)

# --- Step 3: Prepare Features and Labels ---
print("⚙️ Step 3: Preparing features and labels...")

# Drop non-predictive columns and sensors that don't vary in this dataset
drop_cols = ['unit', 'time', 'RUL', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
feature_cols = [col for col in data.columns if col not in drop_cols]

X = data[feature_cols] # Features (inputs)
y = data['RUL']        # Labels (the target we want to predict)

# --- Step 4: Scale Data and Split for Training ---
print("⚙️ Step 4: Scaling data and splitting for training...")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Step 5: Train the Random Forest Model ---
print("⚙️ Step 5: Training the Random Forest model...")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- Step 6: Evaluate the Model on the Test Set ---
print("⚙️ Step 6: Evaluating the model...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance on Test Set:")
print(f"   - Mean Squared Error (MSE): {mse:.2f}")
print(f"   - R-squared (R²): {r2:.4f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Remaining Useful Life (RUL)')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 7: Save Model and Scaler for Future Use ---
print("⚙️ Step 7: Saving the model and scaler...")
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/rul_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
pickle.dump(feature_cols, open('model/feature_list.pkl', 'wb'))
print("✅ Model, scaler, and feature list saved in the 'model' directory.")

# --- Step 8: Demonstrate a Prediction on New Data ---
print("\n⚙️ Step 8: Demonstrating a prediction for a new engine...")

# Sample data for a single engine's current state (sensor readings)
# This would come from a real-world engine monitoring system
new_engine_data = {
    'operational_setting_1': 0.0,
    'operational_setting_2': 0.0,
    'operational_setting_3': 100.0,
    'sensor_2': 642.82, 'sensor_3': 1589.66, 'sensor_4': 1405.6,
    'sensor_7': 39.06, 'sensor_8': 23.28, 'sensor_9': 100.0,
    'sensor_11': 39.0, 'sensor_12': 23.23, 'sensor_13': 47.47,
    'sensor_14': 522.49, 'sensor_15': 2388.06, 'sensor_17': 8.42,
    'sensor_20': 2388.0, 'sensor_21': 100.0
}

# Load the saved components
loaded_model = pickle.load(open('model/rul_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('model/scaler.pkl', 'rb'))
loaded_features = pickle.load(open('model/feature_list.pkl', 'rb'))

# Convert input data to a DataFrame and ensure column order matches training data
input_df = pd.DataFrame([new_engine_data])[loaded_features]

# Scale the new input data using the loaded scaler
scaled_input = loaded_scaler.transform(input_df)

# Make the prediction
predicted_rul = loaded_model.predict(scaled_input)[0]

print(f"\n🔧 Predicted Remaining Useful Life (RUL) for the new engine: {predicted_rul:.2f} cycles")
print("✅ Script execution complete.")