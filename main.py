# ---------- main.py ----------
import pandas as pd
import numpy as np
import os

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv('Pune_House_Data.csv')

# Select and clean relevant features
df_model = df[['location', 'size', 'total_sqft', 'bath', 'price']].copy()
df_model['bhk'] = df_model['size'].str.extract(r'(\d+)').astype(float)

def convert_sqft_to_num(x):
    try:
        if '-' in x:
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return None

df_model['total_sqft'] = df_model['total_sqft'].apply(convert_sqft_to_num)
df_model.dropna(subset=['location', 'total_sqft', 'bath', 'price', 'bhk'], inplace=True)

# Create price_per_sqft and remove outliers
df_model['price_per_sqft'] = (df_model['price'] * 100000) / df_model['total_sqft']

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = subdf['price_per_sqft'].mean()
        st = subdf['price_per_sqft'].std()
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] < (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df_clean = remove_pps_outliers(df_model)

# One-hot encode locations
location_dummies = pd.get_dummies(df_clean['location'], prefix='loc')
df_encoded = pd.concat([df_clean, location_dummies], axis=1)
df_encoded.drop(['location', 'size'], axis=1, inplace=True)

# Prepare training data
X = df_encoded.drop(['price', 'price_per_sqft'], axis=1).values.astype(np.float64)
y = df_encoded['price'].values.reshape(-1, 1)

# Feature scaling
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

# Save for use in Flask
np.save('model/mean.npy', mean)
np.save('model/std.npy', std)
np.save('model/columns.npy', df_encoded.drop(['price', 'price_per_sqft'], axis=1).columns)

# Initialize model
m, n = X_scaled.shape
w = np.zeros((n, 1))
b = 0

# Helper functions
def predict(X, w, b):
    return np.dot(X, w) + b

def compute_cost(y, y_pred):
    return (1 / (2 * len(y))) * np.sum((y_pred - y) ** 2)

def compute_gradient(X, y, y_pred):
    m = len(y)
    dw = (1 / m) * np.dot(X.T, (y_pred - y))
    db = (1 / m) * np.sum(y_pred - y)
    return dw, db

# Gradient descent
learning_rate = 0.01
epochs = 1000
for i in range(epochs):
    y_pred = predict(X_scaled, w, b)
    cost = compute_cost(y, y_pred)
    dw, db = compute_gradient(X_scaled, y, y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db
    if i % 100 == 0:
        print(f"Epoch {i}: Cost = {cost:.2f}")

# Save model
np.save('model/weights.npy', w)
np.save('model/bias.npy', b)

# Evaluation
y_final_pred = predict(X_scaled, w, b)
mae = np.mean(np.abs(y - y_final_pred))
mse = np.mean((y - y_final_pred)**2)
rmse = np.sqrt(mse)

print(f"\nMAE (in lakhs): {mae:.2f}")
print(f"MSE (in lakhs): {mse:.2f}")
print(f"RMSE (in lakhs): {rmse:.2f}")