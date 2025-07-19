from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model parameters
columns = np.load('model/columns.npy', allow_pickle=True).tolist()
mean = np.load('model/mean.npy')
std = np.load('model/std.npy')
weights = np.load('model/weights.npy')
bias = np.load('model/bias.npy')

# Extract location names from column names
locations = sorted([col.replace('loc_', '') for col in columns if col.startswith('loc_')])

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_sqft = float(request.form['total_sqft'])
        bhk = int(request.form['bhk'])
        location = request.form['location']

        # Input validation
        if bhk > 4 or total_sqft > 5000:
            return render_template('index.html', locations=locations, prediction="❌ Limit Exceeded!")

        # Create input vector
        input_dict = {
            'total_sqft': total_sqft,
            'bhk': bhk,
            f'loc_{location}': 1
        }

        for col in columns:
            if col not in input_dict:
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[columns]

        # Apply scaling
        input_scaled = (input_df.values - mean) / std

        # Predict
        prediction = np.dot(input_scaled, weights) + bias
        predicted_price = round(prediction[0][0], 2)

        return render_template('index.html', locations=locations, prediction=predicted_price)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
