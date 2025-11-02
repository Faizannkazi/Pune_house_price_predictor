# Pune House Price Prediction

This project is a web application that predicts house prices in Pune based on user inputs like area (in square feet), number of bedrooms (BHK), and location. It uses a simple linear regression model trained on historical housing data.

---

## 📊 Dataset

The dataset contains historical housing prices for various locations in Pune. It includes:
- Location
- Total square feet
- BHK (Bedroom, Hall, Kitchen)
- Price (in Lakhs)

The data is cleaned and preprocessed before training, including handling missing values and one-hot encoding for location features.

---

## 🛠️ Tools & Technologies

**Frontend:**
- HTML
- CSS (custom styling + glassmorphism effect)
- JavaScript
- Select2.js (for searchable dropdown)

**Backend:**
- Python
- Flask

**Machine Learning:**
- NumPy
- Pandas

---

## 📁 Project Structure

house_price_pune/
├── app.py # Flask backend
├── main.py # ML training script
├── data.csv # Raw dataset
├── model/
│ ├── weights.npy # Trained weights
│ ├── bias.npy # Bias value
│ ├── mean.npy # Feature means
│ ├── std.npy # Feature standard deviations
│ └── columns.json # Feature column names
├── templates/
│ └── index.html # HTML frontend
├── static/
│ └── style.css # Web page styling

---

## 🔧 How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/Faizannkazi/Pune_house_price_predictor.git
cd Pune_house_price_predictor
---

### Step 2: Install Requirements
pip install flask numpy pandas
---


### Step 3: Train the Model
Run this once to train the model and save weights:
python main.py
---

### Step 4: Run the Web App
python app.py
Visit http://127.0.0.1:5000 in your browser.
🔍 Input Details
Total Sqft: Maximum allowed is 5000 sqft
BHK: Only 1–4 BHK are supported
Location: Must be selected from dropdown (search supported)
🖥️ Output
The model predicts the estimated house price in Lakhs based on user input. The prediction is displayed instantly on the same page.
---

### 📌 Example
Input:
Total Sqft: 1500
BHK: 3
Location: Baner
Output:
Predicted Price: ₹ 72.84 Lakh
----


