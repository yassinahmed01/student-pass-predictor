from flask import Flask, render_template, request
import numpy as np
import pickle
import xgboost as xgb

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the features (in the exact order used during training)
FEATURES = ['sex', 'age', 'studytime', 'failures', 'absences', 
            'G1', 'G2', 'goout', 'Medu', 'Fedu']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        data = [float(request.form.get(feat)) for feat in FEATURES]

        # Scale input
        X_input = np.array(data).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # Predict with XGBoost model (requires DMatrix)
        dmatrix = xgb.DMatrix(X_scaled)
        prediction = model.predict(dmatrix)[0]
        result = "PASS" if prediction >= 0.5 else "FAIL"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=port)

