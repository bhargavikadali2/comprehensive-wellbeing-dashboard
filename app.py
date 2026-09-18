
from flask import Flask, render_template, request
import joblib
import pandas as pd


# ==========================================
# Create Flask App
# ==========================================

app = Flask(__name__)


# ==========================================
# Load Trained Model
# ==========================================

model = joblib.load("wellbeing_model.pkl")


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from HTML form

        life_expectancy = float(
            request.form["life_expectancy"]
        )

        mean_years = float(
            request.form["mean_years"]
        )

        expected_years = float(
            request.form["expected_years"]
        )

        gni = float(
            request.form["gni"]
        )


        # ==========================================
        # Create DataFrame
        # ==========================================

        input_data = pd.DataFrame(
            [[
                life_expectancy,
                mean_years,
                expected_years,
                gni
            ]],
            columns=[
                "Life expectancy",
                "Mean years of schooling",
                "Expected years of schooling",
                "Gross national income (GNI) per capita"
            ]
        )


        # ==========================================
        # Predict HDI
        # ==========================================

        prediction = model.predict(input_data)[0]


        # Keep prediction between 0 and 1

        prediction = max(0, min(1, prediction))


        # ==========================================
        # Determine HDI Level
        # ==========================================

        if prediction < 0.550:
            level = "Low Human Development"

        elif prediction < 0.700:
            level = "Medium Human Development"

        elif prediction < 0.800:
            level = "High Human Development"

        else:
            level = "Very High Human Development"


        # ==========================================
        # Show Result
        # ==========================================

        return render_template(
            "prediction.html",
            prediction=round(float(prediction), 3),
            level=level
        )


    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/">Go Back</a>
        """


# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)

