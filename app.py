from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model/wellbeing_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    country = int(request.form["Country"])
    life = float(request.form["Life expectancy"])
    expected = float(request.form["Expected years of schooling"])
    mean = float(request.form["Mean years of schooling"])
    gni = float(request.form["Gross national income (GNI) per capita"])

    data = pd.DataFrame([[country, life, expected, mean, gni]],
                        columns=[
                            "Country",
                            "Life expectancy",
                            "Expected years of schooling",
                            "Mean years of schooling",
                            "Gross national income (GNI) per capita"
                        ])

    prediction = model.predict(data)[0]
    prediction = round(prediction, 3)

    if prediction < 0.55:
        level = "Low Human Development"
    elif prediction < 0.70:
        level = "Medium Human Development"
    elif prediction < 0.80:
        level = "High Human Development"
    else:
        level = "Very High Human Development"

    return render_template(
        "result.html",
        prediction=prediction,
        level=level
    )


if __name__ == "__main__":
    app.run(debug=True)