from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the Model
model = joblib.load("mental_health_model.joblib")

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/form")
def form_page():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Collect form data
    input_data = pd.DataFrame([{
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Country": request.form["Country"],
        "self_employed": request.form["self_employed"],
        "family_history": request.form["family_history"],
        "work_interfere": request.form["work_interfere"],
        "no_employees": request.form["no_employees"],
        "remote_work": request.form["remote_work"],
        "benefits": request.form["benefits"],
        "care_options": request.form["care_options"],
        "wellness_program": request.form["wellness_program"],
        "anonymity": request.form["anonymity"],
        "leave": request.form["leave"],
        "coworkers": request.form["coworkers"],
        "supervisor": request.form["supervisor"],
    }])

    # Predictions
    pred = model.predict(input_data)[0]
    # prob = model.predict_proba(input_data)[0][1]

    # Final refined message
    if pred == 1:
        insight = [
        "Your responses suggest that you may benefit from a little extra wellbeing support.",
        "Trying small habits—like short breaks, light movement, or speaking with someone you trust—can make a positive difference.",
        "Exploring available workplace resources or support options may also offer helpful guidance."
        ]
    else:
        insight = [
        "Your responses reflect a generally steady wellbeing outlook.",
        "Staying aware of available support resources can still be beneficial whenever you need them.",
        "Maintaining healthy routines and open communication can continue to support your overall wellbeing."
        ]

    return render_template(
        "result.html",
        insight=insight,
        # probability=f"{prob:.2f}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

