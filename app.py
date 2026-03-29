from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

# --- Train/load model ---
if not os.path.exists('loan_model.pkl'):
    df = pd.read_csv('loan_data.csv')

    # IQR for Income
    Q1 = df['Income'].quantile(0.25)
    Q3 = df['Income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Income'] >= lower_bound) & (df['Income'] <= upper_bound)]

    # Train model
    X = df[['Income', 'Loan_Amount', 'Credit_History']]
    y = df['Loan_Status']
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

    # Save model
    pickle.dump(model, open('loan_model.pkl', 'wb'))
else:
    model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    # Default values
    name_val = ""
    income_val = ""
    loan_val = ""
    history_val = ""

    if request.method == "POST":
        name_val = request.form["name"]
        income_val = request.form["income"]
        loan_val = request.form["loan"]
        history_val = request.form["history"]

        try:
            name = name_val
            income = float(income_val)
            loan = float(loan_val)
            history = int(history_val)

            if history not in [0, 1]:
                result = "Credit History must be 0 or 1"
            else:
                input_df = pd.DataFrame([[income, loan, history]],
                                        columns=['Income', 'Loan_Amount', 'Credit_History'])
                prediction = model.predict(input_df)[0]
                result = f"Loan Status for {name}: {prediction}"
        except ValueError:
            result = "Please enter valid numeric values"

    return render_template("index.html",
                           result=result,
                           name_val=name_val,
                           income_val=income_val,
                           loan_val=loan_val,
                           history_val=history_val)

if __name__ == "__main__":
    app.run(debug=True)