from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['Pclass']),
            int(request.form['Sex']),
            float(request.form['Age']),
            int(request.form['SibSp']),
            int(request.form['Parch']),
            float(request.form['Fare']),
            int(request.form['Embarked']),
            int(request.form['FamilySize']),
            int(request.form['IsAlone']),
            int(request.form['Title'])
        ]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        output = "Survived" if prediction[0] == 1 else "Did Not Survive"
        return render_template('index.html', prediction_text=f"The passenger would have: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)