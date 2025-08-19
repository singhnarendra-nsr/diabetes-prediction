from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load saved scaler and model
scaler = pickle.load(open('Scaler.pkl', 'rb'))
model = pickle.load(open('SVM_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form values
            pregs = int(request.form.get('pregs'))
            gluc = int(request.form.get('gluc'))
            bp = int(request.form.get('bp'))
            skin = int(request.form.get('skin'))
            insulin = float(request.form.get('insulin'))
            bmi = float(request.form.get('bmi'))
            func = float(request.form.get('func'))
            age = int(request.form.get('age'))

            # Prepare input for model
            input_features = np.array([[pregs, gluc, bp, skin, insulin, bmi, func, age]])
            scaled_features = scaler.transform(input_features)

            # Make prediction
            prediction = model.predict(scaled_features)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
