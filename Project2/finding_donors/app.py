
import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# Get project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved models and preprocessing information
models_bundle = joblib.load(
    os.path.join(BASE_DIR, 'models_bundle.pkl')
)

models = models_bundle['models']
scaler = models_bundle['scaler']
feature_columns = models_bundle['feature_columns']
numerical_columns = models_bundle['numerical_columns']
skewed_columns = models_bundle['skewed_columns']


@app.route('/', methods=['GET', 'POST'])
def home():

    predictions = None

    if request.method == 'POST':

        # 1. Get input data from the HTML form
        user_data = {
            'age': int(request.form['age']),
            'workclass': request.form['workclass'],
            'education_level': request.form['education_level'],
            'education-num': int(request.form['education_num']),
            'marital-status': request.form['marital_status'],
            'occupation': request.form['occupation'],
            'relationship': request.form['relationship'],
            'race': request.form['race'],
            'sex': request.form['sex'],
            'capital-gain': int(request.form['capital_gain']),
            'capital-loss': int(request.form['capital_loss']),
            'hours-per-week': int(request.form['hours_per_week']),
            'native-country': request.form['native_country']
        }

        input_df = pd.DataFrame([user_data])

        # 2. Apply log transformation to skewed features
        input_df[skewed_columns] = np.log1p(
            input_df[skewed_columns]
        )

        # 3. Scale numerical features
        input_df[numerical_columns] = scaler.transform(
            input_df[numerical_columns]
        )

        # 4. Apply One-Hot Encoding
        input_df = pd.get_dummies(input_df)

        # 5. Match the training features
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # 6. Predict using all three models
        predictions = {}

        for name, model in models.items():

            # Get prediction
            prediction = model.predict(input_df)[0]

            # Calculate confidence for the predicted class
            probability = None

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]

                predicted_class_index = list(model.classes_).index(
                    prediction
                )

                probability = round(
                    probabilities[predicted_class_index] * 100, 2
                )

            # Store results
            predictions[name] = {
                'prediction': '>50K' if prediction == 1 else '<=50K',
                'probability': probability
            }

    return render_template(
        'index.html',
        predictions=predictions
    )


if __name__ == '__main__':
    app.run(debug=True)
