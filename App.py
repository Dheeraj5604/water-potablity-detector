from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
scaler = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global model, scaler

    message = None
    error = None
    result = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                model, scaler = train_model(file_path)
                message = "Dataset uploaded and model trained successfully."
            except Exception as e:
                error = f"Error during model training: {e}"

        elif model and scaler:
            
            try:
                test_data = []
                for field in ['ph', 'hardness', 'solids', 'chloramines', 'sulfate',
                              'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']:
                    if field not in request.form or not request.form[field]:
                        raise ValueError(f"Missing or invalid input for {field}")
                    test_data.append(float(request.form[field]))

                test_data = scaler.transform([test_data])
                prediction = model.predict(xgb.DMatrix(test_data))
                prediction = int(prediction[0] > 0.5)
                result = "Water is safe to drink." if prediction == 1 else "Water is not safe to drink."
            except Exception as e:
                error = f"Prediction error: {e}"

    return render_template('index.html', message=message, error=error, result=result)


def train_model(file_path):
    data = pd.read_csv(file_path)

    if 'Potability' not in data.columns:
        raise ValueError("The dataset must contain a 'Potability' column.")

   
    data = data.fillna(data.mean())

    
    X = data.drop(columns=['Potability'])
    y = data['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'eval_metric': 'logloss'
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Model Accuracy: {accuracy}")

    return model, scaler


if __name__ == '__main__':
    app.run(debug=True)
