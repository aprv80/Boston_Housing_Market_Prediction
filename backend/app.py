
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask("Boston Housing Price Prediction")

# Load the trained model
loaded_model = joblib.load('xgb_model_boston_v1.0')

@app.get('/')
def home():
    return 'Boston Housing Price Prediction'

@app.post('/v1/single')

def predict_house_price():
    house_data = request.get_json()
    data = {
        'CRIM': house_data['CRIM'],
        'ZN': house_data['ZN'],
        'INDUS': house_data['INDUS'],
        'CHAS': house_data['CHAS'],
        'NOX': house_data['NOX'],
        'RM': house_data['RM'],
        'AGE': house_data['AGE'],
        'DIS': house_data['DIS'],
        'RAD': house_data['RAD'],
        'TAX': house_data['TAX'],
        'PTRATIO': house_data['PTRATIO'],
        'LSTAT': house_data['LSTAT'],
    }
    input_data = pd.DataFrame([data])
    predicted_price = loaded_model.predict(input_data)
    return jsonify({'predicted_price': float(predicted_price[0])})

def predict_batch_price():
    file = request.files['file']
    df = pd.read_csv(file)
    predicted_prices = loaded_model.predict(df).tolist()
    return jsonify({'predicted_prices': predicted_prices})

if __name__ == '__main__':
    app.run(debug=True)
    return jsonify({'predicted_prices': predicted_prices.tolist()})
