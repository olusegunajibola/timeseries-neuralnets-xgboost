# import pickle
import xgboost as xgb
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

model = '../model/xgb_model.json'
# model = 'xgb_model.json'

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model)
print("Model loaded successfully")

app = Flask('bnbpriceprediction')

@app.route('/bnbpriceprediction', methods=['POST'])
def predict():

    price_attributes = request.get_json()
    features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3", "rolling_mean_7"]
    X_new = np.array([[price_attributes[feature] for feature in features]])

    # Predict using the loaded model
    new_prediction = xgb_model.predict(X_new)

    result = {
        'new_prediction': float(new_prediction[0])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9100)