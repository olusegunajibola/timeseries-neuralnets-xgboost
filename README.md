# Binance Coin (BNB) Price Prediction Flask App
A project that applies LSTM (PyTorch) and XGBoost to time series prediction of BNB cryptocurrency price.

This project is a Flask-based web application that predicts the price of Binance Coin (BNB) using a trained XGBoost regression model. The app takes price-related features as input and returns a prediction of the BNB price.


View the deployed model at [https://bit.ly/deploy-time-series](https://bit.ly/deploy-time-series)
## Features
- **ML Model**: A pre-trained XGBoost regression model for BNB price prediction.
- **REST API**: A POST endpoint for submitting price attributes and receiving predictions.
- **Dockerized Deployment**: Run the application in a containerized environment for ease of deployment and scalability.

## File Structure
```plaintext
.
├── data
    └── BNB-USD.csv               # Dataset from Yahoo Finance
├── model/
│   └── xgb_model.json            # Pre-trained XGBoost model
├── notebooks/
│   └── 01_model_training.ipynb   # Notebook to train model
├── utils/
│   └── predict_price_app.py      # Main Flask application
│   └── test_predict.py           # Test Flask application
├── Pipfile                       # Python dependencies
├── Pipfile.lock                  # Locked dependencies
├── Dockerfile                    # Docker build instructions
├── info.txt                      # Instructions to run Pipenv and Docker
├── README.md                     # Project documentation
```

## Getting Started
### Prerequisites
- **Python 3.11**
- **Pipenv**
- **Docker**

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/olusegunajibola/timeseries-neuralnets-xgboost.git
   cd bnb-price-predictor
   ```
2. Install dependencies using `pipenv`:
   ```bash
   pipenv install --system --deploy
   ```

### Running the App Locally
1. Navigate to the project directory.
2. Start the Flask app:
   ```bash
   python utils/predict_price_app.py
   ```
3. The app will run on `http://127.0.0.1:9100` by default.

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t bnb-price-predictor .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 9100:9100 bnb-price-predictor
   ```
3. The app will be available at `http://localhost:9100`.

## Usage
### API Endpoint
#### URL
`POST /bnbpriceprediction`

#### Request Body
Send a JSON object with the following keys:
```json
{
  "lag_1": 320.1,
  "lag_2": 315.5,
  "lag_3": 310.2,
  "rolling_mean_3": 315.27,
  "rolling_std_3": 5.04,
  "rolling_mean_7": 318.54
}
```

#### Response
```json
{
  "new_prediction": 317.45
}
```

### Example Usage
```python
import requests

url = 'http://localhost:9100/bnbpriceprediction'
price_attributes = {
    "lag_1": 320.1,
    "lag_2": 315.5,
    "lag_3": 310.2,
    "rolling_mean_3": 315.27,
    "rolling_std_3": 5.04,
    "rolling_mean_7": 318.54
}

response = requests.post(url, json=price_attributes).json()
print(response)
```

## Troubleshooting
- **Model File Not Found**: Ensure the `xgb_model.json` file is in the `model/` directory.
- **Port Conflicts**: Change the `-p` flag in the `docker run` command if port `9100` is already in use.
- **Dependency Issues**: Update the `Pipfile` with compatible versions of `xgboost` and `scikit-learn`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Contact
For questions or support, contact [olusegunajibola.e@gmail.com].



