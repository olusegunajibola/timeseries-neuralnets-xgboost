import requests

url = "http://localhost:9100/bnbpriceprediction"
price_attributes = {
    "lag_1": 320.1,
    "lag_2": 315.5,
    "lag_3": 310.2,
    "rolling_mean_3": 315.27,
    "rolling_std_3": 5.04,
    "rolling_mean_7": 318.54,
}

response = requests.post(url, json=price_attributes)
print(response.json())
