# Use a slim Python 3.11 image as the base
FROM python:3.11-slim

# Install pipenv and gunicorn
RUN pip install pipenv gunicorn

# Set the working directory inside the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock to the container
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies from Pipfile
RUN pipenv install --system --deploy

# Copy the application script from the utils folder D:\Data\PyCharmProjects\timeseries-neuralnets-xgboost\utils
COPY utils/predict_price_app.py ./predict_price_app.py

# Copy the model file from the model folder
COPY model/xgb_model.json ./xgb_model.json

# Expose port 9100 for the Flask app
EXPOSE 9100

# Set the command to run the app with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9100", "predict_price_app:app"]
