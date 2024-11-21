from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the app
app = FastAPI()

# Load the saved model and preprocessing tools
model = joblib.load('car_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Define numerical and categorical features
numerical_features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Define input schema
class CarFeatures(BaseModel):
    Year: int
    Kilometers_Driven: int
    Mileage: float
    Engine: int
    Power: float
    Seats: int
    Brand: str
    Model: str
    Fuel_Type: str
    Transmission: str
    Owner_Type: str

# API root endpoint
@app.get("/")
def root():
    return {"message": "Car Price Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: CarFeatures):
    # Convert input data to a dictionary
    user_input = features.dict()

    # Encode categorical features
    for col in categorical_columns:
        le = label_encoders[col]
        if user_input[col] not in le.classes_:
            user_input[col] = le.transform([le.classes_[0]])[0]  # Default to the first class
        else:
            user_input[col] = le.transform([user_input[col]])[0]

    # Create a DataFrame with the correct feature order
    features_data = pd.DataFrame([[user_input[col] for col in feature_columns]], columns=feature_columns)

    # Scale numerical features
    features_data[numerical_features] = scaler.transform(features_data[numerical_features])

    # Make prediction
    predicted_price = model.predict(features_data)

    return {
        "predicted_price": round(predicted_price[0], 2),
        "currency": "INR"
    }
