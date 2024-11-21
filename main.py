import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
df = pd.read_csv('/content/cars.csv')  # Replace with your dataset path

# Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type']
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Save the encoder for future use

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features (X) and target (y)
X = df.drop(columns=['Price', 'Car_ID'])
y = df['Price']

# Save the feature column names
feature_columns = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Save the trained model and preprocessing tools
joblib.dump(model, 'car_price_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')  # Save feature columns
print("Model and preprocessing tools saved successfully.")

# Function to preprocess user input and make predictions
def predict_car_price_interactive():
    """
    Take user inputs interactively and predict the price of a car.
    """
    # Load preprocessing tools
    loaded_model = joblib.load('car_price_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')  # Load feature columns

    print("Enter the car details below:")
    year = int(input("Year (e.g., 2019): "))
    kilometers_driven = int(input("Kilometers Driven (e.g., 40000): "))
    mileage = float(input("Mileage (km/l) (e.g., 17.0): "))
    engine = int(input("Engine (CC) (e.g., 1597): "))
    power = float(input("Power (bhp) (e.g., 140): "))
    seats = int(input("Seats (e.g., 5): "))
    brand = input("Brand (e.g., Honda): ")
    model_name = input("Model (e.g., Civic): ")
    fuel_type = input("Fuel Type (e.g., Petrol): ")
    transmission = input("Transmission (e.g., Automatic): ")
    owner_type = input("Owner Type (e.g., First): ")

    # Preprocess user input
    user_input = {
        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
        "Brand": brand,
        "Model": model_name,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
    }

    # Encode categorical features
    for col in categorical_columns:
        le = le_dict[col]
        if user_input[col] not in le.classes_:
            print(f"Warning: '{user_input[col]}' is an unseen category for {col}. Using default.")
            user_input[col] = le.transform([le.classes_[0]])[0]  # Default to the first class
        else:
            user_input[col] = le.transform([user_input[col]])[0]

    # Create a DataFrame with feature names in the correct order
    features = [[
        user_input['Year'],
        user_input['Kilometers_Driven'],
        user_input['Mileage'],
        user_input['Engine'],
        user_input['Power'],
        user_input['Seats'],
        user_input['Brand'],
        user_input['Model'],
        user_input['Fuel_Type'],
        user_input['Transmission'],
        user_input['Owner_Type']
    ]]
    features_df = pd.DataFrame(features, columns=feature_columns)  # Use the correct feature order

    # Scale numerical features
    features_df[numerical_features] = scaler.transform(features_df[numerical_features])

    # Predict the price
    predicted_price = loaded_model.predict(features_df)
    print(f"\nThe predicted selling price of the car is: PKR {predicted_price[0]:.2f}")

# Call the function to interact with the user
predict_car_price_interactive()
