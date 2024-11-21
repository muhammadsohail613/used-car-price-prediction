import streamlit as st
import requests
import base64

# Function to set a background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")  # Encode image to Base64
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
        }}
        .custom-title {{
            color: red;
            background-color: rgba(255, 255, 255, 0.5);
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            font-weight: bold;
        }}
        .custom-input {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 10px;
            border-radius: 5px;
        }}
        .custom-output {{
            background-color: rgba(50, 50, 50, 0.8);
            color: green;
            text-align: center;
            padding: 20px;
            border-radius: 20px;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("background.jpg")  # Replace with your image file

# Custom title
st.markdown('<h1 class="custom-title">Used Car Price Prediction</h1>', unsafe_allow_html=True)

# Input form with custom styling
st.sidebar.markdown('<div class="custom-input">Enter Car Details</div>', unsafe_allow_html=True)
year = st.sidebar.number_input("Year (e.g., 2019)", min_value=1980, max_value=2023, step=1, value=2019)
kilometers_driven = st.sidebar.number_input("Kilometers Driven (e.g., 30000)", min_value=0, step=1000, value=30000)
mileage = st.sidebar.number_input("Mileage (km/l) (e.g., 17.0)", min_value=0.0, step=0.1, value=17.0)
engine = st.sidebar.number_input("Engine (CC) (e.g., 1597)", min_value=500, max_value=10000, step=50, value=1597)
power = st.sidebar.number_input("Power (bhp) (e.g., 140)", min_value=50.0, max_value=1000.0, step=1.0, value=140.0)
seats = st.sidebar.number_input("Seats (e.g., 5)", min_value=2, max_value=10, step=1, value=5)
brand = st.sidebar.text_input("Brand (e.g., Honda)", value="Honda")
model_name = st.sidebar.text_input("Model (e.g., Civic)", value="Civic")
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "LPG"], index=0)
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"], index=1)
owner_type = st.sidebar.selectbox("Owner Type", ["First", "Second", "Third", "Fourth or More"], index=0)

# Predict button
if st.sidebar.button("Predict Price"):
    # Create input data
    input_data = {
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
        "Owner_Type": owner_type
    }

    # Send POST request to FastAPI
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.markdown(
                f'<div class="custom-output">The predicted selling price of the car is: PKR {result["predicted_price"]:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
