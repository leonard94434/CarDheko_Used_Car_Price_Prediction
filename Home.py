import openpyxl
import pandas as pd
import joblib
import streamlit as st

# Load the saved XGBoost model and the list of encoded column names
model = joblib.load("xgboost_ml_model.pkl")
encoded_columns = joblib.load("encoded_columns.pkl")

# Load the dataset to extract unique values for dropdown options in the UI
df_cars = pd.read_excel("Preprocessed_data.xlsx")

# Identify categorical columns and extract unique values for each column
categorical_columns = ['ft', 'bt', 'transmission', 'company', 'model', 
                       'Insurance Validity', 'Color', 'Location', 
                       'RTO_region', 'Drive_Type_Classified']
# Create a dictionary to store unique values for each categorical column
unique_values = {col: df_cars[col].unique().tolist() for col in categorical_columns}

# Create a dictionary to map car brands to available models (brand-model mapping)
brand_model_mapping = df_cars.groupby('company')['model'].unique().to_dict()

# Function to preprocess input data for prediction
def preprocess_input(data):
    # Map Turbo Charger values from string to boolean
    data['Turbo Charger'] = data['Turbo Charger'].map({'True': True, 'False': False})
    # One-hot encode categorical columns and ensure all columns match the trained model's structure
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    # Reindex the data to match the trained model's column structure, filling missing columns with 0
    return data_encoded.reindex(columns=encoded_columns, fill_value=0)

# Function to predict car price using the pre-trained model
def predict_price(input_data):
    # Preprocess the input data before making predictions
    processed_data = preprocess_input(input_data)
    # Predict the price using the trained model
    prediction = model.predict(processed_data)
    # Return the predicted price (single value)
    return prediction[0]

# Function to format large numbers into Indian Rupee format
def format_inr(number):
    # Partition the number into integer and decimal parts
    s, *d = str(number).partition(".")
    # Add commas for every thousandth place
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

# Streamlit app layout and logic
def main():
    # Set the app page configuration (title, icon)
    st.set_page_config(page_title="Car Price Prediction")
    
    # Enhanced CSS styling for the app
    st.markdown("""
        <style>
        body {
            background-color: #f4f7f6;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
        }
        .big-text {
            font-size: 48px;
            font-weight: bold;
            color: #007bff;
            text-align: center;
        }
        .stSelectbox, .stNumberInput {
            border: 2px solid #ff8c42;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 15px;
            width: 100%;
            background-color: #fdfdfd;
            color: #2f3542;
            font-size: 14px;
        }
        .stButton button {
            background-color: #ff8c42;
            color: white;
            border: 2px solid #ff6b35;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            width: 100%;
            cursor: pointer;
            transition: background-color 0.3s ease, border 0.3s ease;
        }
        .stButton button:hover {
            background-color: #ff6b35;
            border-color: #ff5400;
        }
        .stButton button:active {
            background-color: #ff5400;
            border-color: #e84118;
        }
        h1 {
            font-size: 48px;
            color: #4b6584;
            font-weight: 600;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 20px;
        }
        .predicted-price {
            font-size: 56px;
            font-weight: bold;
            color: #ff8c42;
            text-align: center;
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .stSelectbox, .stNumberInput {
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .block-container {
            padding: 2rem;
        }
        .css-1d391kg, .css-18e3th9 {
            border-radius: 10px;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        .stTextInput {
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            border: 2px solid #ff8c42;
            box-shadow: none;
        }
        header, footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display the logo at the top
    st.image('logo.png', width=700)
    
    # Display the main header for the app
    st.write("<h1 style='text-align: center;'>Used Car Price Prediction App</h1>", unsafe_allow_html=True)

    # Sidebar header to instruct users to enter car details
    st.sidebar.header("Enter the car details:")

    # Add a Reset button to clear inputs
    if st.sidebar.button("Reset Inputs"):
        st.session_state.clear()  # Clear all session states to reset

    # Sidebar inputs for categorical features using dropdowns
    fuel_type = st.sidebar.selectbox('Fuel Type', unique_values['ft'], index=0)
    body_type = st.sidebar.selectbox('Body Type', unique_values['bt'], index=0)
    transmission = st.sidebar.selectbox('Transmission', unique_values['transmission'], index=0)
    company = st.sidebar.selectbox('Company', unique_values['company'], index=0)
    
    # Model dropdown options depend on the selected company (brand-model mapping)
    selected_model = st.sidebar.selectbox('Model', brand_model_mapping.get(company, []), index=0)

    insurance_validity = st.sidebar.selectbox('Insurance Validity', unique_values['Insurance Validity'], index=0)
    color = st.sidebar.selectbox('Color', unique_values['Color'], index=0)
    location = st.sidebar.selectbox('Location', unique_values['Location'], index=0)
    rto_region = st.sidebar.selectbox('RTO Region', unique_values['RTO_region'], index=0)
    drive_type = st.sidebar.selectbox('Drive Type', unique_values['Drive_Type_Classified'], index=0)

    # Sidebar inputs for numerical features using sliders and number input fields
    owner_no = st.sidebar.number_input('Owner Number', min_value=1, max_value=5, value=1)
    model_year = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024, value=2022)
    km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, value=10000)
    mileage = st.sidebar.number_input('Mileage (kmpl)', min_value=0.0, value=15.0)
    engine_cc = st.sidebar.number_input('Engine Displacement (CC)', min_value=500, max_value=5000, value=1000)
    turbo_charger = st.sidebar.selectbox('Turbo Charger', ['True', 'False'])

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'ft': [fuel_type],
        'bt': [body_type],
        'transmission': [transmission],
        'company': [company],
        'model': [selected_model],
        'modelYear': [model_year],
        'km': [km_driven],
        'Insurance Validity': [insurance_validity],
        'Mileage': [mileage],
        'Color': [color],
        'Displacement': [engine_cc],
        'Turbo Charger': [turbo_charger],
        'Location': [location],
        'RTO_region': [rto_region],
        'Drive_Type_Classified': [drive_type],
        'ownerNo': [owner_no]
    })

    # When the user clicks the "Predict Price" button
    if st.sidebar.button("Predict Price"):
        # Make the price prediction using the pre-trained model
        predicted_price = predict_price(input_data)
        # Format the predicted price into INR format
        formatted_price = format_inr(predicted_price)
        
        # Display the predicted price with styling
        st.write("<p class='predicted-price'>Predicted Price: ₹ {}</p>".format(formatted_price), unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
