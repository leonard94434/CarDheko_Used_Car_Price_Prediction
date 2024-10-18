# CarDheko_Used_Car_Price_Prediction

This repository contains the code for the **Used Car Price Prediction** software, which uses machine learning models to predict the price of a used car based on its features such as model, fuel type, body type, mileage, transmission, and location. The software is built using Python, and the web app is powered by **Streamlit** for real-time price prediction.

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Streamlit Web App Overview](#streamlit-web-app-overview)
7. [How the Prediction Works](#how-the-prediction-works)
8. [Mathematical and Computational Concepts](#mathematical-and-computational-concepts)
9. [Understanding the User Interface](#understanding-the-user-interface)
10. [Advanced Topics](#advanced-topics)
11. [FAQs](#faqs)
12. [Process Summary](#process-summary)

---

## 1. Introduction

The **Used Car Price Prediction** software is designed to predict the price of a used car based on various factors like car model, fuel type, body type, mileage, transmission, and location. It uses machine learning techniques, primarily **XGBoost** and **Random Forest**, to analyze past car sales and make price predictions.

This README provides instructions for setting up and running the app, as well as additional details on the machine learning process, data preprocessing, and deployment with Streamlit.

---

## 2. System Requirements

Ensure that your system meets the following requirements:

- **Python version**: 3.7 or higher
- **Required Libraries**:
  - `streamlit`
  - `pandas`
  - `joblib`
  - `openpyxl`
  - `xgboost`
  - `scikit-learn`
  - `matplotlib`
  - `numpy`

---

## 3. Installation Guide

Follow these steps to set up the software:

1. **Install Python**: 
   - Download and install the latest version of Python from [python.org](https://www.python.org/downloads/).

2. **Set up a virtual environment** *(optional but recommended)*:
   - Create a virtual environment using the command:
     ```bash
     python -m venv car_prediction_env
     ```
   - Activate the environment:
     - On Windows:
       ```bash
       car_prediction_env\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source car_prediction_env/bin/activate
       ```

3. **Install the required libraries**:
   - Run the following command to install the necessary Python libraries:
     ```bash
     pip install streamlit pandas joblib openpyxl xgboost scikit-learn matplotlib numpy
     ```

4. **Run the application**:
   - Navigate to the folder containing the app and run the following command to start the app:
     ```bash
     streamlit run app.py
     ```

---

## 4. Data Preprocessing

Data preprocessing is an important step where the raw data is cleaned and transformed. The key steps involved are:

- **Data Cleaning**: Remove columns with over 50% missing values and duplicates. Outliers are removed using the **Z-score** method.
- **Feature Engineering**: Use **One-Hot Encoding** to convert categorical variables (e.g., fuel type, body type) into numerical formats that machine learning models can understand.
- **Handling Categorical Data**: Convert features like 'Fuel Type', 'Body Type', and 'Drive Type' into numerical representations.
- **Final Dataset**: After preprocessing, the dataset is cleaned and ready for training the machine learning models.

---

## 5. Model Training and Evaluation

The software uses machine learning models like **Random Forest** and **XGBoost** to predict used car prices.

### Steps involved:

1. **Training the Model**:
   - Train Random Forest and XGBoost models on the cleaned dataset using **Supervised Learning**.
   - Use **Cross-validation** to ensure the model generalizes well to unseen data.

2. **Hyperparameter Tuning**:
   - Apply **Random Search Cross-Validation (CV)** to optimize model parameters.

3. **Evaluation Metrics**:
   - **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, **R-Squared (R²)**, and **Mean Absolute Percentage Error (MAPE)** are used to evaluate the model’s performance.

---

## 6. Streamlit Web App Overview

The **Used Car Price Prediction** app uses **Streamlit**, a Python-based framework to create web apps with minimal code. The app allows users to input car details and get real-time price predictions.

### Key Features:
- **Categorical Inputs**: Dropdown menus to select car features like fuel type, body type, and color.
- **Numerical Inputs**: Sliders and number inputs for engine displacement, mileage, and kilometers driven.
- **Reset Button**: Allows users to reset inputs to default values.

---

## 7. How the Prediction Works

Once the user enters the car details, the following steps occur:

1. **Input Handling**: The app collects the user inputs such as fuel type, body type, engine capacity, and other details.
2. **Preprocessing**: Input data is preprocessed using the same steps as the training data (e.g., One-Hot Encoding for categorical variables).
3. **Prediction**: The preprocessed data is passed to the trained **XGBoost** model, which makes a price prediction based on the learned patterns.
4. **Result Display**: The predicted price is displayed in **Indian Rupees (INR)**, formatted with commas for better readability.

---

## 8. Mathematical and Computational Concepts

Here are some key computational and mathematical concepts used in the software:

- **One-Hot Encoding**: Converts categorical variables into binary columns.
- **Supervised Learning**: The model is trained on labeled data (input features and car price).
- **Cross-Validation**: The data is split into training and validation sets multiple times to evaluate model performance.
- **Random Search Cross-Validation**: A method for hyperparameter tuning that searches through random combinations of parameters.

---

## 9. Understanding the User Interface

### Sidebar:
- **Dropdown Menus**: Used to select car details like fuel type, body type, and location.
- **Number Inputs**: Used to enter values like kilometers driven, mileage, and engine displacement.

### Main Page:
- **Predict Price Button**: Clicking this button triggers the price prediction process.
- **Predicted Price Display**: The predicted price of the car is displayed in Indian Rupees (INR).

---

## 10. Advanced Topics

- **Feature Importance**: Visualizes the most important features that influence the car price prediction. This can be done using feature importance plots.
- **Model Saving**: Models are saved using the **joblib** library, allowing reuse without retraining.
- **Future Improvements**: Potential enhancements include integrating deep learning models for better accuracy and adding user feedback mechanisms to improve model predictions over time.

---

## 11. FAQs

**Q1: How is the price prediction made?**

- The price is predicted using machine learning models trained on thousands of past car sales records.

**Q2: How accurate is the price prediction?**

- The model has been evaluated using metrics like **Mean Squared Error (MSE)** and **R-Squared (R²)**, showing strong performance on test data.

**Q3: Can I use this app for any car model?**

- The app supports a wide range of car models and brands, provided they are present in the dataset.

---

## 12. Process Summary

This section summarizes the process we followed while creating this documentation and software.

- **JSON Parsing**: We generated JSON-based instructions for processing data and model evaluation. We utilized tools like **XGBoost**, **Random Forest**, and **Streamlit** to make price predictions based on inputs.
- **Documentation**: Created a comprehensive guide on how to install and run the software, including steps on how to preprocess the data, train the model, and evaluate performance.
- **Streamlit Guide**: Streamlit is used to create the app interface where users can input details and get real-time price predictions.
- **PDF Generation**: We used a custom tool to generate a PDF version of the complete guide for distribution and user reference.

---

## How to Use the App (Step-by-Step)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/used-car-price-prediction.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Input car details** using the sidebar in the app.

5. **Click the "Predict Price" button** and the app will display the predicted price based on the machine learning model.

---

If you have any questions or encounter issues, feel free to open an issue in this repository or contact the repository maintainers.

Happy predicting! 🚗💡

---
