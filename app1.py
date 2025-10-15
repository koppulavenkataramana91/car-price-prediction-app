import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved model and data as described above
df = pickle.load(open('car_df.pkl', 'rb'))  # Your original DataFrame
pipe = pickle.load(open('car_pipe.pkl', 'rb'))  # Your trained pipeline

st.title("Car Price Predictor App")

# Inputs from user for prediction
company_name = st.selectbox("Company Name", df['company name'].unique())
model = st.selectbox("Model", df['Model'].unique())
fuel_type = st.selectbox("Fuel Type", df['Fuel Type'].unique())
transmission = st.selectbox("Transmission", df['Transmission'].unique())
location = st.selectbox("Location", df['Location'].unique())
color = st.selectbox("Color", df['Color'].unique())
owner = st.selectbox("Owner", df['Owner'].unique())
seller_type = st.selectbox("Seller Type", df['Seller Type'].unique())
year = st.slider("Year", int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].median()))
kilometer = st.number_input("Kilometer Driven", min_value=0, max_value=int(df['Kilometer'].max()), value=10000, step=1000)
seating_capacity = st.slider("Seating Capacity", int(df['Seating Capacity'].min()), int(df['Seating Capacity'].max()), int(df['Seating Capacity'].median()))

if st.button("PREDICT PRICE"):
    # Build input as DataFrame (correct format!)
    input_df = pd.DataFrame([{
        'company name': company_name,
        'Model': model,
        'Year': year,
        'Kilometer': kilometer,
        'Fuel Type': fuel_type,
        'Transmission': transmission,
        'Location': location,
        'Color': color,
        'Owner': owner,
        'Seller Type': seller_type,
        'Seating Capacity': seating_capacity
    }])

    # Predict
    try:
        price_pred = pipe.predict(input_df)
        final_price = int(round(price_pred[0], -3))  # Rounded to nearest 1000
        st.subheader(f"Estimated Price: ₹{final_price}")
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")
