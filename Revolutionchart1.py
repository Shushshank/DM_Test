import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('my_model.h5')

st.title("Revolution Cart Revenue Prediction")

# Input features
st.header("Enter Store Details:")
avg_order_value = st.number_input("Average Order Value", min_value=0.0)
avg_orders_per_day = st.number_input("Average Orders Per Day", min_value=0.0)
customer_lifetime_value = st.number_input("Customer Lifetime Value", min_value=0.0)
total_customers = st.number_input("Total Customers", min_value=0.0)
conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0)


# Create a button to trigger prediction
if st.button("Predict Monthly Revenue"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'avg_order_value': [avg_order_value],
        'avg_orders_per_day': [avg_orders_per_day],
        'customer_lifetime_value': [customer_lifetime_value],
        'total_customers': [total_customers],
        'conversion_rate': [conversion_rate]
    })

    # Make prediction using the loaded model
    predicted_revenue = model.predict(input_data)[0][0]

    # Display the predicted revenue
    st.header("Predicted Monthly Revenue:")
    st.write(f"${predicted_revenue:.2f}")
