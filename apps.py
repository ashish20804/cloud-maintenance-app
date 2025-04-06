#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# -------------------------------
# Email Alert Function
# -------------------------------
def send_email_alert(subject, body, to):
    try:
        email = EmailMessage()
        email.set_content(body)
        email['Subject'] = subject
        email['From'] = "your_email@gmail.com"         # Change this
        email['To'] = to

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("your_email@gmail.com", "your_password")  # Use app password or secure way
        server.send_message(email)
        server.quit()
        st.success("Alert email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# -------------------------------
# Simulate IoT Data
# -------------------------------
def generate_fake_data(rows=100):
    data = {
        'temperature': np.random.normal(50, 10, rows),
        'vibration': np.random.normal(1.5, 0.5, rows),
        'pressure': np.random.normal(100, 20, rows),
    }
    df = pd.DataFrame(data)
    df['failure'] = ((df['temperature'] > 65) | (df['vibration'] > 2.5) | (df['pressure'] > 140)).astype(int)
    return df

# -------------------------------
# Machine Learning Model
# -------------------------------
def train_model(df):
    X = df[['temperature', 'vibration', 'pressure']]
    y = df['failure']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# -------------------------------
# GUI Layout
# -------------------------------
st.title("🔧 Cloud-Based Predictive Maintenance System")
st.markdown("Monitor and Predict Equipment Failures with Real-Time Data and Alerts")

# Sidebar Controls
st.sidebar.header("📡 Simulated Sensor Input")
temp = st.sidebar.slider("Temperature (°C)", 20.0, 100.0, 55.0)
vib = st.sidebar.slider("Vibration (g)", 0.0, 5.0, 1.0)
pres = st.sidebar.slider("Pressure (kPa)", 50.0, 200.0, 110.0)

# Load or Generate Data
df = generate_fake_data()
model = train_model(df)

# Prediction
input_data = pd.DataFrame([[temp, vib, pres]], columns=['temperature', 'vibration', 'pressure'])
prediction = model.predict(input_data)[0]

# Display Input
st.subheader("📊 Input Sensor Readings")
st.write(input_data)

# Result
st.subheader("⚠️ Prediction Result")
if prediction == 1:
    st.error("Failure Likely Detected! 🚨")
    if st.button("Send Alert Email"):
        send_email_alert(
            subject="⚠️ Predictive Maintenance Alert!",
            body=f"Failure predicted at:\nTemperature: {temp}\nVibration: {vib}\nPressure: {pres}",
            to="receiver_email@example.com"  # Change this
        )
else:
    st.success("System is Healthy ✅")

# Plotting Historical Trends
st.subheader("📈 Historical Data Trend")
st.line_chart(df[['temperature', 'vibration', 'pressure']])

# About Section
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    - **Cloud Storage:** You can link this with AWS/GCP using Boto3/BigQuery client.
    - **Real IoT Data:** Replace simulated data with data from AWS IoT Core or Google IoT.
    - **Alerting:** Integrate Twilio for SMS or use AWS SNS for scalable alerts.
    """)



# In[ ]:





# In[ ]:




