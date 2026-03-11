import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

try:
    with open("model_pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)

    with open("clean_data.pkl", "rb") as f:
        df = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

st.title("💻 Laptop Price Predictor")
st.write("Enter laptop configuration to predict the price.")

brand = st.selectbox("Brand", df['Company'].unique())

type_name = st.selectbox("Type", df['TypeName'].unique())

ram = st.selectbox("RAM (GB)", [2,4,8,12,16,24,32,64])

weight = st.number_input(
    "Weight of Laptop (kg)",
    min_value=0.5,
    max_value=5.0,
    step=0.1
)

touchscreen = st.selectbox("Touchscreen", ["No","Yes"])

ips = st.selectbox("IPS Display", ["No","Yes"])

screen_size = st.number_input(
    "Screen Size (inches)",
    min_value=10.0,
    max_value=20.0,
    step=0.1
)

resolution = st.selectbox(
    "Screen Resolution",
    ['1366x768','1600x900','1920x1080','2160x1440','2560x1440','3840x2160']
)

cpu = st.selectbox("CPU Brand", df['Cpu Brand'].unique())

hdd = st.selectbox("HDD (GB)", [0,128,256,512,1024,2048])

ssd = st.selectbox("SSD (GB)", [0,128,256,512,1024])

gpu = st.selectbox("GPU Brand", df['Gpu Brand'].unique())

os = st.selectbox("Operating System", df['Os'].unique())

if st.button("Predict Price"):

    try:

        if screen_size <= 0:
            st.warning("Screen size must be greater than 0.")
            st.stop()

        if weight <= 0:
            st.warning("Weight must be greater than 0.")
            st.stop()

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])

        ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

        touchscreen_val = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        query = pd.DataFrame({
            'Company':[brand],
            'TypeName':[type_name],
            'Ram':[ram],
            'Weight':[weight],
            'Touchscreen':[touchscreen_val],
            'Ips':[ips_val],
            'ppi':[ppi],
            'Cpu Brand':[cpu],
            'HDD':[hdd],
            'SSD':[ssd],
            'Gpu Brand':[gpu],
            'Os':[os]
        })

        prediction = pipe.predict(query)[0]

        price = int(np.exp(prediction))

        st.success(f"💰 Predicted Laptop Price: ₹ {price:,}")

    except ValueError:
        st.error("Invalid numeric input. Please check your values.")

    except KeyError:
        st.error("Input feature mismatch with trained model.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")