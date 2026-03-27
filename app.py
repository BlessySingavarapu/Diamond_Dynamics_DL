import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("D:/Python_Code/Projects_In_Course_Time/Project-4_Diamond_Dynamics/best_model.pkl", "rb"))
kmeans = pickle.load(open("D:/Python_Code/Projects_In_Course_Time/Project-4_Diamond_Dynamics/kmeans_model.pkl", "rb"))
encoder = pickle.load(open("D:/Python_Code/Projects_In_Course_Time/Project-4_Diamond_Dynamics/encoder.pkl", "rb"))
scaler = pickle.load(open("D:/Python_Code/Projects_In_Course_Time/Project-4_Diamond_Dynamics/scaler.pkl", "rb"))

# =========================
# FEATURE ORDER (VERY IMPORTANT)
# =========================
selected_features = [
    'carat',
    'surface_area',
    'x',
    'y',
    'clarity',
    'color',
    'z',
    'symmetry',
    'volume',
    'density',
    'cut',
    'depth_ratio'
]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("💎 Diamond App")

option = st.sidebar.radio(
    "Select Module",
    ["Price Prediction", "Market Segmentation"]
)

# =========================
# USER INPUTS
# =========================
st.sidebar.header("Enter Diamond Details")

carat = st.sidebar.number_input("Carat", min_value=0.0, step=0.1)
x = st.sidebar.number_input("Length (x)", min_value=0.0)
y = st.sidebar.number_input("Width (y)", min_value=0.0)
z = st.sidebar.number_input("Depth (z)", min_value=0.0)

cut = st.sidebar.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'])
color = st.sidebar.selectbox("Color", ['J','I','H','G','F','E','D'])
clarity = st.sidebar.selectbox("Clarity", ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])

# =========================
# INPUT PREPARATION
# =========================
def prepare_input():

    # -------------------------
    # CREATE carat_category ✅
    # -------------------------
    if carat < 0.5:
        carat_category = "Light"
    elif carat < 1.5:
        carat_category = "Medium"
    else:
        carat_category = "Heavy"

    # Encode categorical features
    cat_input = encoder.transform([[cut, color, clarity, carat_category]])
    cut_enc, color_enc, clarity_enc, carat_category_enc = cat_input[0]

    # Derived features
    volume = x * y * z
    density = carat / volume if volume != 0 else 0
    surface_area = 2 * (x*y + y*z + x*z)
    symmetry = x / y if y != 0 else 0
    depth_ratio = z / carat
    # price_inr_per_carat = 0

    # Create input dictionary
    input_dict = {
        'carat': carat,
        'surface_area': surface_area,
        'x': x,
        'y': y,
        'clarity': clarity_enc,
        'color': color_enc,
        'z': z,
        'symmetry': symmetry,
        'volume': volume,       
        'density': density,
        'cut': cut_enc,
        'depth_ratio' : depth_ratio
    }

    # Maintain column order
    input_df = pd.DataFrame([input_dict])[selected_features]

    return input_df

# =========================
# PRICE PREDICTION
# =========================
if option == "Price Prediction":
    st.title("💰 Diamond Price Prediction")

    if st.button("Predict Price"):
        input_df = prepare_input()

        prediction = model.predict(input_df)

        price_inr = prediction[0]

        #price_inr = price_inr * 83 

        st.success(f"💰 Predicted Price: ₹ {price_inr:,.2f}")

# =========================
# CLUSTER PREDICTION
# =========================
elif option == "Market Segmentation":
    st.title("📊 Diamond Market Segment")

    if st.button("Predict Cluster"):
        input_df = prepare_input()

        # Scale for clustering
        input_scaled = scaler.transform(input_df)

        cluster = kmeans.predict(input_scaled)[0]

        cluster_names = {
            0: "💰 Affordable Small Diamonds",
            1: "⚖️ Mid-range Balanced Diamonds",
            2: "💎 Premium Heavy Diamonds"
        }

        st.success(f"Cluster: {cluster}")
        st.info(f"Segment: {cluster_names.get(cluster)}")
