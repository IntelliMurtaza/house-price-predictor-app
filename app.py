import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# -------------------------------
# Dark Mode Toggle
# -------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    bg_color = "#0e1117"
    text_color = "white"
    card_color = "#161b22"
else:
    bg_color = "#f5f7fa"
    text_color = "black"
    card_color = "white"

# -------------------------------
# Styling
# -------------------------------
st.markdown(f"""
    <style>
    .main {{
        background-color: {bg_color};
        color: {text_color};
    }}


    .card {{
        background-color: {card_color};
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid rgba(0,0,0,0.05);
    }}
    
    /* 🔥 REMOVE Streamlit input background */
    div[data-baseweb="input"] > div {{
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    
    div[data-baseweb="select"] > div {{
        background-color: transparent !important;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 18px;
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("House_Price_Prediction.pkl")
    cols = joblib.load("columns.pkl")
    return model, cols

model, cols = load_model()

# -------------------------------
# Header
# -------------------------------
st.markdown("<h2 style='text-align:center;'>🏡 House Price Predictor</h2>", unsafe_allow_html=True)

# -------------------------------
# Compact Input Layout
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    overall_qual = st.slider("⭐ Quality", 1, 10, 5)
    gr_liv_area = st.number_input("📐 Area (sq ft)", value=1500)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    garage_cars = st.slider("🚗 Garage Capacity", 0, 4, 1)
    total_bsmt = st.number_input("🏗 Basement (sq ft)", value=800)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    year_built = st.number_input("📅 Built Year", value=2000)
    lot_area = st.number_input("🌳 Lot Area (sq ft)", value=5000)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    full_bath = st.slider("🛁 Bathrooms", 0, 4, 2)
    bedrooms = st.slider("🛏 Bedrooms", 1, 6, 3)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

if st.button("🔮 Predict"):

    input_data = pd.DataFrame(columns=cols)
    input_data.loc[0] = 0

    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["GarageCars"] = garage_cars
    input_data["TotalBsmtSF"] = total_bsmt
    input_data["YearBuilt"] = year_built
    input_data["FullBath"] = full_bath
    input_data["BedroomAbvGr"] = bedrooms
    input_data["LotArea"] = lot_area

    prediction = model.predict(input_data)

    # Confidence
    rf_model = model.named_steps['model']
    # Transform input using preprocessing
    processed_input = model.named_steps['preprocessing'].transform(input_data)

    # Predict from each tree
    preds = np.array([tree.predict(processed_input) for tree in rf_model.estimators_])
    std_dev = preds.std()

    colA, colB = st.columns(2)

    with colA:
        st.metric("💰 Price", f"${prediction[0]:,.0f}")

    with colB:
        st.metric("📉 Uncertainty", f"± ${std_dev:,.0f}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Comparison (Compact)
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### 🌍 Compare")

colA, colB = st.columns(2)

with colA:
    area1 = st.number_input("Area 1", value=1500)
    qual1 = st.slider("Quality 1", 1, 10, 5)

with colB:
    area2 = st.number_input("Area 2", value=1800)
    qual2 = st.slider("Quality 2", 1, 10, 6)

if st.button("Compare"):

    def predict(area, qual):
        temp = pd.DataFrame(columns=cols)
        temp.loc[0] = 0
        temp["GrLivArea"] = area
        temp["OverallQual"] = qual
        return model.predict(temp)[0]

    p1 = predict(area1, qual1)
    p2 = predict(area2, qual2)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("House 1", f"${p1:,.0f}")

    with c2:
        st.metric("House 2", f"${p2:,.0f}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("ML-based house price prediction app")