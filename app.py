import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
import gdown
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Indian Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Create sample data if not found
@st.cache_data
def create_sample_data():
    if not os.path.exists("crop_yield.csv"):
        states = ["Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Haryana",
                  "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
                  "Punjab", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]
        crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut", "Potato"]
        seasons = ["Kharif", "Rabi", "Whole Year", "Summer"]

        data = []
        for state in states:
            for crop in crops:
                for season in seasons:
                    if (crop == "Wheat" and season != "Rabi") or \
                       (crop == "Rice" and season not in ["Kharif", "Rabi"]) or \
                       (crop == "Cotton" and season != "Kharif"):
                        continue
                    for _ in range(3):
                        area = np.random.uniform(5, 100)
                        production = area * np.random.uniform(10, 40)
                        data.append([state, crop, season, area, production])

        df = pd.DataFrame(data, columns=["State", "Crop", "Season", "Area", "Production"])
        df.to_csv("crop_yield.csv", index=False)
        return df
    else:
        return pd.read_csv("crop_yield.csv")

@st.cache_data
def load_data():
    if os.path.exists("crop_yield.csv"):
        df = pd.read_csv("crop_yield.csv")
    else:
        df = create_sample_data()

    for col in df.columns:
        if col.lower() in ["state_name", "state"]:
            df.rename(columns={col: "State"}, inplace=True)
        elif col.lower() in ["crop_name", "crop"]:
            df.rename(columns={col: "Crop"}, inplace=True)
        elif col.lower() in ["season_name", "season"]:
            df.rename(columns={col: "Season"}, inplace=True)

    df.dropna(subset=["State", "Crop", "Season"], inplace=True)
    return df

@st.cache_resource
def get_model(data):
    model_path = "crop_yield_model.pkl"
    scaler_path = "scaler.pkl"
    columns_path = "columns.pkl"

    def download_from_drive(file_id, output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    # Replace these with your actual file IDs
    model_id = "1XxcDDecSf_iKFSmyzph8y82AAb_B-90o"
    scaler_id = "1tT37kr1UDCpcsK1tFT9o-ppRQRhsp1Hq"
    columns_id = "1B58A4LtTrInBQcGLpC1YuFFFiXv97CVd"

    if not os.path.exists(model_path):
        download_from_drive(model_id, model_path)
    if not os.path.exists(scaler_path):
        download_from_drive(scaler_id, scaler_path)
    if not os.path.exists(columns_path):
        download_from_drive(columns_id, columns_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(columns_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns

data = load_data()
if data.empty:
    st.error("Dataset is empty or missing required columns.")
    st.stop()

model, scaler, feature_columns = get_model(data)

states = sorted(data["State"].unique())
crops = sorted(data["Crop"].unique())
seasons = sorted(data["Season"].unique())

st.title("🌾 Indian Crop Yield Prediction")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Parameters")
    state = st.selectbox("Select State", states)
    crop = st.selectbox("Select Crop", crops)
    season = st.selectbox("Select Season", seasons)
    area = st.number_input("Cultivation Area (hectares)", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)
    filtered_data = data[(data["State"] == state) &
                         (data["Crop"] == crop) &
                         (data["Season"] == season)]
    predict_button = st.button("🔮 Predict Yield", type="primary")

with col2:
    st.subheader("Prediction Results")
    if predict_button:
        input_df = pd.DataFrame([[state, crop, season, area]],
                                columns=["State", "Crop", "Season", "Area"])
        input_encoded = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)
        scaled_input = scaler.transform(input_encoded)
        prediction = model.predict(scaled_input)[0]
        yield_per_hectare = prediction / area

        st.markdown("### 📊 Predicted Results")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(label="Total Production", value=f"{prediction:.2f} quintals")
        metric_col2.metric(label="Yield per Hectare", value=f"{yield_per_hectare:.2f} q/ha")
        metric_col3.metric(label="Area Cultivated", value=f"{area:.2f} hectares")

        if not filtered_data.empty:
            avg_production = filtered_data["Production"].mean()
            avg_yield = avg_production / filtered_data["Area"].mean()
            st.markdown("### 📈 Historical Comparison")
            st.markdown(f"Average historical production for {crop} in {state} during {season} season: **{avg_production:.2f}** quintals")
            st.markdown(f"Average historical yield: **{avg_yield:.2f}** quintals per hectare")
            diff_percentage = ((prediction - avg_production) / avg_production) * 100
            st.markdown(f"Your predicted yield is **{abs(diff_percentage):.1f}%** {'above' if diff_percentage > 0 else 'below'} average.")
    else:
        st.info("Enter your cultivation details and click 'Predict Yield' to see the prediction")
        if not data.empty:
            st.markdown("### 📊 Sample Historical Data")
            sample_df = data.sample(min(5, len(data)))[["State", "Crop", "Season", "Area", "Production"]].copy()
            sample_df.index = range(1, len(sample_df) + 1)
            st.dataframe(sample_df, use_container_width=True)

st.sidebar.title("ℹ️ Information")
st.sidebar.markdown("""
### About the App
This application predicts crop yield based on historical data from Indian agricultural records.

The machine learning model considers:
- **State**
- **Crop**
- **Season**
- **Cultivation Area**

### Notes
- 1 quintal = 100 kg
- Model predictions are estimates; real-world yield may vary.
""")
