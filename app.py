import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Indian Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Create sample data if it doesn't exist (for demo purposes)
@st.cache_data
def create_sample_data():
    if not os.path.exists("crop_yield.csv"):
        # Create sample data
        states = ["Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Haryana", 
                  "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", 
                  "Punjab", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]
        
        crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut", "Potato"]
        
        seasons = ["Kharif", "Rabi", "Whole Year", "Summer"]
        
        # Generate sample data
        data = []
        for state in states:
            for crop in crops:
                for season in seasons:
                    # Not all crops grow in all seasons, add some realistic constraints
                    if (crop == "Wheat" and season not in ["Rabi"]) or \
                       (crop == "Rice" and season not in ["Kharif", "Rabi"]) or \
                       (crop == "Cotton" and season not in ["Kharif"]):
                        continue
                        
                    # Add some sample records with varied production values
                    for _ in range(3):
                        area = np.random.uniform(5, 100)
                        production = area * np.random.uniform(10, 40)  # production in quintals per hectare
                        data.append([state, crop, season, area, production])
        
        df = pd.DataFrame(data, columns=["State", "Crop", "Season", "Area", "Production"])
        df.to_csv("crop_yield.csv", index=False)
        
        return df
    else:
        return pd.read_csv("crop_yield.csv")

# Cache the dataset loading for performance
@st.cache_data
def load_data():
    if os.path.exists("crop_yield.csv"):
        df = pd.read_csv("crop_yield.csv")
    else:
        df = create_sample_data()
    
    # Standardize column names
    for col in df.columns:
        if col.lower() in ["state_name", "state"]:
            df.rename(columns={col: "State"}, inplace=True)
        elif col.lower() in ["crop_name", "crop"]:
            df.rename(columns={col: "Crop"}, inplace=True)
        elif col.lower() in ["season_name", "season"]:
            df.rename(columns={col: "Season"}, inplace=True)
            
    # Optionally, drop rows with missing values in important columns
    df.dropna(subset=["State", "Crop", "Season"], inplace=True)
    return df

# Function to train/load the model
@st.cache_resource
def get_model(data):
    model_path = "crop_yield_model.pkl"
    scaler_path = "scaler.pkl"
    columns_path = "columns.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(columns_path):
        # Load existing model
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)
        with open(columns_path, "rb") as file:
            feature_columns = pickle.load(file)
    else:
        # Train a new model
        # Prepare data
        df = data.copy()
        
        # Calculate yield from production and area if not present
        if "Yield" not in df.columns:
            df["Yield"] = df["Production"] / df["Area"]
        
        # One-hot encode categorical variables
        X = pd.get_dummies(df[["State", "Crop", "Season", "Area"]])
        y = df["Production"]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save feature columns
        feature_columns = X.columns.tolist()
        
        # Save model and artifacts
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)
        with open(columns_path, "wb") as file:
            pickle.dump(feature_columns, file)
    
    return model, scaler, feature_columns

# Load dataset
data = load_data()
if data.empty:
    st.error("Dataset is empty or missing required columns.")
    st.stop()

# Get model and required artifacts
model, scaler, feature_columns = get_model(data)

# Get unique values for dropdowns
states = sorted(data["State"].unique())
crops = sorted(data["Crop"].unique())
seasons = sorted(data["Season"].unique())

# App UI
st.title("🌾 Indian Crop Yield Prediction")

# Layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Parameters")
    
    # Input fields
    state = st.selectbox("Select State", states)
    crop = st.selectbox("Select Crop", crops)
    season = st.selectbox("Select Season", seasons)
    area = st.number_input("Cultivation Area (hectares)", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)
    
    # Filter data based on selections to show historical information
    filtered_data = data[(data["State"] == state) & 
                         (data["Crop"] == crop) & 
                         (data["Season"] == season)]
    
    # Predict button
    predict_button = st.button("🔮 Predict Yield", type="primary")

with col2:
    st.subheader("Prediction Results")
    
    if predict_button:
        # Create input for prediction
        input_df = pd.DataFrame([[state, crop, season, area]], 
                              columns=["State", "Crop", "Season", "Area"])
        
        # One-hot encode the input and align columns with training features
        input_encoded = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)
        
        # Scale input
        scaled_input = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Calculate yield per hectare
        yield_per_hectare = prediction / area
        
        # Display results
        st.markdown("### 📊 Predicted Results")
        
        # Create metrics in a row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(label="Total Production", value=f"{prediction:.2f} quintals")
            
        with metric_col2:
            st.metric(label="Yield per Hectare", value=f"{yield_per_hectare:.2f} q/ha")
            
        with metric_col3:
            st.metric(label="Area Cultivated", value=f"{area:.2f} hectares")
        
        # Show historical data comparison if available
        if not filtered_data.empty:
            avg_production = filtered_data["Production"].mean()
            avg_yield = avg_production / filtered_data["Area"].mean()
            
            st.markdown("### 📈 Historical Comparison")
            st.markdown(f"Average historical production for {crop} in {state} during {season} season: **{avg_production:.2f}** quintals")
            st.markdown(f"Average historical yield: **{avg_yield:.2f}** quintals per hectare")
            
            # Calculate if prediction is above or below average
            diff_percentage = ((prediction - avg_production) / avg_production) * 100
            st.markdown(f"Your predicted yield is **{abs(diff_percentage):.1f}%** {'above' if diff_percentage > 0 else 'below'} average.")
    else:
        st.info("Enter your cultivation details and click 'Predict Yield' to see the prediction")
        
        # Show some sample historical data
        if not data.empty:
            st.markdown("### 📊 Sample Historical Data")
            # Fix the indexing issue by adding .copy() and then resetting the index starting from 1
            sample_df = data.sample(min(5, len(data)))[["State", "Crop", "Season", "Area", "Production"]].copy()
            sample_df.index = range(1, len(sample_df) + 1)  # Start index from 1 instead of 0
            st.dataframe(sample_df, use_container_width=True)

# Add information section
st.sidebar.title("ℹ️ Information")
st.sidebar.markdown("""
### About the App
This application predicts crop yield based on historical data from Indian agricultural records. 
The machine learning model takes into account:

- **State**: Geographic location within India
- **Crop Type**: The specific crop being cultivated
- **Season**: Kharif (monsoon), Rabi (winter), etc.
- **Cultivation Area**: Land area in hectares

### Notes
- 1 quintal = 100 kg
- Model predictions are estimates based on historical trends
- Actual yield may vary due to weather conditions, soil quality, etc.
""")

