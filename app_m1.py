import streamlit as st
st.set_page_config(page_title="Injury Predictor", layout="centered")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import urllib.request
import joblib

# -----------------------------
# Load model + preprocessing
# -----------------------------

def load_model(url, filename):
    urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

final = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_1/main/xgb_model.pkl",
    "xgb_model.pkl"
)
preprocessor_all = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_1/main/m1_preprocessor.pkl",
    "m1_preprocessor.pkl"
)

# -----------------------------
# Mappings
# -----------------------------
age_mapping = {
    'Under 16': 1,
    '16-19': 2,
    '20-24': 3,
    '25-34': 4,
    '35-44': 5,
    '45-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65+': 9,
    'Unknown': 0
}

severity_mapping = {
    'Over-3/7-Day': 0,
    'Major/Specified': 1,
    'Non-fatal MoP': 2,
    'Fatal': 3,
    'Fatal MoP': 4
}

reverse_mapping = {v: k for k, v in severity_mapping.items()}


one_hot_cols = ['main_activity', 'Kind_group', 'riskcat', 'gender', 'Region']

X_train_columns = ['age_band', 'main_activity_Construction of buildings',
       'main_activity_Specialised activities',
       'Kind_group_Contact with electricity',
       'Kind_group_Contact with machinery',
       'Kind_group_Drowned or asphyxiated', 'Kind_group_Exposed to explosion',
       'Kind_group_Exposed to fire',
       'Kind_group_Exposure to harmful substance',
       'Kind_group_Fall from height', 'Kind_group_Injured by an animal',
       'Kind_group_Lifting and handling injuries',
       'Kind_group_Physical assault', 'Kind_group_Slip, trip, fall same level',
       'Kind_group_Struck against', 'Kind_group_Struck by moving vehicle',
       'Kind_group_Struck by object',
       'Kind_group_Trapped by something collapsing', 'riskcat_Assault',
       'riskcat_Burns from hot substances/surfaces',
       'riskcat_Chemical harm, irritant or corrosive',
       'riskcat_Confined Spaces', 'riskcat_Crushed by excavation',
       'riskcat_Electric Shock', 'riskcat_Electric shock',
       'riskcat_Fall from ladder', 'riskcat_Fall from open edge',
       'riskcat_Fall from scaffold', 'riskcat_Fall through fragile material',
       'riskcat_Fire/explosion', 'riskcat_MEWP operations',
       'riskcat_Machinery guarding',
       'riskcat_Materials Handling including Manual handling',
       'riskcat_Mechanical Lifting Operations',
       'riskcat_Mechanical lifting operations', 'riskcat_Other',
       'riskcat_Other - bitten by dog',
       'riskcat_Other - episode of illness at work',
       'riskcat_Other - infection acquired at work',
       'riskcat_Other - injury whilst driving plant',
       'riskcat_Other - road traffic accident', 'riskcat_Other - rope access',
       'riskcat_Overturning plant or moving machinery',
       'riskcat_Public protection', 'riskcat_Slip or trip on same level',
       'riskcat_Struck by falling object', 'riskcat_Struck by flying object',
       'riskcat_Struck by moving vehicle', 'riskcat_Unintended collapse',
       'riskcat_Using hand/power tools', 'gender_Male', 'Region_Eastern',
       'Region_London', 'Region_North East', 'Region_North West',
       'Region_Scotland', 'Region_South East', 'Region_South West',
       'Region_Wales', 'Region_West Midlands', 'Region_Yorkshire and the Humber']


# -----------------------------
# UI
# -----------------------------
st.title("🚧 Injury Severity Prediction")

main_activity = st.selectbox("Main Activity", [
    'Construction of buildings', 'Civil engineering', 'Specialised activities'
])

kind_group = st.selectbox("Kind Group", ['Slip, trip, fall same level', 'Lifting and handling injuries',
       'Struck by object', 'Another kind of accident', 'Fall from height',
       'Exposure to harmful substance', 'Contact with machinery',
       'Struck against', 'Struck by moving vehicle',
       'Contact with electricity', 'Exposed to fire', 'Physical assault',
       'Trapped by something collapsing', 'Injured by an animal',
       'Exposed to explosion', 'Drowned or asphyxiated'])

riskcat = st.selectbox("Risk Category", ['Slip or trip on same level',
       'Materials Handling including Manual handling',
       'Fall from scaffold', 'Struck by flying object',
       'Fall from ladder', 'Using hand/power tools',
       'Fall from open edge', 'Chemical harm, irritant or corrosive',
       'Struck by falling object', 'MEWP operations',
       'Mechanical Lifting Operations', 'Public protection',
       'Struck by moving vehicle', 'Machinery guarding', 'Electric shock',
       'Overturning plant or moving machinery',
       'Fall through fragile material', 'Fire/explosion',
       'Burns from hot substances/surfaces', 'Assault',
       'Unintended collapse', 'Other - road traffic accident', 'Other',
       'Confined Spaces', 'Other - infection acquired at work',
       'Other - bitten by dog', 'Other - rope access',
       'Other - injury whilst driving plant',
       'Other - episode of illness at work', 'Asbestos',
       'Mechanical lifting operations', 'Crushed by excavation'])

age_band = st.selectbox("Age Band", ['Under 16', '16-19', '20-24', '25-34', '35-44', '45-54', '55-59', '60-64', '65+', 'Unknown'])
gender = st.selectbox("Gender", ["Male", "Female"])
region = st.selectbox("Region", ['London', 'South East', 'Scotland', 'North West', 'South West', 'East Midlands', 'Eastern', 'North East', 'Wales', 'West Midlands', 'Yorkshire and the Humber']
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    ## Step 1: build input DataFrame (RAW format)
    input_data = {
        "main_activity": main_activity,
        "Kind_group": kind_group,
        "riskcat": riskcat,
        "age_band": age_band,
        "gender": gender,
        "Region": region
    }


    feature_cols = X_train_columns  # must be saved before training

    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0  # initialize all zeros


    ## Mapping
    row['age_band'] = age_mapping[input_data['age_band']]

    # helper function
    def set_feature(prefix, value):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row[col] = 1

    set_feature("main_activity", input_data["main_activity"])
    set_feature("Kind_group", input_data["Kind_group"])
    set_feature("riskcat", input_data["riskcat"])
    set_feature("gender", input_data["gender"])
    set_feature("Region", input_data["Region"])


    # Step 2: preprocess
    X_input = preprocessor_all.transform(row)

    # Step 3: prediction
    prediction = final.predict(X_input)
    proba = max(final.predict_proba(X_input)[0])

    # Step 4: decode label
    pred_label = reverse_mapping[prediction[0]]
    confidence = proba * 100

    # -----------------------------
    # Output
    # -----------------------------
    st.success(f"Prediction: {pred_label}")
    st.info(f"Confidence: {confidence:.2f}%")
