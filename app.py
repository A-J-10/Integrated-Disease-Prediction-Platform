import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, date
from streamlit_option_menu import option_menu

# ─── 1) Must be first ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ─── 2) Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .sidebar .sidebar-content { background-color: #FFFFFF; }
  [data-testid="metric-container"] {
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 8px;
  }
  button[kind="primary"] {
    background-color: #2E86AB;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ─── 3) Categorical mappings for Diabetes inputs ────────────────────────────────
gender_map = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}

def encode_diabetes_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map(gender_map)
    df["smoking_history"] = df["smoking_history"].map(smoking_map)
    return df

# ─── 4) Load models ────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
diabetes_model   = pickle.load(open(f"{base}/Models/diabetes.sav", "rb"))
heart_model      = pickle.load(open(f"{base}/Models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open(f"{base}/Models/parkinsons_model.sav", "rb"))

# ─── 5) Session state for recording predictions ────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []  # each: {timestamp, disease, result, risk}

def get_risk(model, inp):
    try:
        return float(model.predict_proba([inp])[0][1])
    except AttributeError:
        try:
            score = model.decision_function([inp])[0]
            return float(1 / (1 + np.exp(-score)))
        except Exception:
            return float(model.predict([inp])[0])

def update_record(disease, result, risk):
    st.session_state["predictions"].append({
        "timestamp": datetime.now(),
        "disease": disease,
        "result": result,
        "risk": risk
    })

# ─── 6) Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    choice = option_menu("Health Assistant",
                         ["Dashboard", "Single Prediction", "Batch Prediction"],
                         icons=["speedometer", "clipboard-data", "cloud-upload"],
                         default_index=0)
# ─── 7) Dashboard ─────────────────────────────────────────────────────────────
if choice == "Dashboard":
    st.title("📊 Dashboard")
    today = date.today()
    start_date, end_date = st.date_input(
        "Select date range",
        value=(today - timedelta(days=7), today)
    )

    # Convert session state predictions to DataFrame
    df = pd.DataFrame(st.session_state["predictions"])
    if not df.empty:
        # Add a date column for filtering
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # Filter data based on the selected date range
        df_f = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        if df_f.empty:
            st.info("No records found in the selected date range.")
        else:
            # Format the timestamp to include only date and time in HH:MM format
            df_f["timestamp"] = pd.to_datetime(df_f["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

            # Display metrics for each disease
            c1, c2, c3 = st.columns(3)
            for col, disease in zip((c1, c2, c3), ["Diabetes", "Heart Disease", "Parkinson's"]):
                avg_r = df_f[df_f["disease"] == disease]["risk"].mean()
                col.metric(f"{disease} Avg Risk", f"{avg_r:.2f}" if not pd.isna(avg_r) else "N/A")

            # Plot the line chart for each predicted risk
            fig = px.line(
                df_f,
                x="timestamp",
                y="risk",
                color="disease",
                title="Predicted Risk Over Time",
                markers=True,
                labels={"risk": "Predicted Risk", "timestamp": "Timestamp", "disease": "Disease"}
            )

            # Adjust layout for better readability
            fig.update_layout(
                xaxis=dict(title="Timestamp",tickformat="%Y-%m-%d %H:%M"),
                yaxis=dict(title="Predicted Risk"),
                margin=dict(l=40, r=40, t=40, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display the filtered data
            st.dataframe(df_f[["timestamp", "disease", "result", "risk"]])
        
    else:
        st.info("No predictions made yet.")

    # Generate a shareable link with query parameters
    if st.button("Generate Shareable Link"):
        st.experimental_set_query_params(
            start=str(start_date), end=str(end_date)
        )
        st.write("🔗 Copy & share this URL to preserve your filters")

# ─── 8) Single Prediction ─────────────────────────────────────────────────────
elif choice == "Single Prediction":
    tabs = st.tabs(["Diabetes", "Heart Disease", "Parkinson's"])

    # — Diabetes —
    with tabs[0]:
        st.header("Diabetes Prediction")
        with st.expander("▶️ Input parameters"):
            gender = st.selectbox("Gender", list(gender_map.keys()))
            age = st.number_input("Age", 0, 120)
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_dz = st.selectbox("Heart Disease", [0, 1])
            smoking = st.selectbox("Smoking History", list(smoking_map.keys()))
            bmi = st.number_input("BMI", 0.0, 100.0, step=0.1)
            hba1c = st.number_input("HbA1c", 0.0, 14.0, step=0.1)
            gluc = st.number_input("Blood Glucose", 0, 1000)
        if st.button("Predict Diabetes"):
            # encode
            gender_n = gender_map[gender]
            smoking_n = smoking_map[smoking]
            inp = [gender_n, age, hypertension, heart_dz,
                   smoking_n, bmi, hba1c, gluc]
            risk = get_risk(diabetes_model, inp)
            res  = "Diabetic" if risk > 0.5 else "Not Diabetic"
            st.write(f"**{res}**  (Risk: {risk:.2f})")
            update_record("Diabetes", res, risk)

    # — Heart Disease —
    with tabs[1]:
        st.header("Heart Disease Prediction")
        with st.expander("▶️ Input parameters"):
            age_hd = st.number_input("Age", 0, 120, key="hd_age")
            sex = st.selectbox("Sex", [0, 1])
            cp = st.selectbox("Chest Pain Type", list(range(4)))
            trest = st.number_input("Resting BP", 0, 300)
            chol = st.number_input("Cholesterol", 0, 600)
            fbs = st.selectbox("Fasting BS>120", [0, 1])
            ecg = st.selectbox("Rest ECG", list(range(3)))
            thal = st.number_input("Max Heart Rate", 0, 300)
            exang = st.selectbox("Exercise Angina", [0, 1])
            oldp = st.number_input("ST Depression", 0.0, 10.0, step=0.1)
            slope = st.selectbox("Slope", list(range(3)))
            ca = st.selectbox("Major Vessels", list(range(5)))
            thal_d = st.selectbox("Thalassemia", list(range(4)))
        if st.button("Predict Heart Disease"):
            inp = [age_hd, sex, cp, trest, chol, fbs, ecg,
                   thal, exang, oldp, slope, ca, thal_d]
            risk = get_risk(heart_model, inp)
            res  = "Has Disease" if risk > 0.5 else "No Disease"
            st.write(f"**{res}**  (Risk: {risk:.2f})")
            update_record("Heart Disease", res, risk)

    # — Parkinson's —
    # ─── Parkinson’s Prediction (Single) ──────────────────────────────────────────
    with tabs[2]:
        st.header("Parkinson's Prediction")
        with st.expander("▶️ Input parameters"):
            patient_id       = st.number_input("Patient ID", min_value=0, step=1, key="parkinson_pid")
            age               = st.number_input("Age", 0, 120, step=1, key="parkinson_age")
            gender            = st.selectbox("Gender (0=Female,1=Male)", [0, 1], key="parkinson_gender")
            ethnicity         = st.selectbox("Ethnicity", [0, 1, 2, 3], key="parkinson_ethnicity")
            education         = st.selectbox("Education Level (0–3)", [0, 1, 2, 3], key="parkinson_education")
            bmi               = st.number_input("BMI", 0.0, 100.0, step=0.1, key="parkinson_bmi")
            smoking           = st.selectbox("Smoking (0=No,1=Yes)", [0, 1], key="parkinson_smoking")
            alcohol           = st.number_input("Alcohol Consumption (units/day)", 0.0, step=0.1, key="parkinson_alcohol")
            physical          = st.number_input("Physical Activity (hrs/week)", 0.0, step=0.1, key="parkinson_physical")
            diet              = st.number_input("Diet Quality (0–10)", 0.0, 10.0, step=0.1, key="parkinson_diet")
            sleep             = st.number_input("Sleep Quality (0–10)", 0.0, 10.0, step=0.1, key="parkinson_sleep")
            family_history    = st.selectbox("Family History Parkinson’s", [0, 1], key="parkinson_family")
            tbi               = st.selectbox("Traumatic Brain Injury", [0, 1], key="parkinson_tbi")
            hypertension      = st.selectbox("Hypertension", [0, 1], key="parkinson_hypertension")
            diabetes          = st.selectbox("Diabetes", [0, 1], key="parkinson_diabetes")
            depression        = st.selectbox("Depression", [0, 1], key="parkinson_depression")
            stroke            = st.selectbox("Stroke", [0, 1], key="parkinson_stroke")
            systolic          = st.number_input("Systolic BP", 0, 300, step=1, key="parkinson_systolic")
            diastolic         = st.number_input("Diastolic BP", 0, 200, step=1, key="parkinson_diastolic")
            chol_total        = st.number_input("Cholesterol Total", 0.0, step=0.1, key="parkinson_chol_tot")
            chol_ldl          = st.number_input("Cholesterol LDL", 0.0, step=0.1, key="parkinson_chol_ldl")
            chol_hdl          = st.number_input("Cholesterol HDL", 0.0, step=0.1, key="parkinson_chol_hdl")
            chol_trig         = st.number_input("Cholesterol Triglycerides", 0.0, step=0.1, key="parkinson_chol_trig")
            updrs             = st.number_input("UPDRS Score", 0.0, step=0.1, key="parkinson_updrs")
            moca              = st.number_input("MoCA Score", 0.0, step=0.1, key="parkinson_moca")
            functional        = st.number_input("Functional Assessment", 0.0, step=0.1, key="parkinson_func")
            tremor            = st.selectbox("Tremor", [0, 1], key="parkinson_tremor")
            rigidity          = st.selectbox("Rigidity", [0, 1], key="parkinson_rigidity")
            bradykinesia      = st.selectbox("Bradykinesia", [0, 1], key="parkinson_brady")
            postural          = st.selectbox("Postural Instability", [0, 1], key="parkinson_postural")
            speech            = st.selectbox("Speech Problems", [0, 1], key="parkinson_speech")
            sleep_disorders   = st.selectbox("Sleep Disorders", [0, 1], key="parkinson_sleep_dis")
            constipation      = st.selectbox("Constipation", [0, 1], key="parkinson_constip")
            # ← new input, not used for prediction
            doctor_in_charge  = st.text_input("Doctor In Charge", key="parkinson_doctor")

        if st.button("Predict Parkinson's", key="parkinson_predict"):
            try:
                features = [
                    age, gender, ethnicity, education, bmi, smoking, alcohol,
                    physical, diet, sleep, family_history, tbi, hypertension,
                    diabetes, depression, stroke, systolic, diastolic,
                    chol_total, chol_ldl, chol_hdl, chol_trig, updrs, moca,
                    functional, tremor, rigidity, bradykinesia, postural,
                    speech, sleep_disorders, constipation
                ]
                risk = get_risk(parkinsons_model, features)
                res = "Has Parkinson's" if risk > 0.5 else "No Parkinson's"
                st.write(f"**{res}**  (Risk: {risk:.2f})")
                # You can optionally record doctor_in_charge in your session state if desired:
                update_record("Parkinson's", res, risk)
            except ValueError:
                st.error("Invalid input! Please check all fields.")

# ─── 9) Batch Prediction ──────────────────────────────────────────────────────
else:
    st.title("📂 Batch Prediction")
    disease_batch = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinson's"])
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        if disease_batch == "Diabetes":
            df_enc = encode_diabetes_inputs(df_batch)
            cols = ["gender","age","hypertension","heart_disease",
                    "smoking_history","bmi","hba1c","blood_glucose"]
            inputs = df_enc[cols].values.tolist()
            probs  = [get_risk(diabetes_model, row) for row in inputs]
        elif disease_batch == "Heart Disease":
            inputs = df_batch.values.tolist()
            probs  = [get_risk(heart_model, row) for row in inputs]
        else:
            inputs = df_batch.drop(columns=["name"], errors="ignore").values.tolist()
            probs  = [get_risk(parkinsons_model, row) for row in inputs]

        df_batch["risk"]       = probs
        df_batch["prediction"] = (df_batch["risk"] > 0.5).astype(int)
        st.download_button(
            "Download Results",
            df_batch.to_csv(index=False),
            file_name="batch_predictions.csv"
        )
