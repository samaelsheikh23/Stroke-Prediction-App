import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================
# Load Model Files
# ==========================

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")



# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)



# ==========================
# Dark Medical CSS
# ==========================

st.markdown(
"""
<style>

/* Background */

.stApp {

    background:
    linear-gradient(
        135deg,
        #020617,
        #111827
    );

    color:white;

}


/* Hide Streamlit Header */

header {
    background: transparent;
}


[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}


footer {
    visibility: hidden;
}


[data-testid="stToolbar"] {
    right: 2rem;
}



/* Titles */

h1 {

    text-align:center;

    color:#00d4ff;

    font-size:45px;

    font-weight:800;

}


h2,h3 {

    color:#00d4ff;

}



/* Text */

p {

    color:#e5e7eb;

    font-size:17px;

}



/* Cards */

.card {


    background:

    rgba(255,255,255,0.08);


    padding:25px;


    border-radius:20px;


    border:

    1px solid rgba(255,255,255,0.15);


    box-shadow:

    0 10px 30px rgba(0,0,0,0.5);


    margin-bottom:20px;


}



/* Inputs */

label {

    color:white !important;

    font-weight:600 !important;

}



input,
div[data-baseweb="select"] > div {


    background:#1f2937 !important;

    color:white !important;

    border-radius:12px !important;


}



/* Predict Button */

.stButton button {

    width:100%;

    height:60px;

    border-radius:20px;


    background:

    linear-gradient(
        90deg,
        #800020,
        #a52a2a
    );


    color:white;


    font-size:22px;


    font-weight:bold;


    border:none;


}



.stButton button:hover {


    background:

    linear-gradient(
        90deg,
        #a52a2a,
        #800020
    );


    transform:scale(1.04);


}



/* Sidebar */

section[data-testid="stSidebar"] {


    background:

    linear-gradient(
    180deg,
    #020617,
    #0f172a
    );


}



section[data-testid="stSidebar"] * {

    color:white !important;

}




/* Metrics */

[data-testid="stMetricValue"] {

    color:#00d4ff !important;

}


</style>

""",
unsafe_allow_html=True
)





# ==========================
# Header
# ==========================

st.markdown(
"""
<div class="card">

<h1>
🧠 Stroke Prediction
</h1>

</div>

""",
unsafe_allow_html=True
)






# ==========================
# Sidebar
# ==========================


with st.sidebar:


    st.markdown(
    """

    <h1 style="
    color:#00d4ff;
    font-size:32px;
    ">
    🧠 Stroke Prediction
    </h1>

    """,

    unsafe_allow_html=True
    )



    st.markdown(
    """

    <div class="card">


    <h3>
    About The Application
    </h3>


    <p>

    Stroke AI is an intelligent prediction
    system that analyzes patient medical
    information to estimate stroke risk.

    </p>


    <h3>
    Main Features
    </h3>


    <p>

    ✔ Patient Risk Assessment

    <br>

    ✔ Stroke Probability Prediction

    <br>

    ✔ AI-Based Clinical Support

    <br>

    ✔ Fast Automated Analysis

    </p>


    </div>


    """,

    unsafe_allow_html=True

    )






# ==========================
# Patient Information
# ==========================


st.markdown(
"""

<div class="card">

<h2>
👤 Patient Information
</h2>

</div>

""",

unsafe_allow_html=True
)




col1, col2, col3 = st.columns(3)




with col1:


    age = st.number_input(
        "Age",
        0,
        120,
        45
    )



    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )



    bmi = st.number_input(
        "BMI",
        10.0,
        70.0,
        25.0
    )





with col2:


    hypertension = st.selectbox(
        "Hypertension",
        [0,1],

        format_func=lambda x:
        "Yes" if x==1 else "No"
    )



    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1],

        format_func=lambda x:
        "Yes" if x==1 else "No"
    )



    glucose = st.number_input(
        "Average Glucose Level",
        0.0,
        300.0,
        100.0
    )





with col3:


    married = st.selectbox(
        "Ever Married",
        [
            "Yes",
            "No"
        ]
    )



    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )



    residence = st.selectbox(
        "Residence Type",
        [
            "Urban",
            "Rural"
        ]
    )




# ==========================
# Center Smoking Feature
# ==========================


st.write("")


center1, center2, center3 = st.columns([1,2,1])


with center2:


    smoking = st.selectbox(
        "🚬 Smoking Status",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

# ==========================
# Prediction
# ==========================

if st.button("🔍 Predict Stroke Risk"):

    # ==========================
    # Create Input DataFrame
    # ==========================

    input_data = pd.DataFrame({

        "age": [age],

        "hypertension": [hypertension],

        "heart_disease": [heart_disease],

        "avg_glucose_level": [glucose],

        "bmi": [bmi],

        "gender": [gender],

        "ever_married": [married],

        "work_type": [work_type],

        "Residence_type": [residence],

        "smoking_status": [smoking]

    })


    # ==========================
    # SAME ENCODING AS NOTEBOOK
    # ==========================

    input_data["gender"] = input_data["gender"].map({
        "Male": 0,
        "Female": 1
    })

    input_data["ever_married"] = input_data["ever_married"].map({
        "No": 0,
        "Yes": 1
    })


    # ==========================
    # SAME ONE-HOT ENCODING
    # Notebook used:
    # pd.get_dummies(..., drop_first=True)
    # ==========================

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "work_type",
            "Residence_type",
            "smoking_status"
        ],
        drop_first=True
    )


    # ==========================
    # Convert Boolean to Integer
    # ==========================

    bool_cols = input_data.select_dtypes(
        include="bool"
    ).columns

    input_data[bool_cols] = (
        input_data[bool_cols].astype(int)
    )


    # ==========================
    # SAME LOG TRANSFORMATION
    # Notebook:
    # avg_glucose_level
    # bmi
    # ==========================

    input_data["avg_glucose_level"] = np.log1p(
        input_data["avg_glucose_level"]
    )

    input_data["bmi"] = np.log1p(
        input_data["bmi"]
    )


    # ==========================
    # SAME FEATURE ORDER
    # ==========================

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )


    # ==========================
    # Scaling
    # ==========================

    input_scaled = scaler.transform(
        input_data
    )


    # ==========================
    # Prediction
    # ==========================

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]


    confidence = max(
        probability,
        1 - probability
    )


    # ==========================
    # Result
    # ==========================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        risk = "High Risk"

        st.error(
            "⚠️ Stroke Risk Detected"
        )

        interpretation = (
            "The model predicts a higher "
            "risk of stroke."
        )

        recommendation = (
            "Medical consultation and further "
            "clinical evaluation are recommended."
        )


    else:

        risk = "Low Risk"

        st.success(
            "✅ Low Stroke Risk"
        )

        interpretation = (
            "The model predicts a lower "
            "risk of stroke."
        )

        recommendation = (
            "Continue healthy habits and "
            "regular health monitoring."
        )


    # ==========================
    # Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Risk Level",
            risk
        )


    with col2:

        st.metric(
            "Stroke Probability",
            f"{probability * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ==========================
    # Probability Chart
    # ==========================

    st.subheader(
        "📈 Risk Probability Analysis"
    )

    probability_df = pd.DataFrame({

        "Risk Category": [
            "No Stroke Risk",
            "Stroke Risk"
        ],

        "Probability (%)": [
            (1 - probability) * 100,
            probability * 100
        ]

    })


    st.bar_chart(
        probability_df.set_index(
            "Risk Category"
        )
    )


    # ==========================
    # Patient Summary
    # ==========================

    st.subheader(
        "📝 Patient Information Summary"
    )

    summary = pd.DataFrame({

        "Feature": [
            "Age",
            "Gender",
            "BMI",
            "Hypertension",
            "Heart Disease",
            "Glucose Level",
            "Ever Married",
            "Work Type",
            "Residence Type",
            "Smoking Status"
        ],

        "Value": [
            str(age),
            str(gender),
            str(bmi),

            "Yes" if hypertension == 1
            else "No",

            "Yes" if heart_disease == 1
            else "No",

            str(glucose),
            str(married),
            str(work_type),
            str(residence),
            str(smoking)
        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ==========================
    # Assessment Summary
    # ==========================

    st.subheader(
        "📌 Assessment Summary"
    )

    st.markdown(
        f"""
        <div class="card">

        <p>
        <b>Prediction:</b>
        {risk}
        </p>

        <p>
        <b>Stroke Probability:</b>
        {probability * 100:.2f}%
        </p>

        <p>
        <b>Confidence:</b>
        {confidence * 100:.2f}%
        </p>

        <p>
        <b>Interpretation:</b>
        {interpretation}
        </p>

        <p>
        <b>Recommendation:</b>
        {recommendation}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================
# Footer
# ==========================


st.markdown(
"""

<br>

<center>

<span style="
color:#94a3b8;
font-size:14px;
">

AI-Based Stroke Prediction System

</span>

</center>


""",

unsafe_allow_html=True
)
