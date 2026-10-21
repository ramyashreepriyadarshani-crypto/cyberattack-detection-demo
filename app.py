
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Cyberattack Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Explainable AI for Cyberattack Detection")
st.write(
    "Detect suspicious network traffic using Machine Learning "
    "and understand which features influence the model."
)

# Load project files
try:
    model = joblib.load("cyberattack_rf_no_ip.pkl")
    data = pd.read_csv("cyberattack_test_data (1).csv")
    importance = pd.read_csv("feature_importance.csv")
    comparison = pd.read_csv("model_comparison.csv")

except Exception as e:
    st.error(f"File loading error: {e}")
    st.info(
        "Keep app.py and all project files "
        "inside the same folder."
    )
    st.stop()

# Match test data with model features
features = list(model.feature_names_in_)
missing = [col for col in features if col not in data.columns]

if missing:
    st.error(f"Dataset is missing these features: {missing}")
    st.stop()

data = data[features]

st.success("Model and dataset loaded successfully!")

st.divider()
st.header("1. Cyberattack Prediction")

record = st.number_input(
    "Select network traffic record",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

sample = data.iloc[[int(record)]]

with st.expander("View network traffic features"):
    st.dataframe(sample, use_container_width=True)

if st.button("Detect Cyberattack", type="primary"):

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    confidence = probabilities[prediction] * 100

    if prediction == 1:
        st.error("🚨 Prediction: CYBERATTACK")
    else:
        st.success("✅ Prediction: NORMAL TRAFFIC")

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.caption(
        "Confidence is the model's estimated probability, "
        "not a guarantee that the prediction is correct."
    )

st.divider()
st.header("2. Explainable AI - Feature Importance")

st.write(
    "The chart below shows which features are important "
    "to the model overall. It does not explain an individual "
    "prediction."
)

# Identify feature importance columns
feature_col = next(
    (c for c in importance.columns
     if c.lower() in ["feature", "features", "feature_name"]),
    None
)

score_col = next(
    (c for c in importance.columns
     if c.lower() in [
         "importance",
         "feature_importance",
         "importance_score"
     ]),
    None
)

if feature_col and score_col:

    top = importance.sort_values(
        score_col, ascending=False
    ).head(10).sort_values(score_col)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top[feature_col], top[score_col])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Important Features")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:
    st.dataframe(importance, use_container_width=True)

st.divider()
st.header("3. Machine Learning Model Comparison")
st.dataframe(comparison, use_container_width=True)

st.divider()
st.header("4. About the Project")

st.markdown("""
- **Dataset:** CICIDS-2017 sample
- **Model:** Random Forest
- **Objective:** Classify network traffic as normal or attack
- **Explainability:** Feature importance
- **Interface:** Streamlit
""")

st.warning(
    "This is a research prototype evaluated on a limited test sample. "
    "High test scores do not guarantee real-world performance."
)
