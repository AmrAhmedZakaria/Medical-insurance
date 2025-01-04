import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest.pkl")

st.title("Medical Insurance cost prediction")
st.write("""
This application predicts the medical insurance cost based on various factors such as age, sex, BMI, number of children, smoking status, and region. 
The model used for prediction is a Random Forest Regressor trained on a dataset of medical insurance costs.
""")


def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    children = st.sidebar.slider("Children", 0, 10, 0)
    smoker = st.sidebar.selectbox("Smoker", [0, 1])
    region = st.sidebar.selectbox("Region", [0, 1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()


if st.button("Predict"):
    try:
        prediction = model.predict(df)
        st.subheader("Prediction")
        st.markdown(f"The predicted medical insurance cost is: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
