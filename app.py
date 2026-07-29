
import streamlit as st
import pandas as pd
import joblib

loaded_model = joblib.load('xgb_model_boston_v1.0')

st.title('Boston House Price Prediction')
CRIM = st.slider('Per Capita Crime Rate(CRIM)',0.,100.,0.2,0.1)
ZN = st.slider('Proportion of Residential Land Over 25,000 sq. ft.(ZN)',0.,100.,0.2,0.1)
INDUS = st.slider('Proportion of Non-Retail Business Acres(INDUS)',0.,100.,0.2,0.1)
NX = st.slider('Nitric Oxides Concentration(NOX)',0.,1.,0.55,0.01)
RM = st.slider('Average Number of Rooms Per House(RM)',3,9,6,3)
AGE = st.slider("Proportion of owner-occupied units built prior to 1940 (AGE)", 0.0, 100.0, 65.0, 1.0)
DIS = st.slider("Weighted distances to employment centers (DIS)", 1.0, 12.0, 4.0, 0.1)
RAD = st.slider("Index of accessibility to radial highways (RAD)", 1, 24, 4, 1)
TAX = st.slider("Full-value property tax rate per $10,000 (TAX)", 100, 700, 300, 1)
PTRATIO = st.slider("Pupil-teacher ratio by town (PTRATIO)", 10.0, 25.0, 19.0, 0.1)
LSTAT = st.slider("% lower status of the population (LSTAT)", 0.0, 40.0, 12.0, 0.1)

CHAS = st.radio('Charles River Dummy Variable?',('Yes','No'))
if CHAS == 'Yes':
    CHAS = 1
else:
    CHAS = 0

input_data = pd.DataFrame([{
    'CRIM': CRIM,
    'ZN': ZN,
    'INDUS': INDUS,
    'NX': NX,
    'RM': RM,
    'AGE': AGE,
    'DIS': DIS,
    'RAD': RAD,
    'TAX': TAX,
    'PTRATIO': PTRATIO,
    'LSTAT': LSTAT,
    'CHAS': CHAS
}])

# Predict button
if st.button("Predict MEDV"):
    predicted_price = loaded_model.predict(input_data)[0]
    st.success(f"💰 Estimated Median Value of Home (MEDV): ${predicted_price*1000:,.2f}")
