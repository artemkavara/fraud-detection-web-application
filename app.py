import streamlit as st

from src.constants import STATES, GENDERS, CLASSIFIER
from src.transaction import Transaction
from datetime import datetime

st.set_page_config(layout="wide", page_title="Fraud Detection")
st.title('Credit card fraud detection')

col1, col2, col3 = st.columns(3)

with col1:
    st.header('Input data')
    
    category = st.text_input("Category")
    gender = st.selectbox("Gender", GENDERS)
    state = st.selectbox("State", STATES)
    job = st.text_input("Job")

with col2:    
    amt = st.number_input("Amount", min_value=0.0, format="%f")
    city_pop = st.number_input("Population of city", min_value=0, format="%i")
    birth_date = st.date_input("Date of birth")
    date_of_transaction = st.date_input("Date of transaction")
    time_of_transaction = st.time_input("Time of transaction")
    type_of_model = st.radio("Type of model", CLASSIFIER)
    clicked = st.button("Predict")

transaction = Transaction(category, gender, state, job, amt, city_pop, birth_date, date_of_transaction, time_of_transaction)
transaction.set_up_model(type_of_model)

with col3:
    st.header('Prediction')

    if clicked:
        st.markdown(f"This transaction is probably {'**fraud**' if transaction.predict() else '**not fraud**'}.")
        st.caption("Probability")
        st.table(transaction.predict_proba())
        
        st.download_button(
            label="Download data as txt",
            data=transaction.create_output(),
            file_name=f'fraud_transaction_pred_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt',
            mime='text/txt',
        )

