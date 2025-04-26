# streamlit_app.py

import streamlit as st
import requests

st.title("🎧 Customer Churn Prediction App")

st.write("Fill the user details below to predict churn probability:")

# Input fields
city = st.number_input("City", min_value=-1, max_value=50, value=1)
bd = st.number_input("Age (Birth Day)", min_value=0.0, max_value=100.0, value=30.0)
registered_via = st.number_input("Registered Via", min_value=-1, max_value=10, value=7)
num_25 = st.number_input("Number of 25% songs", min_value=0.0, value=100.0)
num_50 = st.number_input("Number of 50% songs", min_value=0.0, value=50.0)
num_75 = st.number_input("Number of 75% songs", min_value=0.0, value=30.0)
num_985 = st.number_input("Number of 98.5% songs", min_value=0.0, value=5.0)
num_100 = st.number_input("Number of 100% songs", min_value=0.0, value=10.0)
num_unq = st.number_input("Unique Songs", min_value=0.0, value=120.0)
total_secs = st.number_input("Total Seconds Listened", min_value=0.0, value=15000.0)
active_days = st.number_input("Active Days", min_value=0.0, value=20.0)
payment_method_id = st.number_input("Payment Method ID", min_value=-1, value=41)
payment_plan_days = st.number_input("Payment Plan Days", min_value=0.0, value=30.0)
plan_list_price = st.number_input("Plan List Price", min_value=0.0, value=100.0)
actual_amount_paid = st.number_input("Actual Amount Paid", min_value=0.0, value=90.0)
is_auto_renew = st.selectbox("Is Auto Renew?", options=[0,1])
is_cancel = st.selectbox("Is Cancel?", options=[0,1])
song_completion_rate = st.number_input("Song Completion Rate", min_value=0.0, max_value=1.0, value=0.8)
avg_secs_per_song = st.number_input("Average Seconds Per Song", min_value=0.0, value=180.0)
high_engagement = st.selectbox("High Engagement?", options=[0,1])
membership_days = st.number_input("Membership Days", min_value=0.0, value=365.0)
discount_rate = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, value=0.1)
is_trial = st.selectbox("Is Trial User?", options=[0,1])
auto_renew_but_cancel = st.selectbox("Auto Renew But Cancel?", options=[0,1])
age_group = st.selectbox("Age Group", options=[0,1,2,3,4,5])
account_age_days = st.number_input("Account Age Days", min_value=0.0, value=800.0)
gender = st.selectbox("Gender", options=["male", "female", "other"])
payment_method_risk = st.selectbox("Payment Method Risk", options=["low_risk", "medium_risk", "high_risk", "no_txn"])
has_transaction = st.selectbox("Has Transaction?", options=[0,1])
no_txn_and_churn = st.selectbox("No Transaction and Churn?", options=[0,1])

# When button clicked
if st.button("Predict Churn Probability"):
    input_data = {
        "city": city,
        "bd": bd,
        "registered_via": registered_via,
        "num_25": num_25,
        "num_50": num_50,
        "num_75": num_75,
        "num_985": num_985,
        "num_100": num_100,
        "num_unq": num_unq,
        "total_secs": total_secs,
        "active_days": active_days,
        "payment_method_id": payment_method_id,
        "payment_plan_days": payment_plan_days,
        "plan_list_price": plan_list_price,
        "actual_amount_paid": actual_amount_paid,
        "is_auto_renew": is_auto_renew,
        "is_cancel": is_cancel,
        "song_completion_rate": song_completion_rate,
        "avg_secs_per_song": avg_secs_per_song,
        "high_engagement": high_engagement,
        "membership_days": membership_days,
        "discount_rate": discount_rate,
        "is_trial": is_trial,
        "auto_renew_but_cancel": auto_renew_but_cancel,
        "age_group": age_group,
        "account_age_days": account_age_days,
        "gender": gender,
        "payment_method_risk": payment_method_risk,
        "has_transaction": has_transaction,
        "no_txn_and_churn": no_txn_and_churn
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)
        if response.status_code == 200:
            churn_probability = response.json()['churn_probability']
            st.success(f"✅ Predicted Churn Probability: {churn_probability:.2%}")
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"❌ Exception occurred: {e}")
