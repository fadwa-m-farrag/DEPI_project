# app.py (FastAPI backend)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load the retrained pipeline
model = joblib.load('D:/DEPI Graduation Project/model(1).pkl')

# Create FastAPI app
app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input fields — only raw clean fields now
class UserFeatures(BaseModel):
    city: int
    bd: float
    registered_via: int
    num_25: float
    num_50: float
    num_75: float
    num_985: float
    num_100: float
    num_unq: float
    total_secs: float
    active_days: float
    payment_method_id: int
    payment_plan_days: float
    plan_list_price: float
    actual_amount_paid: float
    is_auto_renew: float
    is_cancel: float
    song_completion_rate: float
    avg_secs_per_song: float
    high_engagement: int
    membership_days: float
    discount_rate: float
    is_trial: int
    auto_renew_but_cancel: int
    age_group: int
    account_age_days: float
    gender: str
    payment_method_risk: str
    has_transaction: int
    no_txn_and_churn: int

@app.post("/predict/")
async def predict_churn(user: UserFeatures):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user.dict()])

    # Predict churn probability using the pipeline
    churn_probability = model.predict_proba(user_df)[0][1]

    return {"churn_probability": churn_probability}
