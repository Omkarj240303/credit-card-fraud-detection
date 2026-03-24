from fastapi import FastAPI
from pydantic import BaseModel, Field,computed_field
from typing import Annotated,Literal
import pandas as pd
import joblib

# Initialize app
app = FastAPI()

# Load saved model and preprocessing files
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")

model_virsion = 1.0

# Input schema (validation)
from pydantic import BaseModel, Field
from typing import Annotated

class Transaction(BaseModel):
    amount: Annotated[float, Field(..., gt=0, description="Transaction amount must be greater than 0")]
    sender_old_balance: Annotated[float, Field(..., ge=0, description="Sender's old balance")]
    sender_new_balance: Annotated[float, Field(..., ge=0, description="Sender's new balance")]
    receiver_old_balance: Annotated[float, Field(..., ge=0, description="Receiver's old balance")]
    receiver_new_balance: Annotated[float, Field(..., ge=0, description="Receiver's new balance")]
    step: Annotated[int, Field(..., ge=0, description="Transaction step")]
    city: Annotated[Literal['Delhi','Pune','Bangalore','Mumbai','Lucknow','Nashik','Chennai','kolkata','Jaipur','Hyderabad'], Field(...,description="City of transaction")]
    transaction_type: Annotated[Literal['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], Field(...,description="Type of transaction")]


# Home route
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running "}

@app.get("/health")
def health():
    return {"status": "healthy", "version": model_virsion}

# Prediction route
@app.post("/predict")
def predict(data: Transaction):
    try:
        # Convert input to dict
        input_data = data.dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encoding
        df = pd.get_dummies(df)

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Scaling
        df_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        # Convert to Yes/No
        result = "Yes" if prediction == 1 else "No"

        return {
            "fraud_prediction": result,
            "fraud_probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}