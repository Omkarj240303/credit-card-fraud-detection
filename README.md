## Overview
This project focuses on building a complete end-to-end machine learning system to detect fraudulent financial transactions.
The solution covers the entire pipeline from data preprocessing and model training to deployment using FastAPI for real-time predictions.  

## Objective 
The goal of this project is to accurately identify fraudulent transactions while minimizing missed fraud cases.
Special emphasis is placed on improving recall due to the highly imbalanced nature of the dataset.

## Tech Stack
### Python
### Pandas, NumPy
### Scikit-learn
### XGBoost
### Imbalanced-learn (SMOTE)
### FastAPI
### Joblib

## Data Preprocessing
Removed irrelevant features such as sender and receiver IDs
Handled missing values using appropriate statistical methods
Applied one-hot encoding for categorical features
Addressed skewness and outliers in transaction-related features

## Handling Class Imbalance
The dataset was highly imbalanced (very few fraud cases)
Applied SMOTE (Synthetic Minority Oversampling Technique) on training data
Improved model’s ability to detect fraud patterns

## Model Development
Built a baseline model using XGBoost
Evaluated model performance before and after applying SMOTE
Focused on recall as the primary evaluation metric
Performed hyperparameter tuning on a sampled dataset for efficiency

## Model Performance
Achieved high recall for fraud detection (~92%)
Reduced false negatives significantly (very few fraud cases missed)
Maintained strong overall performance despite class imbalance

## Deployment
Model deployed using FastAPI
Built REST API for real-time fraud prediction
Implemented input validation using Pydantic
Ensured consistent preprocessing using saved model artifact
## Model Structure
fraud_api/
│
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
## sample input
{
  "amount": 5000,
  "sender_old_balance": 10000,
  "sender_new_balance": 5000,
  "receiver_old_balance": 2000,
  "receiver_new_balance": 7000,
  "step": 10,
  "city": "Mumbai",
  "transaction_type": "CASH_OUT"
}
## Sample output
{
  "fraud_prediction": "No",
  "fraud_probability": 0.02
}

# Conclusion
This project demonstrates a practical implementation of a fraud detection system, 
combining machine learning with backend deployment to create a real-world usable solution

