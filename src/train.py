import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import joblib

features_path = "../data/processed/features.csv"
print(f"Loading features from {features_path}...")
df = pd.read_csv(features_path)

print("Splitting data into train and test sets...")
X = df.drop(columns=['Attrition'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

mlflow.set_experiment("Employee_Attrition_Prediction")

with mlflow.start_run():
    print("Training Random Forest model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model F1-Score: {f1:.4f}")
    print(f"Model AUC: {auc:.4f}")
    
    # Log parameters and metrics to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    print("Model logged to MLflow successfully!")
    
    # Save model locally
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/random_forest_model.pkl")
    print("Model saved locally to ../models/random_forest_model.pkl")
