import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import sys

model_path = "../models/random_forest_model.pkl"
data_path = "../data/processed/features.csv"

print(f"Loading model from {model_path}...")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print("Model file not found. Please run train.py first.")
    sys.exit(1)
    
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

X = df.drop(columns=['Attrition'])
y = df['Attrition']

print("Generating predictions...")
y_pred = model.predict(X)

print("\n--- Champion Model Evaluation Report ---")
print("\nClassification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
