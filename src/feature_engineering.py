import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("Loading data...")
df = pd.read_csv("../data/processed/emp_attrition_cleaned.csv")

print("Encoding features...")
# Drop Employee ID
if 'Employee ID' in df.columns:
    df = df.drop(columns=['Employee ID'])
    
# Map target variable
if 'Attrition' in df.columns:
    df['Attrition'] = df['Attrition'].map({'Stayed': 0, 'Left': 1})
    
# One-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Scaling features...")
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint8']).columns.tolist()
if 'Attrition' in numeric_cols:
    numeric_cols.remove('Attrition')

if numeric_cols:
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

output_path = "../data/processed/features.csv"
print(f"Saving engineered features to {output_path}...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print("Feature engineering complete!")
