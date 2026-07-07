import numpy as np
import pandas as pd


def preprocess_data():
    """
    Clean the raw employee attrition dataset.

    This function performs the following preprocessing steps:
    - Loads the raw dataset.
    - Removes duplicate records.
    - Handles missing values using the median.
    - Corrects spelling and formatting issues in categorical values.
    - Removes irrelevant features.
    - Saves the cleaned dataset to the processed folder.

    Returns:
        None
    """

    print("Loading Dataset...")
    df = pd.read_csv("../data/raw/emp_attrition_csv.csv")

    print("Cleaning Dataset...")

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Handle Missing Values
    df["Distance from Home"] = df["Distance from Home"].fillna(df["Distance from Home"].median())
    df["Company Tenure (In Months)"] = df["Company Tenure (In Months)"].fillna(df["Company Tenure (In Months)"].median())

    # Spelling Correction
    df["Education Level"] = df["Education Level"].str.strip()
    df["Education Level"] = df["Education Level"].str.replace("Masterâ€™s Degree","Master's Degree")
    df["Education Level"] = df["Education Level"].str.replace("Bachelorâ€™s Degree","Bachelor's Degree")

    # Drop Irrelevant Feature
    df.drop(["Employee ID"], axis=1, inplace=True)

    # Save Cleaned Dataset
    df.to_csv("../data/processed/emp_attrition_cleaned.csv",index=False)

    print("Dataset saved successfully!")


if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"Error: {e}")

