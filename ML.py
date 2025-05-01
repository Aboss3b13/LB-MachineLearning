import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from datetime import datetime

# Load dataset
df = pd.read_csv("processed_data.csv")

# Clean up and prepare for model training
df = df.drop(columns=["Column1", "brokered_by", "street", "status"], errors='ignore')
df = df.dropna(subset=["price"])

# Estimate growth for region: Trussville, AL 35173
region_df = df.copy()
region_df = region_df.dropna(subset=["prev_sold_date", "price"])
region_df = region_df[
    (region_df["city"] == "Trussville") &
    (region_df["state"] == "Alabama") &
    (region_df["zip_code"] == 35173)
]

# Parse dates
region_df["prev_sold_date"] = pd.to_datetime(region_df["prev_sold_date"], errors='coerce')
region_df = region_df.dropna(subset=["prev_sold_date"])

# Estimate yearly growth rates
def calc_growth(row):
    years = (datetime.today() - row["prev_sold_date"]).days / 365.25
    if years <= 0 or pd.isna(row["house_size"]):
        return None
    try:
        return (row["price"] / (row["price"] / 1.3))**(1/years) - 1  # Assume 30% increase over holding period
    except:
        return None

region_df["estimated_growth"] = region_df.apply(calc_growth, axis=1)
region_growth = region_df["estimated_growth"].dropna().mean()

# Fall back to 3% if nothing was calculated
if pd.isna(region_growth):
    region_growth = 0.03

# Prepare ML model
df = df.drop(columns=["prev_sold_date"], errors='ignore')
X = df.drop(columns=["price"])
y = df["price"]

# Identify feature types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict current price for the house
new_house = pd.DataFrame([{
    "bed": 3,
    "bath": 3,
    "acre_lot": 0.82,
    "city": "Trussville",
    "state": "Alabama",
    "zip_code": 35173,
    "house_size": 3570
}])
current_price = model.predict(new_house)[0]

# Predict 2-year price using calculated region growth
future_price = current_price * ((1 + region_growth) ** 2)

# Output
print(f"Estimated current price: ${current_price:,.2f}")
print(f"Estimated annual growth for Trussville, AL 35173: {region_growth * 100:.2f}%")
print(f"Predicted price in 2 years: ${future_price:,.2f}")
