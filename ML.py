import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
import joblib
import os

# Load and clean dataset
df = pd.read_csv("realtor-data.zip.csv")
df = df.drop(columns=["Column1", "brokered_by", "street", "status"], errors='ignore')
df = df.dropna(subset=["price"]).copy()

# Function to estimate annual growth for a given region
def estimate_region_growth(df, city, state, zipcode, default_growth=0.03):
    region_df = df.dropna(subset=["prev_sold_date", "price"]).copy()
    region_df = region_df[
        (region_df["city"].str.lower() == city.lower()) &
        (region_df["state"].str.lower() == state.lower()) &
        (region_df["zip_code"] == zipcode)
    ]
    region_df["prev_sold_date"] = pd.to_datetime(region_df["prev_sold_date"], errors='coerce')
    region_df = region_df.dropna(subset=["prev_sold_date"])

    def calc_growth(row):
        years = (datetime.today() - row["prev_sold_date"]).days / 365.25
        if years <= 0:
            return None
        base_price = row["price"] / 1.3
        return (row["price"] / base_price) ** (1/years) - 1

    values = region_df.apply(calc_growth, axis=1).dropna()
    return values.mean() if not values.empty else default_growth

# Prepare and train/load ML pipeline
df_ml = df.drop(columns=["prev_sold_date"], errors='ignore')
X = df_ml.drop(columns=["price"])
y = df_ml["price"]
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
model_path = "house_price_model.pkl"
if os.path.exists(model_path):
    model_pipeline = joblib.load(model_path)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    joblib.dump(model_pipeline, model_path)

# === User inputs ===
new_house = pd.DataFrame([{  
    "bed": 4,
    "bath":  3,
    "acre_lot": 0.13,
    "city": "philadelphia",
    "state": "pennsylvania",
    "zip_code": 19135,
    "house_size": 2850
}])
# Target projection year
projection_year = 2027

city = new_house.loc[0, "city"]
state = new_house.loc[0, "state"]
zipcode = new_house.loc[0, "zip_code"]

# Regional growth estimation
growth_rate = estimate_region_growth(df, city, state, zipcode)

# Predict current price
current_price = model_pipeline.predict(new_house)[0]

# Years until projection_year
current_year = datetime.today().year
years_ahead = projection_year - current_year
if years_ahead <= 0:
    raise ValueError(f"Projection year must be in the future. Got {projection_year}.")

# Future price calculation for calendar year projection_year
future_price = current_price * ((1 + growth_rate) ** years_ahead)

# Implied annual growth over that period
implied_growth = (future_price / current_price) ** (1/years_ahead) - 1

# Output
print(f"Estimated current price: ${current_price:,.2f}")
print(f"Estimated annual regional growth for {city.title()}, {state} {zipcode}: {growth_rate * 100:.2f}%")
print(f"Projected price in year {projection_year}: ${future_price:,.2f}")
print(f"Implied annual growth rate from projection: {implied_growth * 100:.2f}%")