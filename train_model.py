"""
Trains a regression model to predict house prices.
Uses a realistic synthetic dataset (no internet download needed),
so it works instantly and reproducibly on any machine.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

np.random.seed(42)
n = 2000

# Feature generation (realistic ranges, Indian metro context)
area_sqft = np.random.normal(1200, 400, n).clip(300, 4000)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
age_years = np.random.randint(0, 30, n)
distance_from_city_km = np.random.uniform(0.5, 25, n)
locality_score = np.random.uniform(1, 10, n)  # 1=low demand, 10=prime locality

# Price formula with noise (in lakhs INR)
price = (
    area_sqft * 0.045
    + bedrooms * 8
    + bathrooms * 5
    - age_years * 0.8
    - distance_from_city_km * 1.2
    + locality_score * 6
    + np.random.normal(0, 8, n)
)
price = price.clip(10, None)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_from_city_km": distance_from_city_km,
    "locality_score": locality_score,
    "price_lakhs": price
})

df.to_csv("housing_data.csv", index=False)

X = df.drop("price_lakhs", axis=1)
y = df["price_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f} lakhs")
print(f"R2 Score: {r2:.3f}")

joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")
