import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# === 1. Daten einlesen ===
df = pd.read_csv("weather_data.csv")

# === 2. Feature- und Zielspalten definieren ===
features = [
    "temperature_2_m_above_gnd",
    "relative_humidity_2_m_above_gnd",
    "mean_sea_level_pressure_MSL",
    "total_precipitation_sfc",
    "snowfall_amount_sfc",
    "total_cloud_cover_sfc",
    "high_cloud_cover_high_cld_lay",
    "medium_cloud_cover_mid_cld_lay",
    "low_cloud_cover_low_cld_lay",
    "shortwave_radiation_backwards_sfc",
    "wind_speed_10_m_above_gnd",
    "wind_direction_10_m_above_gnd",
    "wind_speed_80_m_above_gnd",
    "wind_direction_80_m_above_gnd",
    "wind_speed_900_mb",
    "wind_direction_900_mb",
    "wind_gust_10_m_above_gnd",
    "angle_of_incidence",
    "zenith",
    "azimuth"
]

target = "generated_power_kw"

X = df[features]
y = df[target]

# === 3. Trainings- und Testdaten aufteilen ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Modelle definieren ===
modelle = {
    "Lineare Regression": LinearRegression(),
    "Entscheidungsbaum": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42)
}

# === 5. Modelle trainieren und vergleichen ===
ergebnisse = {}

for name, modell in modelle.items():
    modell.fit(X_train, y_train)
    y_pred = modell.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    ergebnisse[name] = rmse
    print(f"{name}: RMSE = {rmse:.2f}")

# === 6. Bestes Modell identifizieren ===
beste = min(ergebnisse, key=ergebnisse.get)
print(f"\n👉 Bestes Modell: {beste} mit RMSE = {ergebnisse[beste]:.2f}")

# === 7. Beispielwert prüfen ===
row = 9
neuer_datensatz = df[features].iloc[[row]]

print("\n📊 Vorhersage für neuen Datensatz:")
for name, modell in modelle.items():
    vorhersage = modell.predict(neuer_datensatz)[0]
    print(f"{name}: {vorhersage:.2f} kW")

print(f"Daten: {df[target].iloc[row]:.2f} kW")
