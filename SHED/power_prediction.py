import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# === 1. Daten einlesen ===
csv_path = "weather_data.csv"
df = pd.read_csv(csv_path, parse_dates=["Date/Time"])

# === 2. Feature-Engineering für Uhrzeit ===
df["hour"] = df["Date/Time"].dt.hour
df["minute"] = df["Date/Time"].dt.minute

# === 3. Feature- und Zielspalten definieren ===
features = [
    "hour",
    "minute",
    "WEATHER_STATION_AI_CONTROLWEB_BackofModuleTemperature1",
    "WEATHER_STATION_AI_CONTROLWEB_BackofModuleTemperature2",
    "WEATHER_STATION_AI_MGATE_AmbientTemperature",
    "WEATHER_STATION_AI_MGATE_AmbientTemperaturewithOffset",
    "WEATHER_STATION_AI_MGATE_DewpointValue",
    "WEATHER_STATION_AI_MGATE_Humidity",
    "WEATHER_STATION_AI_MGATE_HumidityOffset",
    "WEATHER_STATION_AI_MGATE_HumiditywithOffset",
    "WEATHER_STATION_AI_MGATE_Pyranometer1BodyTemperature",
    "WEATHER_STATION_AI_MGATE_Pyranometer2BodyTemperature",
    "WEATHER_STATION_AI_MGATE_SolarIrradiancePyranometer1",
    "WEATHER_STATION_AI_MGATE_SolarIrradiancePyranometer2",
    "WEATHER_STATION_AI_MGATE_TemperatureOffset",
    "WEATHER_STATION_AI_MGATE_WindDirection",
    "WEATHER_STATION_AI_MGATE_WindDirection_Out",
    "WEATHER_STATION_AI_MGATE_WindSpeed",
]

target = "PV3_C1_AI_P"

X = df[features]
y = df[target]

# === 4. Zeitgesteuertes Splitting ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 5. Modelle definieren und bewerten ===
modelle = {
    "Lineare Regression": LinearRegression(),
    "Entscheidungsbaum": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}

ergebnisse = {}
for name, modell in modelle.items():
    modell.fit(X_train, y_train)
    y_pred = modell.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    ergebnisse[name] = rmse
    print(f"{name}: RMSE = {rmse:.2f}")

beste = min(ergebnisse, key=ergebnisse.get)
print(f"\nBestes Modell: {beste} mit RMSE = {ergebnisse[beste]:.2f}")
