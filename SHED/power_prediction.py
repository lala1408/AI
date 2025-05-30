import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

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
    # "WEATHER_STATION_AI_CONTROLWEB_BackofModuleTemperature1",
    # "WEATHER_STATION_AI_CONTROLWEB_BackofModuleTemperature2",
    "WEATHER_STATION_AI_MGATE_AmbientTemperature",
    # "WEATHER_STATION_AI_MGATE_AmbientTemperaturewithOffset",
    "WEATHER_STATION_AI_MGATE_DewpointValue",
    "WEATHER_STATION_AI_MGATE_Humidity",
    # "WEATHER_STATION_AI_MGATE_HumidityOffset",
    # "WEATHER_STATION_AI_MGATE_HumiditywithOffset",
    # "WEATHER_STATION_AI_MGATE_Pyranometer1BodyTemperature",
    # "WEATHER_STATION_AI_MGATE_Pyranometer2BodyTemperature",
    # "WEATHER_STATION_AI_MGATE_SolarIrradiancePyranometer1",
    # "WEATHER_STATION_AI_MGATE_SolarIrradiancePyranometer2",
    # "WEATHER_STATION_AI_MGATE_TemperatureOffset",
    "WEATHER_STATION_AI_MGATE_WindDirection",
    # "WEATHER_STATION_AI_MGATE_WindDirection_Out",
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

# === Korrelationsmatrix und Heatmap ===
corr_matrix = df[features + [target]].corr()

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Korrelationsmatrix der Features und des Ziels", fontsize=16)
plt.tight_layout()

# === 6. Feature-Importances für das beste Modell visualisieren ===
# Nehmen wir Random Forest oder Gradient Boosting (falls sie das beste Modell sind)

if beste in ["Random Forest", "Gradient Boosting"]:
    bestes_modell = modelle[beste]
    importances = bestes_modell.feature_importances_

    feature_importances_df = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="Importance", y="Feature", data=feature_importances_df, palette="viridis"
    )
    plt.title(f"Feature Importances im besten Modell: {beste}", fontsize=16)
    plt.xlabel("Bedeutung")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
else:
    print("Das beste Modell hat keine Feature-Importances (z.B. Lineare Regression).")


plt.show()
