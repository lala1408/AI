import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# === 1. Daten einlesen ===
df = pd.read_csv("weather_data.csv")

# === 2. Feature- und Zielspalten definieren ===
features = ["Globalstrahlung", "Temperatur", "Windgeschwindigkeit", "Bedeckung"]
target = "PV_Leistung"

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

# === 6. (Optional) Bester RMSE hervorheben ===
beste = min(ergebnisse, key=ergebnisse.get)
print(f"\n👉 Bestes Modell: {beste} mit RMSE = {ergebnisse[beste]:.2f}")

# === 7. Vorhersage für neuen Datensatz mit ALLEN Modellen ===
neuer_datensatz = pd.DataFrame([{
    "Globalstrahlung": 800,
    "Temperatur": 30,
    "Windgeschwindigkeit": 2,
    "Bedeckung": 0.1
}])


print("\n📊 Vorhersage für neuen Datensatz:")
for name, modell in modelle.items():
    vorhersage = modell.predict(neuer_datensatz)[0]
    print(f"{name}: {vorhersage:.2f} W")

