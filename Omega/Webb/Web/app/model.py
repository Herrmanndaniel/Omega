import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib

# ---------------------------
# 1. Načtení a úprava dat
# ---------------------------

# Načtení datového souboru
file_path = 'C:/Users/danhe/Downloads/Webb/Web/crawler/doopravdy_hotove_auta.csv'
df = pd.read_csv(file_path)

# Převedení spotřeby na číselný formát (odstranění jednotek)
df['Kombinovaná'] = df['Kombinovaná'].str.replace(' l/100km', '').astype(float)

# Převedení výkonu motoru na číselný formát
df['Výkon'] = df['Výkon'].str.replace('kW', '').astype(float)

# Výpočet stáří vozidla z roku uvedení do provozu
df['Stáří vozidla'] = 2025 - df['Rok uvedení do provozu']

# Odstranění chybných nebo nerelevantních záznamů (např. 'zaod' v poli Motor)
df = df[df['Motor'] != 'zaod']

# ---------------------------
# 2. Příprava vstupních a cílových dat
# ---------------------------

# Vstupní proměnné (bez cílové proměnné a roku)
X = df.drop(['Kombinovaná', 'Rok uvedení do provozu'], axis=1)

# Cílová proměnná – kombinovaná spotřeba
y = df['Kombinovaná']

# ---------------------------
# 3. Kódování kategoriálních proměnných
# ---------------------------

# Definování kategoriálních sloupců
categorical_cols = ['Karoserie', 'Palivo', 'Motor']

# Použití OneHotEncoder pro zakódování kategorií
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Spojení zakódovaných dat s číselnými vstupy (výkon a stáří)
feature_names = list(encoder.get_feature_names_out(categorical_cols)) + ['Výkon', 'Stáří vozidla']
X_processed = pd.DataFrame(
    np.hstack([X_encoded, X[['Výkon', 'Stáří vozidla']]]),
    columns=feature_names
)

# ---------------------------
# 4. Rozdělení dat
# ---------------------------

# Rozdělení na trénovací a testovací sadu (80 % trénink, 20 % test)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Trénování modelu
# ---------------------------

# Inicializace a trénink Random Forest modelu
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------
# 6. Uložení modelu a encoderu
# ---------------------------

# Serializace modelu a encoderu do souboru pomocí joblib
joblib.dump(model, 'random_forest.pkl')
joblib.dump(encoder, 'encoder.pkl')
print("Model a encoder byly úspěšně uloženy.")

# ---------------------------
# 7. Vyhodnocení modelu
# ---------------------------

# Predikce na testovacích datech
y_pred = model.predict(X_test)

# Výpočet metrik přesnosti
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Výpis metrik
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

# ---------------------------
# 8. Cross-validace modelu
# ---------------------------

# 5-fold cross-validace a výpočet průměrného R² skóre
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
print(f'Cross-val R²: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})')

"""
MAE: 0.06
RMSE: 0.10
R²: 1.00
Cross-val R²: 0.86 (±0.26)
"""

