import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Načtení dat ze souboru
file_path = '../cleaner/doopravdy_hotove_auta.csv'
df = pd.read_csv(file_path)

# Úprava cílové proměnné - odstranění jednotky " l/100km" a převod na float
df['Kombinovaná'] = df['Kombinovaná'].str.replace(' l/100km', '').astype(float)

# Úprava sloupce Výkon - odstranění jednotky "kW" a převod na float
df['Výkon'] = df['Výkon'].str.replace('kW', '').astype(float)

# Přidání nového atributu: stáří vozidla (aktuální rok mínus rok uvedení do provozu)
current_year = 2025
df['Stáří vozidla'] = current_year - df['Rok uvedení do provozu']

# Příprava dat (vlastnosti a cílová proměnná)
X = df.drop(['Kombinovaná', 'Rok uvedení do provozu'], axis=1)
y = df['Kombinovaná']

# Kódování kategoriálních proměnných pomocí One-Hot Encoding
categorical_cols = ['Karoserie', 'Palivo', 'Motor']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Spojení kódovaných kategoriálních vlastností s numerickými vlastnostmi
numerical_cols = ['Výkon', 'Stáří vozidla']
X_processed = pd.concat([
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols)),
    X[numerical_cols].reset_index(drop=True)
], axis=1)

# Standardizace numerických vlastností pro zlepšení výkonu modelu
scaler = StandardScaler()
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

# Rozdělení dat na trénovací a testovací sady
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Vytvoření a trénování modelu Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Predikce na testovací sadě
y_pred = model.predict(X_test)

# Výpočet metrik výkonu modelu
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

# Cross-validace pro ověření stability modelu
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
print(f'Cross-val R²: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})')

# Vizualizace důležitosti proměnných
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10,6))
plt.title('Důležitost proměnných')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X_processed.columns[i] for i in indices])
plt.xlabel('Relativní důležitost')
plt.show()

"""
MAE: 0.09
RMSE: 0.15
R²: 1.00
Cross-val R²: 0.88 (±0.22)
"""


