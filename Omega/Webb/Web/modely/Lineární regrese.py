import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
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
X = df.drop(['Kombinovaná', 'Rok uvedení do provozu'], axis=1)  # Vlastnosti (features)
y = df['Kombinovaná']  # Cílová proměnná (target)

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

# Trénování lineárního regresního modelu
model = LinearRegression()
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
cv_scores = cross_val_score(model, X_processed, y, cv=5)
print(f'Cross-val R²: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})')

# Vizualizace výsledků - skutečná vs. predikovaná spotřeba
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Skutečná spotřeba')
plt.ylabel('Predikovaná spotřeba')
plt.title('Linear Regression: Skutečná vs. Predikovaná spotřeba')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

"""
MAE: 0.44
RMSE: 0.62
R²: 0.91
Cross-val R²: 0.58 (±0.66)
"""
