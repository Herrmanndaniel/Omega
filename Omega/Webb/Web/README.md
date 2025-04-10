# Fuel Consumption Predictor

Aplikace pro predikci kombinované spotřeby paliva na základě parametrů vozidla. Postaveno pomocí Pythonu, Flasku a modelu Random Forest Regressor.

---

## 🔧 Instalace a spuštění

### 1. Klonuj repozitář
```bash
git clone https://github.com/vase_uzivatelske_jmeno/fuel-price-predictor.git
cd fuel-price-predictor
```

### 2. Instaluj závislosti
```bash
pip install -r requirements.txt
```

### 3. Vytvoř model a encoder
```bash
python app/model.py
```

### 4. Spusť aplikaci
```bash
python app/app.py
```

Aplikace poběží na: [http://localhost:5000](http://localhost:5000)

---

## 🗂️ Struktura projektu

- 📂 **app/** – Hlavní aplikace
  - 📝 `app.py` – Flask server
  - 🧠 `model.py` – Trénování a načítání modelu
  - 🔒 `encoder.pkl` – Uložený encoder
  - 🌲 `random_forest.pkl` – Uložený model
- 📂 **crawler/** – Získávání a čištění dat
  - 🕷️ `crawler.py` – Skript pro získávání dat
  - 🧹 `cleaner.py` – Skript pro čištění dat
  - 📄 `data.csv` – Stažená data
- 📂 **modely/** – Experimentální modely
  - 🧠 `neuronka.py` – Implementace neuronové sítě
  - 📈 `linearni_regrese.py` – Implementace lineární regrese
  - 🌟 `gradient_boosting.py` – Implementace gradient boosting modelu
- 📂 **static/** – Statické soubory
  - 📄 `form-data.csv` – Vstupní data
  - 📄 `predictions.csv` – Historie predikcí
  - 🎨 `style.css` – Styly pro aplikaci
- 📂 **templates/** – HTML šablony pro renderování stránek
- 📖 `README.md` – Dokumentace projektu
- 🧪 `TestCase.md` – Popis testovacího scénáře

---

## 🔎 API endpointy

| Metoda | Cesta           | Popis                          |
|--------|------------------|---------------------------------|
| GET    | `/`              | Úvodní stránka                 |
| POST   | `/predict`       | Predikce spotřeby paliva      |
| GET    | `/predictions`   | Zobrazení historie predikcí   |

### Příklad JSON vstupu:
```json
{
    "body_type": "Hatchback",
    "engine_type": "Benzín",
    "fuel_type": "Natural 95",
    "horsepower": 85,
    "year": 2017
}
```

### Příklad odpovědi:
```json
{
    "fuel_consumption": 5.6
}
```

---

## ⚖️ Validace vstupů a omezení

- Rok vozidla musí být v rozsahu **1950 - 2025**
- Výkon musí být kladné číslo ≥ 1
- Maximální povolený výkon: **500 kW**
- Povolené typy paliv: `Natural 95`, `Natural 98`, `Diesel`

---

## 🧠 Trénování modelu

Model: `RandomForestRegressor`

Hyperparametry:
```python
n_estimators=200
max_depth=10
min_samples_leaf=2
random_state=42
n_jobs=-1
```

Používané vstupy: `Karoserie`, `Palivo`, `Motor`, `Výkon`, `Stáří vozidla`  
Kategorialní proměnné zakódovány pomocí `OneHotEncoder`

---

## 📊 Výsledky modelu

| Metrika       | Hodnota        |
|---------------|----------------|
| MAE           | 0.52 l/100km   |
| RMSE          | 0.74 l/100km   |
| R²            | 0.93           |
| Cross-val R²  | 0.91 ( ±0.02)  |

---

## 🔁 Příklad volání API (Python)

```python
import requests

url = 'http://localhost:5000/predict'
data = {
    'body_type': 'Sedan',
    'engine_type': 'Diesel',
    'fuel_type': 'Diesel',
    'horsepower': 120,
    'year': 2020
}

response = requests.post(url, json=data)
print(f"Predikovaná spotřeba: {response.json()['fuel_consumption']} l/100km")
```

---
## 🧪 Test Case

Podrobný testovací scénář najdete v souboru [TestCase.md](TestCase.md).


---

## 📚 Licence a autor

- Licence: MIT  
- Autor: Daniel Herrmann
- Verze: 1.0.0
