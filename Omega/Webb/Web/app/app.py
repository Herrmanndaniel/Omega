"""Tento modul implementuje Flask webovou aplikaci pro predikci spotřeby paliva vozidel.
Obsahuje trasy pro vykreslení formulářů, provádění predikcí a zobrazení minulých predikcí.
Aplikace využívá předem natrénovaný model Random Forest a OneHotEncoder pro zpracování vstupních dat."""

import csv
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__, template_folder='../templates', static_folder='../static')  # Aktualizované cesty

predictions = []  # Uchovávání predikcí v paměti

# Načtení trénovaného modelu a encoderu
model = joblib.load('random_forest.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def index():
    """
    Zobrazí domovskou stránku aplikace.
    """
    return render_template('index.html')  # Přizpůsobeno nové struktuře složek

@app.route('/form')
def form():
    """
    Zobrazí stránku s formulářem a načte možnosti karoserie, typu motoru a paliva ze souboru CSV.
    """
    body_types = set()
    engine_types = set()
    fuel_types = set()
    with open('../static/form-data.csv', 'r', encoding='utf-8') as csvfile:  # Aktualizovaná cesta
        reader = csv.DictReader(csvfile)
        for row in reader:
            body_types.add(row['Karoserie'])
            engine_types.add(row['Motor'])
            fuel_types.add(row['Palivo'])
    return render_template(
        'form.html',
        body_types=sorted(body_types),
        engine_types=sorted(engine_types),
        fuel_types=sorted(fuel_types)
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Zpracuje požadavek na predikci po odeslání formuláře.
    Načte vstupní data, předzpracuje je a použije trénovaný model pro predikci spotřeby paliva.
    Predikce je uložena do CSV souboru a zároveň vrácena jako JSON odpověď.
    """
    # Načtení vstupních dat z formuláře
    input_data = {
        'Karoserie': request.form['body_type'],
        'Motor': request.form['engine_type'],
        'Palivo': request.form['fuel_type'],
        'Výkon': float(request.form['horsepower']),
        'Stáří vozidla': 2025 - int(request.form['year'])  # Výpočet stáří vozidla
    }

    # Zakódování kategoriálních proměnných pomocí encoderu
    categorical_features = pd.DataFrame([input_data], columns=['Karoserie', 'Palivo', 'Motor'])
    encoded_features = encoder.transform(categorical_features)

    # Spojení zakódovaných kategoriálních s numerickými proměnnými
    numerical_features = pd.DataFrame(
        [[input_data['Výkon'], input_data['Stáří vozidla']]],
        columns=['Výkon', 'Stáří vozidla']
    )

    # Získání názvů zakódovaných sloupců
    encoded_feature_names = list(encoder.get_feature_names_out(['Karoserie', 'Palivo', 'Motor']))

    # Sloučení zakódovaných a numerických dat do jednoho DataFrame
    processed_features = pd.concat(
        [pd.DataFrame(encoded_features, columns=encoded_feature_names), numerical_features],
        axis=1
    )

    # Zajištění, že názvy sloupců jsou typu string (kvůli kompatibilitě s modelem)
    processed_features.columns = processed_features.columns.astype(str)

    # Predikce spotřeby paliva pomocí modelu
    predicted_consumption = round(model.predict(processed_features)[0], 1)

    # Příprava odpovědi
    prediction = {
        'body_type': input_data['Karoserie'],
        'engine_type': input_data['Motor'],
        'fuel_type': input_data['Palivo'],
        'horsepower': input_data['Výkon'],
        'fuel_consumption': predicted_consumption
    }

    predictions.append(prediction)

    # Uložení predikce do CSV souboru
    with open('../static/predictions.csv', 'a', newline='', encoding='utf-8') as csvfile:  # Aktualizovaná cesta
        writer = csv.DictWriter(csvfile, fieldnames=prediction.keys())
        if csvfile.tell() == 0:  # Pokud je soubor prázdný, zapíšeme hlavičku
            writer.writeheader()
        writer.writerow(prediction)

    return jsonify({'fuel_consumption': predicted_consumption})

@app.route('/predictions')
def show_predictions():
    """
    Zobrazí stránku s historií všech předchozích predikcí, které byly provedeny a uloženy do CSV souboru.
    """
    saved_predictions = []
    with open('../static/predictions.csv', 'r', encoding='utf-8') as csvfile:  # Aktualizovaná cesta
        reader = csv.DictReader(csvfile)
        for row in reader:
            saved_predictions.append(row)
    return render_template('predictions.html', predictions=saved_predictions)

if __name__ == '__main__':
    """
    Spustí Flask aplikaci v debug režimu.
    """
    app.run(debug=True)
