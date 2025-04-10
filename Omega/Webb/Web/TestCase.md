# Test Case – Spotřeba Paliva

**Test Case ID:** Dan_spotřeba  
**Test Done by:** Oliver Hrazdíra
**Test Name:** Verify prediction with valid input data  
**Brief description:** Ověření, že webová aplikace správně zpracuje vstupní údaje uživatele a vrátí predikci spotřeby paliva.

## Pre-conditions:
- Model (`random_forest.pkl`) a encoder (`encoder.pkl`) musí být natrénován a uložen.
- Musí existovat soubor `form-data.csv` s validními daty.
- Flask server běží na localhostu.

## Dependencies and Requirements:
- Python 3.x, Flask, pandas, numpy, joblib, sklearn
- Webový prohlížeč
- Přístup ke složce `static` pro čtení/zápis dat

## Test Steps:

| Krok | Popis | Testovací Data | Očekávaný Výsledek | Poznámky |
|------|-------|----------------|---------------------|----------|
| 1 | Otevři stránku `/form` ve webovém prohlížeči | – | Formulář je načten s možnostmi pro Karoserii, Palivo a Motor | |
| 2 | Vyplň formulář platnými údaji | Karoserie: Sedan, Motor: 1.6 TDI, Palivo: Nafta, Výkon: 85, Rok: 2018 | Formulář se úspěšně odešle | |
| 3 | Odešli formulář a sleduj výstup | – | Vrátí JSON s predikovanou spotřebou paliva (např. `{"fuel_consumption": 5.4}`) | |
| 4 | Otevři `/predictions` | – | Stránka zobrazí poslední i předchozí predikce | |
| 5 | Zkontroluj soubor `form-data.csv` | – | Nová predikce byla zapsána do souboru jako nový řádek | |
