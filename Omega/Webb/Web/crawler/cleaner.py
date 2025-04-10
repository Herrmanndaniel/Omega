import pandas as pd

# Načteme CSV soubor do DataFrame
df = pd.read_csv("../crawler/vsechna_auta.csv")

# 1. Filtrování: odstranění záznamů, kde je 'Kombinovaná' hodnota null (neúplné záznamy)
df_clean = df[df['Kombinovaná'].notna()]  # Záznamy s platnou hodnotou v 'Kombinovaná'
df_bad = df[df['Kombinovaná'].isna()]  # Záznamy s nulovou hodnotou v 'Kombinovaná'

# Uložení neúplných a čistých záznamů do CSV souborů
df_clean.to_csv("auta_cista.csv", index=False, encoding='utf-8-sig')
print(f"✅ Čistých záznamů: {len(df_clean)}")  # Výpis počtu čistých záznamů
print(f"❌ Neúplných záznamů: {len(df_bad)}")  # Výpis počtu neúplných záznamů

# 2. Filtr: výběr aut, která mají platnou spotřebu, nejsou hybridní, nejsou LPG, a mají benzin nebo diesel
is_spotreba_ok = df_clean['Kombinovaná'].notna()  # Záznamy, které mají platnou spotřebu
is_benzin_or_diesel = df_clean['Palivo'].str.contains('benzín|diesel', case=False, na=False)  # Auta na benzin nebo diesel
is_not_hybrid = ~df_clean['Palivo'].str.contains('hybrid', case=False, na=False)  # Auta, která nejsou hybridní
is_not_lpg = ~df_clean['Palivo'].str.contains('LPG', case=False, na=False)  # Auta, která nejsou LPG
is_not_zaod = ~df_clean['Motor'].str.contains('zaod', case=False, na=False)  # Auta, která nemají označení 'zaod' v motoru

# Filtrace aut, která splňují všechny podmínky
df_clean = df_clean[is_spotreba_ok & is_benzin_or_diesel & is_not_hybrid & is_not_lpg & is_not_zaod]

# Filtrace nevyhovujících záznamů
df_bad = df_clean[~(is_spotreba_ok & is_benzin_or_diesel & is_not_hybrid & is_not_lpg & is_not_zaod)]

# Uložení filtrovaných dat
df_clean.to_csv("auta_cista.csv", index=False, encoding='utf-8-sig')
print(f"✅ Čistých záznamů: {len(df_clean)}")  # Výpis počtu čistých záznamů
print(f"❌ Neúplných záznamů: {len(df_bad)}")  # Výpis počtu neúplných záznamů

# 3. Rozdělení sloupce 'Motor' na 'Motor' a 'Výkon' podle vzoru
df_clean[['Motor', 'Výkon']] = df_clean['Motor'].str.extract(r'^([^,]+),\s*([^,]+)')  # Extrakce 'Motor' a 'Výkon' z jednoho sloupce

# Vyhození sloupce 'Výkon.1', pokud existuje (duplicitní sloupec)
if 'Výkon.1' in df_clean.columns:
    df_clean.drop(columns=['Výkon.1'], inplace=True)

# 4. Funkce pro přiřazení typu motoru na základě velikosti motoru
def convert_engine_type(engine):
    engine = str(engine).strip()  # Odstranění nežádoucích mezer
    if '1.0' in engine:
        return 'I3'  # Tříválec
    elif '1.2' in engine:
        return 'I3'
    elif '1.5' in engine:
        return 'I4'  # Čtyřválec
    elif '1.6' in engine:
        return 'I4'
    elif '1.7' in engine:
        return 'I4'
    elif '1.9' in engine:
        return 'I4'
    elif '2.0' in engine:
        return 'I4'
    elif 'D5' in engine:
        return 'I5'  # Pětiválec
    elif '3.0' in engine:
        return 'I6'  # Šestiválec
    elif 'xDrive30d' in engine:
        return 'I6'
    elif '300 d' in engine:
        return 'I6'
    elif 'E 350 CGI' in engine:
        return 'V6'  # V6 motor
    elif 'S 350 d 4MATIC' in engine:
        return 'V6'
    elif '50 TDI' in engine:
        return 'V6'
    elif '55 TFSI' in engine:
        return 'V6'
    elif 'Flying Spur' in engine:
        return 'V8'  # V8 motor
    elif 'Turbo' in engine or 'L' in engine or 'RS' in engine:
        return None  # Tyto motory nebudeme zpracovávat
    else:
        return 'zaod'  # Neznámý motor, označeno jako 'zaod'

# 5. Aplikace funkce pro úpravu typu motoru na celý sloupec 'Motor'
df_clean['Motor'] = df_clean['Motor'].apply(convert_engine_type)

# 6. Odstranění řádků, kde je v 'Motor' hodnota None (nevalidní hodnoty)
df_clean = df_clean.dropna(subset=['Motor'])

# Uložení výsledného souboru s vyčištěnými daty
df_clean.to_csv('doopravdy_hotove_auta.csv', index=False, encoding='utf-8-sig')
print("✅ Uloženo jako doopravdy_hotove_auta.csv")  # Potvrzení o úspěšném uložení
print(f"Hotovo! Počet záznamů po vyčištění: {len(df_clean)}")  # Výpis počtu záznamů po vyčištění
