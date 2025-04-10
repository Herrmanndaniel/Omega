import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Nastavení hlaviček pro simulaci požadavku od skutečného prohlížeče (pomáhá to předejít blokování webem)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'cs-CZ,cs;q=0.9'
}

# Seznam URL pro různé kategorie aut
categories = [
    "https://www.aaaauto.cz/sleva/",
    "https://www.aaaauto.cz/4x4-offroad-suv/",
    "https://www.aaaauto.cz/luxusni-vozy/",
]

def create_session():
    """
    Vytvoří a vrátí objekt session pro odesílání HTTP požadavků. Konfiguruje politiku opakování požadavků v případě
    selhání. Session je nakonfigurována s politikou opakování, která povolí celkem 5 pokusů s exponenciálním zpožděním
    v případě určitých chyb serveru.
    """
    session = requests.Session()
    retries = Retry(
        total=5,  # Počet pokusů o opakování
        backoff_factor=0.5,  # Zpoždění mezi pokusy roste exponenciálně
        status_forcelist=[429, 500, 502, 503, 504],  # Opakovat při těchto kódech stavu
        allowed_methods=['GET']  # Povolit opakování pouze pro GET požadavky
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def clean_text(text):
    """
    Vyčistí vstupní text tím, že odstraní zbytečné mezery a rozdělí text podle libovolného bílého znaku,
    následně znovu spojí. Pokud je text prázdný nebo None, vrátí None.
    """
    return ' '.join(str(text).strip().split()) if text else None

def extract_car_data(session, url):
    """
    Extrahuje data o autě z dané URL. Odesílá HTTP požadavek na stránku, analyzuje HTML a získá požadované
    informace o autě, jako je kombinovaná spotřeba, rok uvedení do provozu, typ karoserie, palivo, motor a výkon.
    """
    try:
        response = session.get(url, headers=headers, timeout=(5, 10))  # HTTP požadavek
        soup = BeautifulSoup(response.text, 'html.parser')  # Parsování HTML

        # Inicializace slovníku pro uchování získaných dat
        data = {
            'Kombinovaná': None,
            'Rok uvedení do provozu': None,
            'Karoserie': None,
            'Palivo': None,
            'Motor': None,
            'Výkon': None
        }

        # Procházení seznamu 'li' elementů a hledání potřebných informací
        for li in soup.find_all('li'):
            text = clean_text(li.get_text())  # Vyčištění textu
            if not text:  # Pokud je text prázdný, přeskočíme
                continue
            strong = li.find('strong')  # Hledání tagu <strong> pro hodnotu
            value = clean_text(strong.text) if strong else None

            # Uložení hodnot podle klíče (např. Kombinovaná, Rok uvedení do provozu, ...)
            if 'Kombinovaná' in text:
                data['Kombinovaná'] = value
            elif 'Rok uvedení do provozu' in text:
                data['Rok uvedení do provozu'] = value
            elif 'Karoserie' in text:
                data['Karoserie'] = value
            elif 'Palivo' in text:
                data['Palivo'] = value
            elif 'Motor' in text:
                data['Motor'] = value
            elif 'Výkon' in text:
                if value:
                    for part in value.split():  # Hledání výkonu ve formátu s 'kw'
                        if 'kw' in part.lower():
                            data['Výkon'] = part.lower()
                            break

        # Pokud chybí kombinovaná spotřeba, zkusíme ji najít na jiném místě na stránce
        if not data['Kombinovaná']:
            spotreba_span = soup.find('span', class_='countbarValue')
            if spotreba_span:
                nested = spotreba_span.find('span')
                if nested:
                    value = clean_text(nested.get_text())
                    if value and 'l/100km' in value:
                        data['Kombinovaná'] = value

        # Pokud jsou získána nějaká data, vrátíme je, jinak vrátíme None
        return data if any(data.values()) else None
    except Exception as e:
        print(f"Chyba: {url} | {str(e)}")  # Pokud dojde k chybě, vypíše chybu
        return None

def get_car_links(session, base_url):
    """
    Získá všechny odkazy na auta z dané stránky kategorie. Procházením HTML hledá odkazy, které obsahují '/car.html'
    a mají parametr 'id='.
    """
    try:
        response = session.get(base_url, headers=headers, timeout=10)  # HTTP požadavek
        soup = BeautifulSoup(response.text, 'html.parser')  # Parsování HTML
        return list(set(
            urljoin(base_url, a['href'])  # Získání úplných URL
            for a in soup.select('a[href*="/car.html"]')  # Výběr odkazů, které obsahují '/car.html'
            if 'id=' in a['href']  # Kontrola, že URL obsahuje parametr 'id='
        ))
    except Exception as e:
        print(f"Chyba při získávání odkazů: {str(e)}")  # Chyba při získávání odkazů
        return []

def process_category(session, category_url, all_results):
    """
    Zpracovává jednu kategorii aut, prochází jednotlivé stránky (až do stránky 30) a získává odkazy na auta.
    Pro každý odkaz na auto zavolá funkci pro extrakci dat o autě a uloží je do seznamu.
    """
    for page in range(1, 30):  # Prochází až 30 stránek
        if page == 1:
            url = category_url
        else:
            # Vytváří URL pro další stránky podle kategorie
            if "sleva" in category_url:
                url = f"{category_url}#!&category=156&page={page}"
            elif "4x4-offroad-suv" in category_url:
                url = f"{category_url}#!&category=15&page={page}"
            elif "luxusni-vozy" in category_url:
                url = f"{category_url}#!&category=35&sort[]=0&sort[]=1&page={page}"
            else:
                url = f"{category_url}#!&page={page}"

        # Získání odkazů na auta z této stránky
        car_links = get_car_links(session, url)
        if not car_links:  # Pokud nejsou žádné odkazy, ukončíme zpracování
            break

        print(f"\n🔹 Stránka {page} ({len(car_links)} aut)")  # Vypíše počet aut na stránce
        for i, link in enumerate(car_links, 1):
            try:
                time.sleep(random.expovariate(1.5))  # Náhodné zpoždění pro zpomalení scrapování
                print(f"Zpracovávám {i}/{len(car_links)}: {link[:70]}...")  # Informace o zpracovávaném odkazu
                car_data = extract_car_data(session, link)  # Extrahování dat o autě
                if car_data:
                    all_results.append(car_data)  # Přidání dat do seznamu výsledků
                    print("✅ OK")
                else:
                    print("❌ Chyby ve zpracování")
            except KeyboardInterrupt:
                print("\n🛑 Ukončeno uživatelem")  # Ukončení, pokud uživatel přeruší proces
                return True
            except Exception as e:
                print(f"⚠️ Chyba: {str(e)}")  # Vypíše chybu, pokud dojde k nějaké neočekávané chybě
    return False

def save_results(all_results):
    """
    Uloží získaná data o autech do CSV souboru. Pokud nejsou žádná data, vypíše chybovou zprávu.
    """
    if all_results:
        df = pd.DataFrame(all_results)  # Převede seznam dat na DataFrame
        df.to_csv('vsechna_auta.csv', index=False, encoding='utf-8-sig')  # Uloží do CSV
        print(f"\n✅ Uloženo {len(all_results)} záznamů do vsechna_auta.csv")
    else:
        print("\n❌ Žádná data k uložení")

def main():
    """
    Hlavní funkce, která spustí celý scraping. Vytvoří session, prochází všechny kategorie aut
    a ukládá získaná data.
    """
    session = create_session()  # Vytvoření session
    all_results = []  # Seznam pro uchování všech výsledků
    try:
        # Pro každou kategorii aut zavolá funkci pro zpracování
        for category_url in categories:
            print(f"\n📂 Zpracovávám kategorii: {category_url}")
            user_stopped = process_category(session, category_url, all_results)
            if user_stopped:  # Pokud uživatel přeruší, ukončíme
                break
    except KeyboardInterrupt:
        print("\n🛑 Ukončeno uživatelem")  # Ukončení při přerušení uživatelem
    finally:
        save_results(all_results)  # Uložení výsledků
        session.close()  # Uzavření session

if __name__ == "__main__":
    main()  # Spuštění hlavní funkce
