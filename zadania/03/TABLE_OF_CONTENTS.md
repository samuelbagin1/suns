# Analyza
**Analýza, párovanie a rozdelenie dát** — Uveď počet a rozlíšenie obrázkov, stĺpce v `metadata.csv` vrátane ingrediencií a chýbajúce hodnoty. Over, že každý obrázok má práve jeden záznam, spáruj ho s `total_calories` cez `file_name` a dáta rozdeľ na trénovaciu, validačnú a testovaciu množinu.

# CNN
## Generator dat
**Generátory dát** — Popíš generátory pre všetky tri množiny (train, val a test), normalizáciu, zmenu rozmerov obrázkov a nastavenie `shuffle`.

## Trenovanie
### Návrh siete a konfigurácie hyperparametrov
Uveď aspoň dve konvolučné vrstvy, aspoň dve regularizačné alebo normalizačné techniky a aspoň päť cielených konfigurácií. 

### Priebeh trénovania a vyhodnotenie
Pre najlepší model zobraz priebeh trénovania, reziduály a predikované oproti skutočným hodnotám. Vyhodnoť trénovaciu aj testovaciu množinu a posúď podtrénovanie či pretrénovanie. V jednej tabuľke porovnaj MSE, MAE, RMSE a `R²` na trénovacej aj testovacej množine.


# Zhlukovanie pomocou transfer learning
## MobilNet
Uveď zvolenú sieť predtrénovanú na ImageNet a počet jej parametrov. Pre celý dataset ulož každý príznak do samostatného stĺpca spolu s cestou k obrázku a `total_calories`.

## Zhlukovanie príznakov
Na zhlukovanie použi iba obrazové príznaky. Podľa potreby zmenši ich dimenziu. Ak vznikne viac než desať zhlukov, do dokumentácie vyber najzaujímavejšie.

1. Uveď použitý zhlukovací algoritmus.
2. Ukáž výber fotografií z každého zhluku.
3. Pre každý zhluk zobraz priemerný obrázok.
3. **Analýza výsledkov zhlukovania** — Opíš, čo majú fotografie v zhlukoch spoločné, ako sa to prejavuje na priemerných obrázkoch a ako sa líšia priemerné kalorické hodnoty zhlukov.


# Natrénujte sieť na vygenerovaných príznakoch
Dosiahni lepšie výsledky než s vlastnou CNN. Natrénuj regresor z predošlých zadaní na extrahovaných príznakoch. Podľa potreby použi redukciu dimenzie. Uz nemen tak ako je v jupyter notebooku.

V jednej tabuľke porovnaj MSE, MAE, RMSE a `R²` na trénovacej aj testovacej množine.


# Bonus
## Zobrazenie filtrov z konvolučných vrstiev - BONUS
Zobraz filtre farebne a popíš, čo zachytávajú.

## Regresia pomocou Transformera - BONUS
Použi na regresiu model založený na architektúre Transformerov.