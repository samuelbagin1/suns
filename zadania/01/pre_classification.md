# Príprava dát a porovnanie klasifikačných modelov

Bloky kopíruj do `main.ipynb` postupne za čistenie dát; vysvetlenia pod nimi patria do Markdown buniek. Vstupom sú vyčistené dáta z notebooku, bez opätovného načítania CSV.

## 1. Importy a nastavenie experimentov

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    ConfusionMatrixDisplay, f1_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, PredefinedSplit,
    StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# V aktuálnom main.ipynb sa vyčistený DataFrame volá data_clean.
# Ak už máš clean_data, použije sa priamo táto premenná.
if "clean_data" not in globals():
    clean_data = data_clean.copy()

RANDOM_STATE = 42
USE_CROSS_VALIDATION = True  # False: ladenie na validačnej množine
CV_FOLDS = 5
N_JOBS = -1                # Nastav 1, ak chceš obmedziť paralelizáciu.
TOP_FEATURES = 10
PERMUTATION_REPEATS = 10

SPLIT_RATIOS = {
    "70/15/15": (0.70, 0.15, 0.15),
    "80/10/10": (0.80, 0.10, 0.10),
    "70/10/20": (0.70, 0.10, 0.20),
    "60/10/30": (0.60, 0.10, 0.30),
}
skf = StratifiedKFold(
    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
)
```

Premenná `clean_data` nadväzuje na existujúce `data_clean`, takže sa používa výsledok čistenia v notebooku. Nastavenia umožňujú meniť pomery a zapnúť 5-fold cross-validation. Pevné `random_state` umožňuje zopakovať experimenty s rovnakým rozdelením dát.

## 2. Vstupy X, výstup y a kategorické stĺpce

```python
if clean_data["Revenue"].isna().any():
    raise ValueError("Cieľový stĺpec Revenue obsahuje chýbajúce hodnoty.")
if not clean_data["Revenue"].isin([False, True]).all():
    raise ValueError("Revenue musí obsahovať iba False/True alebo 0/1.")

X = clean_data.drop(columns=["SessionID", "Revenue"]).copy()
y = clean_data["Revenue"].astype(int).copy()

# Weekend má iba dve možnosti, preto stačí jeden binárny stĺpec.
X["Weekend"] = X["Weekend"].map({False: 0, True: 1})
binary_columns = ["Weekend"]

# Tieto číselné identifikátory tiež označujú kategórie, nie veľkosť.
category_codes = ["OperatingSystems", "Browser", "Region", "TrafficType"]
categorical_columns = list(dict.fromkeys(
    X.select_dtypes(include=["object", "string", "category", "bool"])
    .columns.tolist()
    + [column for column in category_codes if column in X.columns]
))
categorical_columns = [
    column for column in categorical_columns if column not in binary_columns
]
numeric_columns = [
    column for column in X.columns
    if column not in categorical_columns + binary_columns
]

print(f"X: {X.shape}, y: {y.shape}")
print("Kategorické stĺpce:", categorical_columns)
print("Binárne stĺpce (0/1):", binary_columns)
display(y.value_counts(normalize=True).rename("Podiel tried"))
```

`X` obsahuje vlastnosti návštevy a `y` označuje, či návšteva skončila nákupom (`0` = bez nákupu, `1` = nákup); identifikátor `SessionID` aj cieľový stĺpec `Revenue` vynechávam zo vstupov. `Weekend` prevádzam na jediný stĺpec s hodnotami `0` pre pracovný deň a `1` pre víkend, takže nepotrebuje One-Hot Encoding. Aj kódy prehliadača, operačného systému, regiónu a návštevnosti považujem za kategórie, lebo vyšší kód neznamená vyššiu hodnotu vlastnosti.

## 3. Tréningová, validačná a testovacia množina

```python
def split_dataset(X, y, ratios):
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("Pomery musia byť kladné a ich súčet musí byť 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio,
        stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        stratify=y_train_val, random_state=RANDOM_STATE,
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

experiment_splits = {
    name: split_dataset(X, y, ratios)
    for name, ratios in SPLIT_RATIOS.items()
}

# Bežné premenné pre vlastné pokusy v ďalších bunkách notebooku.
SELECTED_SPLIT = "70/15/15"
selected_data = experiment_splits[SELECTED_SPLIT]
X_train, X_val, X_test = (
    selected_data[key] for key in ["X_train", "X_val", "X_test"]
)
y_train, y_val, y_test = (
    selected_data[key] for key in ["y_train", "y_val", "y_test"]
)
print("Počet riadkov train/val/test:", len(y_train), len(y_val), len(y_test))
```

Vytváram všetky štyri požadované rozdelenia, pričom `stratify` približne zachováva podiel nákupov a návštev bez nákupu. Najprv oddelím testovacie dáta a potom zvyšok rozdelím na tréning a validáciu; počty sa môžu mierne líšiť od pomerov pre zaokrúhlenie na celé riadky. Tréning slúži na učenie a cross-validation, validácia na výber modelu a test na záverečné vyhodnotenie.

## 4. One-Hot Encoding

```python
categorical_preparation = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

One-Hot Encoding vytvorí samostatný binárny stĺpec pre každú kategóriu, preto je vhodný pre diskrétne nominálne dáta, napríklad `VisitorType` a `Month`. Kategóriám nepriraďuje umelé poradie, vzdialenosť ani vzťah medzi nimi; to však neznamená, že výsledné stĺpce sú štatisticky nezávislé. `handle_unknown="ignore"` umožní spracovať aj kategóriu, ktorá sa v tréningu nevyskytla, a imputácia doplní prípadné chýbajúce hodnoty najčastejšou tréningovou kategóriou. [Dokumentácia OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

## 5. Škálovanie bez úniku informácií

```python
numeric_preparation = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preparation, numeric_columns),
    ("categorical", categorical_preparation, categorical_columns),
    ("binary", SimpleImputer(strategy="most_frequent"), binary_columns),
], remainder="drop")

def make_model_pipeline(classifier):
    return Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", clone(classifier)),
    ])
```

`StandardScaler` škáluje číselné vlastnosti na približne nulový priemer a jednotkový rozptyl, aby napríklad čas v sekundách neprevažoval nad mierami odchodu pri učení logistickej regresie. One-Hot stĺpce aj jediný stĺpec `Weekend` ponechávam v hodnotách `0/1`, pričom prípadné chýbajúce hodnoty dopĺňam mediánom pri číselných vlastnostiach a najčastejšou hodnotou pri kategóriách a víkende. Celú prípravu dávam do `Pipeline`, aby sa imputácia, kódovanie aj škálovanie učili iba na tréningovej časti každého foldu.

## 6. Tri modely a mriežky parametrov

```python
parameters = {
    "max_features": [4, 7, 10, 13],
    "min_samples_leaf": [1, 3, 5, 7],
    "max_depth": [5, 10, 15, 20],
}

model_configs = {
    "Random Forest": {
        "classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        # Prefix model__ je potrebný, pretože klasifikátor je v Pipeline.
        "grid": {f"model__{name}": values for name, values in parameters.items()},
    },
    "Logistická regresia": {
        "classifier": LogisticRegression(
            solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE,
        ),
        "grid": {
            "model__C": np.logspace(-5, 0, 6),
            "model__class_weight": [None, "balanced"],
        },
    },
    "Decision Tree": {
        "classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "grid": {
            "model__max_depth": [3, 5, 10, 15, None],
            "model__min_samples_leaf": [1, 5],
        },
    },
}

for name, config in model_configs.items():
    print(f"{name}: {len(ParameterGrid(config['grid']))} kombinácií")
```

Random Forest kombinuje viacero náhodne vytvorených stromov a používa presne zadanú mriežku 64 kombinácií, v ktorej sa ladí počet skúšaných vlastností pri rozdelení uzla, minimálna veľkosť listu a maximálna hĺbka. Pri logistickej regresii podľa článku skúšam šesť hodnôt `C` na logaritmickej škále a dve nastavenia váh tried, teda 12 kombinácií; menšie `C` znamená silnejšiu regularizáciu a Decision Tree má 10 kombinácií. Logistická regresia vytvára lineárnu hranicu, preto bez rozšírenia vstupov nevystihuje niektoré nelineárne vzťahy, napríklad XOR, ktoré stromy dokážu modelovať pomocou postupných rozdelení. Postup vychádza z návodov na [Random Forest](https://mlcourse.ai/book/topic05/topic5_part2_random_forest.html), [možnosti a obmedzenia logistickej regresie](https://mlcourse.ai/book/topic04/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.html) a [Decision Tree](https://mlcourse.ai/book/topic03/topic03_decision_trees_kNN.html).

## 7. GridSearchCV a možnosť vypnúť cross-validation

```python
def train_with_grid_search(config, split):
    pipeline = make_model_pipeline(config["classifier"])

    if USE_CROSS_VALIDATION:
        X_search, y_search = split["X_train"], split["y_train"]
        if y_search.value_counts().min() < CV_FOLDS:
            raise ValueError("Každá trieda potrebuje aspoň CV_FOLDS riadkov.")
        search_cv = skf
    else:
        # -1 = tréningový riadok, 0 = validačný riadok jediného splitu.
        X_search = pd.concat([split["X_train"], split["X_val"]])
        y_search = pd.concat([split["y_train"], split["y_val"]])
        fold_ids = np.concatenate([
            np.full(len(split["y_train"]), -1),
            np.zeros(len(split["y_val"]), dtype=int),
        ])
        search_cv = PredefinedSplit(fold_ids)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["grid"],
        scoring="accuracy",
        cv=search_cv,
        refit=False,
        return_train_score=True,
        n_jobs=N_JOBS,
        error_score="raise",
        verbose=1,
    )
    search.fit(X_search, y_search)

    # Finálny model sa v oboch režimoch učí iba na tréningovej množine.
    fitted_model = clone(pipeline).set_params(**search.best_params_)
    fitted_model.fit(split["X_train"], split["y_train"])
    return search, fitted_model
```

Pri zapnutej cross-validation vyberá `GridSearchCV` parametre podľa priemernej accuracy v piatich stratifikovaných foldoch tréningovej množiny, takže externá validácia zostáva oddelená. Pri vypnutí použije `PredefinedSplit` priamo tréning a validáciu, takže validačný výsledok už slúži aj na ladenie parametrov a je menej nezávislý. `refit=False` umožňuje výslovne natrénovať vybranú konfiguráciu iba na tréningu a paralelizácia prebieha v GridSearchCV, aby sa neprekrývala s paralelizáciou lesa. [Dokumentácia GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

## 8. Štyri experimenty a spoločná tabuľka úspešností

```python
experiment_rows = []
trained_models = {}
grid_searches = {}

for split_name, split in experiment_splits.items():
    for model_name, config in model_configs.items():
        print(f"\nRozdelenie {split_name} | {model_name}")
        search, fitted_model = train_with_grid_search(config, split)
        key = (split_name, model_name)
        trained_models[key] = fitted_model
        grid_searches[key] = search

        train_prediction = fitted_model.predict(split["X_train"])
        val_prediction = fitted_model.predict(split["X_val"])

        experiment_rows.append({
            "Rozdelenie": split_name,
            "Model": model_name,
            "Train n": len(split["y_train"]),
            "Val n": len(split["y_val"]),
            "Test n": len(split["y_test"]),
            "Train accuracy": accuracy_score(split["y_train"], train_prediction),
            "Val accuracy": accuracy_score(split["y_val"], val_prediction),
            "Grid accuracy": search.best_score_,
            "Grid std": search.cv_results_["std_test_score"][search.best_index_],
            "Najlepšie parametre": search.best_params_,
        })
        print("GridSearchCV.best_params_:", search.best_params_)
        print("GridSearchCV.best_score_:", search.best_score_)

results = pd.DataFrame(experiment_rows)

# Model vyberáme pred vyhodnotením na testovacích dátach.
best_row = results.sort_values(
    ["Val accuracy", "Grid accuracy"], ascending=False, kind="stable"
).iloc[0]
best_key = (best_row["Rozdelenie"], best_row["Model"])
best_model = trained_models[best_key]
best_split = experiment_splits[best_key[0]]

# Testy teraz iba vyhodnotíme; nepoužijeme ich na zmenu best_key.
for row in experiment_rows:
    key = (row["Rozdelenie"], row["Model"])
    split = experiment_splits[key[0]]
    prediction = trained_models[key].predict(split["X_test"])
    row.update({
        "Test accuracy": accuracy_score(split["y_test"], prediction),
        "Test balanced accuracy": balanced_accuracy_score(split["y_test"], prediction),
        "Test F1": f1_score(split["y_test"], prediction, zero_division=0),
        "Test recall": recall_score(split["y_test"], prediction, zero_division=0),
        "Baseline test accuracy": accuracy_score(
            split["y_test"],
            np.full(len(prediction), split["y_train"].mode().iloc[0]),
        ),
    })

results = pd.DataFrame(experiment_rows)
results["Train-test rozdiel"] = results["Train accuracy"] - results["Test accuracy"]
experiment_table = results.drop(columns="Najlepšie parametre")
display(experiment_table.round(4))

print("Vybraný model a rozdelenie:", best_key)
print("Vybrané parametre:", grid_searches[best_key].best_params_)
print("Grid accuracy =", f"priemer {CV_FOLDS}-fold CV" if USE_CROSS_VALIDATION
      else "accuracy na validačnej množine")
if not USE_CROSS_VALIDATION:
    print("Grid std = 0 znamená jediný validačný split, nie nulovú neistotu.")
```

Výsledky overeného behu s aktuálnymi nastaveniami; po zmene parametrov použi novo vypočítanú tabuľku z kódu:

| Rozdelenie | Model | Train accuracy (%) | Val accuracy (%) | Test accuracy (%) |
| --- | --- | ---: | ---: | ---: |
| 70/15/15 | Random Forest | 94,55 | 90,16 | 89,13 |
| 70/15/15 | Logistická regresia | 88,06 | 87,51 | 87,23 |
| 70/15/15 | Decision Tree | 89,84 | 89,87 | 87,92 |
| 80/10/10 | Random Forest | 93,41 | 90,94 | 88,44 |
| 80/10/10 | Logistická regresia | 88,01 | 89,04 | 87,14 |
| 80/10/10 | Decision Tree | 90,75 | 90,51 | 88,70 |
| 70/10/20 | Random Forest | 93,59 | 90,77 | 89,99 |
| 70/10/20 | Logistická regresia | 87,76 | 89,47 | 87,66 |
| 70/10/20 | Decision Tree | 90,76 | 89,82 | 88,74 |
| 60/10/30 | Random Forest | 94,73 | 91,46 | 89,61 |
| 60/10/30 | Logistická regresia | 88,06 | 88,18 | 87,74 |
| 60/10/30 | Decision Tree | 90,91 | 90,60 | 89,12 |

Spoločná tabuľka má 12 riadkov: všetky tri modely pre každé zo štyroch rozdelení, s tréningovou, validačnou aj testovacou úspešnosťou. Najlepší model vyberám podľa validačnej accuracy a pri zhode podľa výsledku grid-searchu; testovacie výsledky slúžia až na následné porovnanie. Keďže triedy môžu byť nevyvážené, dopĺňam baseline väčšinovej triedy, balanced accuracy a F1 aj recall pre nákup, aby vysoká accuracy nezakrývala slabé rozpoznávanie nákupov.

## 9. Druhá tabuľka: vyhodnotenie cross-validation

```python
cv_rows = []
cv_table = pd.DataFrame()

if USE_CROSS_VALIDATION:
    for (split_name, model_name), search in grid_searches.items():
        if search.n_splits_ != CV_FOLDS:
            raise ValueError(
                "Najprv zopakuj tréning so zapnutou cross-validation "
                "a aktuálnym počtom foldov."
            )

        best_index = search.best_index_
        scores = search.cv_results_
        train_mean = scores["mean_train_score"][best_index]
        val_mean = scores["mean_test_score"][best_index]

        cv_rows.append({
            "Rozdelenie": split_name,
            "Model": model_name,
            "Počet foldov": search.n_splits_,
            "CV train accuracy (%)": 100 * train_mean,
            "CV val accuracy (%)": 100 * val_mean,
            "CV std (p. b.)": 100 * scores["std_test_score"][best_index],
            "CV train-val rozdiel (p. b.)": 100 * (train_mean - val_mean),
            "Najlepšie parametre": search.best_params_,
        })

    cv_table = pd.DataFrame(cv_rows)
    display(cv_table.round(2))
else:
    print(
        "Tabuľka cross-validation vyžaduje USE_CROSS_VALIDATION = True; "
        "potom znovu spusti tréning a túto bunku."
    )
```

Výsledky overeného 5-fold behu s aktuálnymi nastaveniami:

| Rozdelenie | Model | Foldy | CV train (%) | CV val (%) | CV std (p. b.) | Train–val (p. b.) | Najlepšie parametre |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 70/15/15 | Random Forest | 5 | 94,44 | 90,35 | 0,39 | 4,09 | `max_depth=20, max_features=13, min_samples_leaf=5` |
| 70/15/15 | Logistická regresia | 5 | 88,09 | 87,92 | 0,36 | 0,17 | `C=0.1, class_weight=None` |
| 70/15/15 | Decision Tree | 5 | 89,79 | 89,44 | 0,60 | 0,35 | `max_depth=3, min_samples_leaf=1` |
| 80/10/10 | Random Forest | 5 | 93,46 | 90,02 | 0,50 | 3,44 | `max_depth=20, max_features=13, min_samples_leaf=7` |
| 80/10/10 | Logistická regresia | 5 | 88,07 | 87,85 | 0,54 | 0,23 | `C=1, class_weight=None` |
| 80/10/10 | Decision Tree | 5 | 90,73 | 89,33 | 0,38 | 1,41 | `max_depth=5, min_samples_leaf=1` |
| 70/10/20 | Random Forest | 5 | 93,91 | 90,18 | 0,31 | 3,73 | `max_depth=10, max_features=13, min_samples_leaf=3` |
| 70/10/20 | Logistická regresia | 5 | 87,83 | 87,41 | 0,49 | 0,43 | `C=1, class_weight=None` |
| 70/10/20 | Decision Tree | 5 | 90,90 | 89,92 | 0,46 | 0,97 | `max_depth=5, min_samples_leaf=1` |
| 60/10/30 | Random Forest | 5 | 95,46 | 90,07 | 0,31 | 5,39 | `max_depth=10, max_features=13, min_samples_leaf=1` |
| 60/10/30 | Logistická regresia | 5 | 88,11 | 87,88 | 0,65 | 0,23 | `C=1, class_weight=None` |
| 60/10/30 | Decision Tree | 5 | 90,91 | 89,44 | 0,72 | 1,48 | `max_depth=5, min_samples_leaf=5` |

Každý riadok vyhodnocuje najlepšiu konfiguráciu daného modelu v jednom rozdelení, pričom päť validačných foldov vzniká iba z tréningovej množiny. Uvádzam priemernú tréningovú a validačnú accuracy, štandardnú odchýlku validačnej accuracy v percentuálnych bodoch a rozdiel train–val, ktorý môže naznačiť pretrénovanie. Výsledky CV boli použité pri ladení parametrov, preto ich porovnávam aj s výsledkami na samostatnej validačnej a testovacej množine v prvej tabuľke.

## 10. Najlepší a najhorší výsledok, pretrénovanie a vplyv rozdelenia

```python
# Podrobnosti všetkých prehľadávaných konfigurácií vybraného modelu.
best_search = grid_searches[best_key]
grid_details = pd.DataFrame(best_search.cv_results_)[[
    "params", "mean_train_score", "mean_test_score",
    "std_test_score", "rank_test_score",
]].sort_values("rank_test_score")
display(grid_details.reset_index(drop=True).round(4))

best_grid = grid_details.iloc[0]
worst_grid = grid_details.sort_values("mean_test_score").iloc[0]
print(f"Najlepšia konfigurácia: {best_grid['mean_test_score']:.4f}")
print(best_grid["params"])
print(f"Najhoršia konfigurácia: {worst_grid['mean_test_score']:.4f}")
print(worst_grid["params"])

best_val = results.loc[results["Val accuracy"].idxmax()]
worst_val = results.loc[results["Val accuracy"].idxmin()]
largest_gap = results.loc[results["Train-test rozdiel"].idxmax()]
best_test = results.loc[results["Test accuracy"].idxmax()]
worst_test = results.loc[results["Test accuracy"].idxmin()]

for label, row in [("Najlepšia validácia", best_val),
                   ("Najhoršia validácia", worst_val),
                   ("Najlepší test (opis výsledkov)", best_test),
                   ("Najhorší test (opis výsledkov)", worst_test)]:
    print(f"{label}: {row['Model']}, {row['Rozdelenie']} | "
          f"val={row['Val accuracy']:.4f}, test={row['Test accuracy']:.4f}")

print(f"Najväčší rozdiel train-test: {largest_gap['Model']}, "
      f"{largest_gap['Rozdelenie']} | "
      f"{100 * largest_gap['Train-test rozdiel']:.2f} percentuálneho bodu")

# Vplyv pomeru porovnávame pri rovnakom type modelu.
for model_name in model_configs:
    rows = results.loc[results["Model"] == model_name]
    strongest = rows.loc[rows["Test accuracy"].idxmax()]
    weakest = rows.loc[rows["Test accuracy"].idxmin()]
    print(f"{model_name}: najvyššia test accuracy "
          f"{strongest['Test accuracy']:.4f} pri {strongest['Rozdelenie']}; "
          f"najnižšia {weakest['Test accuracy']:.4f} pri {weakest['Rozdelenie']}.")
```

Pri overenom behu bol podľa externej validácie najlepší Random Forest pri `60/10/30` s `max_depth=10`, `max_features=13` a `min_samples_leaf=1` (validácia 91,46 %, test 89,61 %, CV 90,07 %). Najnižšiu testovaciu accuracy mala logistická regresia pri `80/10/10` (87,14 %) a najväčší rozdiel train–test mal Random Forest pri `70/15/15`, približne 5,42 percentuálneho bodu, čo naznačuje možné pretrénovanie. Vybraný model rozpoznal 57,77 % skutočných nákupov, preto accuracy hodnotím spolu s recall a konfúznou maticou. Random Forest pri `70/10/20` dosiahol na teste 89,99 %, teda viac než pri `80/10/10` (88,44 %), takže viac tréningových dát automaticky neznamená lepší výsledok; menia sa aj riadky, parametre a neistota odhadu, preto jeden beh neurčuje všeobecne najlepší pomer.

## 11. Konfúzne matice najlepšieho modelu

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, dataset, title in [
    (axes[0], "train", "Tréningová množina"),
    (axes[1], "test", "Testovacia množina"),
]:
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        best_split[f"X_{dataset}"], best_split[f"y_{dataset}"],
        labels=[0, 1], display_labels=["Bez nákupu", "Nákup"],
        cmap="Blues", values_format="d", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predikovaná trieda")
    ax.set_ylabel("Skutočná trieda")

fig.suptitle(f"{best_key[1]} | rozdelenie {best_key[0]}")
plt.show()

print(classification_report(
    best_split["y_test"], best_model.predict(best_split["X_test"]),
    labels=[0, 1], target_names=["Bez nákupu", "Nákup"], zero_division=0,
))

```

Konfúzne matice zobrazujú správne predikcie na diagonále a chyby mimo nej; riadky sú skutočné triedy a stĺpce predikované triedy. Ľavá dolná bunka označuje skutočné nákupy, ktoré model prehliadol, a pravá horná návštevy nesprávne označené ako nákup. Tréningová a testovacia množina majú rôzne veľkosti, preto pri posudzovaní zovšeobecnenia porovnávam aj accuracy a recall, nie iba absolútne počty chýb.

## 12. Najlepší Random Forest a najdôležitejšie vlastnosti

```python
# Najlepší les vyberáme podľa externej validácie, nie podľa testu.
rf_row = results.loc[results["Model"] == "Random Forest"].sort_values(
    ["Val accuracy", "Grid accuracy"], ascending=False, kind="stable"
).iloc[0]
best_rf_key = (rf_row["Rozdelenie"], rf_row["Model"])
best_rf_model = trained_models[best_rf_key]
rf_split = experiment_splits[best_rf_key[0]]
gcv = grid_searches[best_rf_key]

print("Rozdelenie pre Random Forest:", best_rf_key[0])
print("gcv.best_params_:", gcv.best_params_)
print("gcv.best_score_:", gcv.best_score_)

# Dôležitosť zakódovaných vstupov podľa poklesu Gini nečistoty.
rf_preprocessing = best_rf_model.named_steps["preprocessing"]
rf_classifier = best_rf_model.named_steps["model"]
rf_feature_importance = pd.DataFrame({
    "Feature": rf_preprocessing.get_feature_names_out(),
    "Dôležitosť": rf_classifier.feature_importances_,
}).sort_values("Dôležitosť", ascending=False).reset_index(drop=True)

print("Najdôležitejšie zakódované features:")
display(rf_feature_importance.head(TOP_FEATURES).round(4))

# Premiešame pôvodný stĺpec pred preprocessingom, takže kategórie
# hodnotíme spoločne, nie ako samostatné One-Hot stĺpce.
rf_permutation = permutation_importance(
    best_rf_model, rf_split["X_val"], rf_split["y_val"],
    scoring="accuracy", n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE, n_jobs=N_JOBS,
)
rf_original_importance = pd.DataFrame({
    "Feature": rf_split["X_val"].columns,
    "Pokles accuracy (p. b.)": 100 * rf_permutation.importances_mean,
    "Std (p. b.)": 100 * rf_permutation.importances_std,
}).sort_values("Pokles accuracy (p. b.)", ascending=False).reset_index(drop=True)

print("Najdôležitejšie pôvodné features podľa validačnej accuracy:")
display(rf_original_importance.head(TOP_FEATURES).round(4))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
top_encoded = rf_feature_importance.head(TOP_FEATURES).iloc[::-1]
axes[0].barh(top_encoded["Feature"], top_encoded["Dôležitosť"], color="steelblue")
axes[0].set(title="Random Forest: feature_importances_",
            xlabel="Relatívny pokles Gini nečistoty")

top_original = rf_original_importance.head(TOP_FEATURES).iloc[::-1]
axes[1].barh(
    top_original["Feature"], top_original["Pokles accuracy (p. b.)"],
    xerr=top_original["Std (p. b.)"], color="seagreen", capsize=3,
)
axes[1].set(title="Permutation importance na validácii",
            xlabel="Pokles accuracy (percentuálne body)")
fig.suptitle(f"Random Forest | rozdelenie {best_rf_key[0]}")
plt.show()
```

Overený les pri `60/10/30`:

- `gcv.best_params_`: `{'model__max_depth': 10, 'model__max_features': 13, 'model__min_samples_leaf': 1}`
- `gcv.best_score_`: `0.9007050256123396` (CV accuracy 90,07 %).

| Zakódovaná feature | Relatívna dôležitosť podľa Gini |
| --- | ---: |
| `numeric__PageValues` | 0,491082 |
| `numeric__ProductRelated_Duration` | 0,074871 |
| `numeric__ExitRates` | 0,062695 |
| `numeric__ProductRelated` | 0,054459 |
| `numeric__BounceRates` | 0,044963 |

| Pôvodná feature | Pokles validačnej accuracy (p. b.) | Std (p. b.) |
| --- | ---: | ---: |
| `PageValues` | 14,79 | 0,79 |
| `Month` | 1,22 | 0,21 |
| `ExitRates` | 0,72 | 0,44 |
| `ProductRelated_Duration` | 0,28 | 0,22 |
| `VisitorType` | 0,27 | 0,16 |

`gcv.best_params_` vypíše najlepšiu konfiguráciu lesa a `gcv.best_score_` jej priemernú CV accuracy, prípadne validačnú accuracy pri vypnutej CV. Podľa článku používam `feature_importances_`, ktoré meria relatívny pokles nečistoty uzlov, a názvy získavam až z natrénovaného preprocessingu, aby zodpovedali One-Hot vstupom. Pokles nečistoty môže zvýhodniť vlastnosti s veľkým počtom hodnôt, preto dopĺňam permutation importance na validácii, kde väčší pokles accuracy po premiešaní pôvodného stĺpca znamená väčšiu závislosť modelu od tohto vstupu. Dôležitosti opisujú konkrétny model, nepreukazujú príčinnosť a pri vzájomne súvisiacich vlastnostiach sa môžu rozdeľovať medzi viac stĺpcov; postup vychádza z [článku o dôležitosti vlastností](https://mlcourse.ai/book/topic05/topic5_part3_feature_importance.html) a [dokumentácie permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## 13. Najlepšia logistická regresia a jej koeficienty

```python
logit_row = results.loc[results["Model"] == "Logistická regresia"].sort_values(
    ["Val accuracy", "Grid accuracy"], ascending=False, kind="stable"
).iloc[0]
best_logit_key = (logit_row["Rozdelenie"], logit_row["Model"])
best_logit_model = trained_models[best_logit_key]
grid_logit = grid_searches[best_logit_key]

print("Rozdelenie pre logistickú regresiu:", best_logit_key[0])
print("grid_logit.best_params_:", grid_logit.best_params_)
print("grid_logit.best_score_:", grid_logit.best_score_)

logit_classifier = best_logit_model.named_steps["model"]
logit_coefficients = pd.DataFrame({
    "Feature": best_logit_model.named_steps["preprocessing"].get_feature_names_out(),
    "Koeficient": logit_classifier.coef_[0],
})
logit_coefficients["Absolútna hodnota"] = logit_coefficients["Koeficient"].abs()
logit_coefficients = logit_coefficients.sort_values(
    "Absolútna hodnota", ascending=False
).reset_index(drop=True)
display(logit_coefficients.head(TOP_FEATURES).round(4))

top_coefficients = logit_coefficients.head(TOP_FEATURES).sort_values("Koeficient")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.barh(
    top_coefficients["Feature"], top_coefficients["Koeficient"],
    color=np.where(top_coefficients["Koeficient"] >= 0, "steelblue", "tomato"),
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(
    title=f"Logistická regresia | rozdelenie {best_logit_key[0]}",
    xlabel="Koeficient pre triedu nákup",
)
plt.show()
```

Ako v článku vypisujem najlepšie `C` a skóre grid-searchu a zobrazujem koeficienty s najväčšou absolútnou hodnotou. Kladný koeficient zvyšuje odhad šance nákupu pri ostatných vstupoch nezmenených a záporný ho znižuje, pričom veľkosť závisí aj od mierky a vzťahov medzi vlastnosťami. Poradie koeficientov preto čítam ako opis lineárneho modelu, nie ako rovnakú mieru dôležitosti, ktorú používa Random Forest; nadväzuje to na [článok o logistickej regresii](https://mlcourse.ai/book/topic04/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.html).
