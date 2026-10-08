# Neurónová sieť – rozdelenie 70/15/15

Text prenes do Markdown buniek a každý blok `python` do samostatnej kódovej bunky v main.ipynb, v uvedenom poradí. Používam výhradne rozdelenie **70/15/15**. Jeden tréningový blok spustí všetkých sedem behov; ostatné kódové bloky pripravujú dáta, definujú funkcie alebo vyhodnocujú už natrénované modely.

Kód nadväzuje na vyčistené `data_clean`, pripravený `preprocessor` a rozdelenie v `experiment_splits["70/15/15"]` z main.ipynb. To obsahuje `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` vytvorené existujúcim postupom. Tieto pôvodné premenné neprepisujem. Pred vložením tejto časti spusti bunky notebooku od čistenia dát po definíciu preprocessor a vytvorenie experiment_splits.

## Príprava dát

Predikujem Revenue: 0 znamená návštevu bez nákupu a 1 návštevu s nákupom. Tréning slúži na učenie váh aj predspracovania, validácia na sledovanie kriviek, EarlyStopping a výber konfigurácie. Testovacie dáta používam na záverečné úspešnosti a konfúzne matice.

Klonujem preprocessor a učím ho iba na tréningu. Doplnenie chýbajúcich hodnôt, škálovanie aj kategórie one-hot encoder preto nevychádzajú z validácie ani testu. Škálovanie je dôležité, aby numerické vstupy s veľkou mierkou neprevážili ostatné príznaky. Výstup transformácie prevediem na float32 tensory pre PyTorch.

```python
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
)
from IPython.display import display, Markdown

NN_SEED = int(RANDOM_STATE)
NN_DEVICE = torch.device("cpu")  # reprodukovateľné porovnanie na CPU
torch.set_num_threads(min(4, torch.get_num_threads()))

def nn_set_seed():
    random.seed(NN_SEED)
    np.random.seed(NN_SEED)
    torch.manual_seed(NN_SEED)

# Použijeme výhradne pripravené rozdelenie 70/15/15 z main.ipynb.
NN_SPLIT = "70/15/15"
nn_data = experiment_splits[NN_SPLIT]

# Vlastná kópia: existujúci preprocessor ani modely sa neprepíšu.
nn_preprocessor = clone(preprocessor)
nn_arrays = {
    "train": nn_preprocessor.fit_transform(nn_data["X_train"]),
    "val": nn_preprocessor.transform(nn_data["X_val"]),
    "test": nn_preprocessor.transform(nn_data["X_test"]),
}
nn_labels = {
    "train": np.asarray(nn_data["y_train"], dtype=np.int64).reshape(-1),
    "val": np.asarray(nn_data["y_val"], dtype=np.int64).reshape(-1),
    "test": np.asarray(nn_data["y_test"], dtype=np.int64).reshape(-1),
}
nn_tensors = {}
for split, values in nn_arrays.items():
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float32)
    labels = nn_labels[split]
    if not np.isfinite(values).all():
        raise ValueError(f"{split}: vstupy obsahujú NaN alebo nekonečné hodnoty.")
    if len(values) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split}: neplatné cieľové hodnoty alebo počet riadkov.")
    nn_tensors[split] = (
        torch.from_numpy(values),
        torch.tensor(labels, dtype=torch.float32),
    )

assert set(np.unique(nn_labels["train"])) == {0, 1}
NN_INPUTS = nn_tensors["train"][0].shape[1]
nn_majority = int(np.bincount(nn_labels["train"]).argmax())
print("Rozdelenie:", NN_SPLIT)
print("Počet vstupov po predspracovaní:", NN_INPUTS)
print("Počty vzoriek:", {s: len(y) for s, y in nn_labels.items()})
print("Podiel nákupov v tréningu:", f"{nn_labels['train'].mean():.2%}")
```

## Sieť, EarlyStopping a vyhodnotenie

Používam plne prepojenú sieť s ReLU v skrytých vrstvách. Výstupný neurón vracia logit; BCEWithLogitsLoss spája sigmoid a binárnu krížovú entropiu numericky stabilne. Pri predikcii použijem sigmoid a pevný prah 0,5. Adam aktualizuje váhy po minibatchoch. Všetky behy používajú rovnaký seed, batch size 128 a maximálne 300 epoch. Dropout ani penalizáciu váh nepoužívam.

EarlyStopping sleduje validačnú stratu. Ak počas 20 epoch nepríde zlepšenie väčšie než 0,0001 oproti poslednému významnému minimu, zastaví tréning. Obnoví váhy z absolútne najnižšej validačnej straty. Pri modeli bez EarlyStopping hodnotím posledné váhy, aby sa pretrénovanie neskrylo obnovením skoršej epochy.

Krivky loss a accuracy meriam na tréningu a validácii po každej epoche. Test nevyhodnocujem počas tréningu. Vo výsledkoch uvádzam accuracy, balanced accuracy a F1 nákupov pre tréning aj test. Balanced accuracy dáva obom triedam rovnakú váhu; F1 opisuje schopnosť predikovať nákupy. Väčšinový baseline predikuje vždy najčastejšiu triedu z tréningu.

```python
class ShopperMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers = []
        previous = NN_INPUTS
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.reference_loss = float("inf")
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.wait = 0

    def step(self, val_loss, model, epoch):
        # Uložíme absolútne najlepšie váhy aj pri malej zmene straty.
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
        # Patience resetuje až zlepšenie väčšie než min_delta.
        if val_loss < self.reference_loss - self.min_delta:
            self.reference_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


@torch.no_grad()
def nn_measure(model, split):
    model.eval()
    x, y = nn_tensors[split]
    logits = model(x.to(NN_DEVICE))
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y.to(NN_DEVICE)
    ).item()
    predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
    return loss, accuracy_score(nn_labels[split], predictions), predictions


def nn_train(name, hidden, lr, max_epochs=300, batch_size=128,
             early_stopping=True, patience=20, min_delta=1e-4):
    nn_set_seed()
    model = ShopperMLP(hidden).to(NN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(NN_SEED)
    loader = DataLoader(
        TensorDataset(*nn_tensors["train"]),
        batch_size=batch_size, shuffle=True, generator=generator,
    )
    stopper = EarlyStopping(patience, min_delta)
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch.to(NN_DEVICE))
            loss = criterion(logits, y_batch.to(NN_DEVICE))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: neplatná strata v epoche {epoch}.")
            loss.backward()
            optimizer.step()

        # Obe krivky meriame s aktuálnymi váhami na konci epochy.
        train_loss, train_acc, _ = nn_measure(model, "train")
        val_loss, val_acc, _ = nn_measure(model, "val")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_acc, "val_accuracy": val_acc,
        })
        should_stop = stopper.step(val_loss, model, epoch)
        if epoch == 1 or epoch % 50 == 0:
            print(f"{name} | epocha {epoch}: "
                  f"train loss={train_loss:.4f}, val loss={val_loss:.4f}")
        if early_stopping and should_stop:
            break

    if early_stopping:
        model.load_state_dict(stopper.best_state)
    selected_epoch = stopper.best_epoch if early_stopping else epoch
    print(f"{name}: vykonaných {epoch} epoch, hodnotené váhy z epochy "
          f"{selected_epoch}.")
    return {
        "name": name, "model": model, "history": pd.DataFrame(history),
        "hidden": tuple(hidden), "lr": lr, "batch_size": batch_size,
        "max_epochs": max_epochs, "early_stopping": early_stopping,
        "patience": patience, "min_delta": min_delta,
        "epochs": epoch, "selected_epoch": selected_epoch,
    }


def nn_evaluate(run):
    rows = []
    for split in ["train", "val", "test"]:
        loss, accuracy, predictions = nn_measure(run["model"], split)
        labels = nn_labels[split]
        rows.append({
            "Množina": split, "Loss": loss,
            "Accuracy (%)": 100 * accuracy,
            "Balanced accuracy (%)": 100 * balanced_accuracy_score(labels, predictions),
            "F1 nákup": f1_score(labels, predictions, zero_division=0),
            "Baseline accuracy (%)": 100 * accuracy_score(
                labels, np.full(len(labels), nn_majority)
            ),
        })
    return pd.DataFrame(rows).set_index("Množina")


def nn_plot(run):
    history = run["history"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for split, label in [("train", "Tréning"), ("val", "Validácia")]:
        axes[0].plot(history["epoch"], history[f"{split}_loss"], label=label)
        axes[1].plot(history["epoch"], 100 * history[f"{split}_accuracy"], label=label)
    for ax in axes:
        ax.axvline(run["selected_epoch"], color="gray", linestyle="--",
                   label="Epocha hodnotených váh")
        ax.set_xlabel("Epocha")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set(title="Binárna krížová entropia", ylabel="Loss")
    axes[1].set(title="Úspešnosť", ylabel="Accuracy (%)")
    fig.suptitle(run["name"])
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, split, label in zip(axes, ["train", "test"], ["Tréning", "Test"]):
        _, _, predictions = nn_measure(run["model"], split)
        ConfusionMatrixDisplay.from_predictions(
            nn_labels[split], predictions, labels=[0, 1],
            display_labels=["Bez nákupu", "Nákup"], cmap="Blues",
            colorbar=False, values_format="d", ax=ax,
        )
        ax.set(title=label, xlabel="Predikovaná trieda", ylabel="Skutočná trieda")
    fig.suptitle(run["name"])
    plt.show()


def nn_report(run):
    display(nn_evaluate(run).loc[["train", "test"]].round(4))
    nn_plot(run)
    for split in ["train", "test"]:
        _, _, predictions = nn_measure(run["model"], split)
        print(f"{run['name']} – {split}")
        print(classification_report(
            nn_labels[split], predictions, labels=[0, 1],
            target_names=["Bez nákupu", "Nákup"], zero_division=0,
        ))
```

## Všetkých sedem tréningov v jednom bloku

O demonštruje pretrénovanie veľkej siete s vrstvami 256, 128 a 64 neurónov pri learning rate 0,001 počas 300 epoch. ES opakuje rovnakú konfiguráciu aj počiatočné nastavenie, ale pridáva EarlyStopping. Každý beh začína od nových váh, nejde o pokračovanie predchádzajúceho tréningu.

E1–E3 menia learning rate pri rovnakých vrstvách 64 a 32 neurónov. E4 a E5 menia architektúru pri learning rate 0,001. Tým porovnávam päť konfigurácií a aspoň dva druhy nastavení. Všetkých sedem behov používa tie isté tréningové, validačné aj testovacie vzorky; nepoužívam iné rozdelenia dát.

```python
# Jediné miesto, kde sa spúšťa tréning všetkých siedmich konfigurácií.
# Spoločné: Adam, ReLU, batch size 128, max. 300 epoch, rovnaký seed.
nn_configs = [
    {"id": "O", "name": "Veľká sieť bez EarlyStopping",
     "hidden": (256, 128, 64), "lr": 0.001, "early_stopping": False},
    {"id": "ES", "name": "Veľká sieť s EarlyStopping",
     "hidden": (256, 128, 64), "lr": 0.001, "early_stopping": True},
    {"id": "E1", "name": "E1", "hidden": (64, 32),
     "lr": 0.0001, "early_stopping": True},
    {"id": "E2", "name": "E2", "hidden": (64, 32),
     "lr": 0.001, "early_stopping": True},
    {"id": "E3", "name": "E3", "hidden": (64, 32),
     "lr": 0.01, "early_stopping": True},
    {"id": "E4", "name": "E4", "hidden": (32,),
     "lr": 0.001, "early_stopping": True},
    {"id": "E5", "name": "E5", "hidden": (128, 64, 32),
     "lr": 0.001, "early_stopping": True},
]

nn_all_runs = {}
for config in nn_configs:
    settings = {key: value for key, value in config.items() if key != "id"}
    nn_all_runs[config["id"]] = nn_train(
        **settings, max_epochs=300, batch_size=128,
        patience=20, min_delta=1e-4,
    )

nn_overfit = nn_all_runs["O"]
nn_early = nn_all_runs["ES"]
nn_runs = {key: run for key, run in nn_all_runs.items() if key.startswith("E") and key != "ES"}
print("Hotovo:", len(nn_all_runs), "tréningov na rozdelení", NN_SPLIT)
```

## Spoločná tabuľka výsledkov a výber experimentov

Jedna tabuľka obsahuje konfigurácie a metriky všetkých siedmich tréningov vrátane E1–E5. Najlepší a najhorší experiment vyberám spomedzi E1–E5 podľa validačnej accuracy; pri zhode rozhoduje nižšia validačná strata. EarlyStopping vyberá epochu podľa loss, takže kritérium výberu epochy a kritérium porovnávania konfigurácií sa líšia.

Rozdiel train-test je v percentuálnych bodoch. Je doplnkovým znakom pretrénovania; samotný malý rozdiel nestačí, ak je výkon na oboch množinách nízky. Výber modelu nezávisí od testovej accuracy.

```python
# Najlepší/najhorší experiment vyberieme iba spomedzi E1–E5.
# Rozhodnutie urobíme podľa validácie pred vyhodnotením testu.
nn_validation = {}
for name, run in nn_runs.items():
    val_loss, val_accuracy, _ = nn_measure(run["model"], "val")
    nn_validation[name] = (val_accuracy, -val_loss)
nn_best_name = max(nn_validation, key=nn_validation.get)
nn_worst_name = min(nn_validation, key=nn_validation.get)
nn_best_model = nn_runs[nn_best_name]["model"]

nn_rows = []
for identifier, run in nn_all_runs.items():
    metrics = nn_evaluate(run)
    run["metrics"] = metrics
    nn_rows.append({
        "Tréning": identifier,
        "Skryté vrstvy": " → ".join(map(str, run["hidden"])),
        "Learning rate": run["lr"],
        "EarlyStopping": run["early_stopping"],
        "Vykonané epochy": run["epochs"],
        "Hodnotená epocha": run["selected_epoch"],
        "Train accuracy (%)": metrics.loc["train", "Accuracy (%)"],
        "Val accuracy (%)": metrics.loc["val", "Accuracy (%)"],
        "Val loss": metrics.loc["val", "Loss"],
        "Test accuracy (%)": metrics.loc["test", "Accuracy (%)"],
        "Train balanced accuracy (%)": metrics.loc["train", "Balanced accuracy (%)"],
        "Test balanced accuracy (%)": metrics.loc["test", "Balanced accuracy (%)"],
        "Train F1 nákup": metrics.loc["train", "F1 nákup"],
        "Test F1 nákup": metrics.loc["test", "F1 nákup"],
        "Baseline train accuracy (%)": metrics.loc["train", "Baseline accuracy (%)"],
        "Baseline test accuracy (%)": metrics.loc["test", "Baseline accuracy (%)"],
        "Train-test rozdiel (p. b.)": (
            metrics.loc["train", "Accuracy (%)"] - metrics.loc["test", "Accuracy (%)"]
        ),
    })
nn_results = pd.DataFrame(nn_rows).set_index("Tréning")
display(nn_results.round(4))
print("Najlepší experiment podľa validácie:", nn_best_name)
print("Najhorší experiment podľa validácie:", nn_worst_name)

nn_history = nn_overfit["history"]
nn_minimum = nn_history.loc[nn_history["val_loss"].idxmin()]
nn_last = nn_history.iloc[-1]
print(f"Bez EarlyStopping: minimum val loss {nn_minimum['val_loss']:.4f} "
      f"v epoche {int(nn_minimum['epoch'])}, na konci {nn_last['val_loss']:.4f}.")
print(f"Train loss v rovnakých epochách: {nn_minimum['train_loss']:.4f} "
      f"→ {nn_last['train_loss']:.4f}.")
if not (nn_last["val_loss"] > nn_minimum["val_loss"] + 0.01
        and nn_last["train_loss"] < nn_minimum["train_loss"]):
    print("Tento beh jednoznačne nepotvrdil pretrénovanie; skontroluj krivky.")

nn_before = nn_overfit["metrics"]
nn_after = nn_early["metrics"]
print(f"EarlyStopping obnovil epochu {nn_early['selected_epoch']} "
      f"po {nn_early['epochs']} vykonaných epochách.")
print("Rozdiel train-test pred/po EarlyStopping (p. b.):",
      round(nn_results.loc["O", "Train-test rozdiel (p. b.)"], 2),
      round(nn_results.loc["ES", "Train-test rozdiel (p. b.)"], 2))
```

## Grafy a konfúzne matice

Zobrazujem priebeh siete bez EarlyStopping, siete s EarlyStopping a najlepšieho aj najhoršieho experimentu E1–E5. Každý má graf tréningovej a validačnej straty aj accuracy, konfúznu maticu tréningu a testu a klasifikačný report pre obe množiny.

Prerušovaná čiara označuje epochu hodnotených váh. Pri EarlyStopping krivka zobrazuje aj neskoršie epochy počas čakania na zlepšenie, ale metriky a konfúzne matice sa počítajú z obnovených váh. Na diagonále matice sú správne predikcie: vľavo hore návštevy bez nákupu a vpravo dole nákupy. Vpravo hore sú falošne predikované nákupy, vľavo dole nezachytené nákupy. Kontrolujem obe triedy, nie iba väčšinovú diagonálu.

```python
# Vyhodnotenie zámerného pretrénovania a rovnakej siete s EarlyStopping.
# Navyše najlepší a najhorší experiment z E1–E5 podľa validácie.
for identifier in dict.fromkeys(["O", "ES", nn_best_name, nn_worst_name]):
    nn_report(nn_all_runs[identifier])
```

## Namerané výsledky overenia

Nasledujúce hodnoty pochádzajú z overovacieho behu na tomto rozdelení. Pri vlastnom spustení ich aktualizuj podľa `nn_results` a grafov. Dataset obsahoval 8 089 tréningových, 1 734 validačných a 1 734 testovacích vzoriek; po predspracovaní mal 72 vstupov. Nákup tvoril 16,52 % tréningu. Test accuracy väčšinového baseline bola 83,51 %, preto samotné prekročenie 50 % accuracy nie je dostatočné porovnanie.

V stĺpci „Hodnotená epocha“ je epocha váh použitých na výsledky; „Vykonané epochy“ je skutočná dĺžka tréningu. Spoločné nastavenia sú opísané vyššie.

| Beh | Skryté vrstvy | LR | EarlyStopping | Train acc. (%) | Val acc. (%) | Test acc. (%) | Train balanced acc. (%) | Test balanced acc. (%) | Train F1 nákup | Test F1 nákup | Hodnotená epocha | Vykonané epochy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O | 256 → 128 → 64 | 0.001 | nie | 99.98 | 87.66 | 86.85 | 99.93 | 73.89 | 0.9993 | 0.5778 | 300 | 300 |
| ES | 256 → 128 → 64 | 0.001 | áno | 90.52 | 88.93 | 88.87 | 83.18 | 79.31 | 0.7156 | 0.6584 | 6 | 26 |
| E1 | 64 → 32 | 0.0001 | áno | 90.96 | 89.50 | 89.04 | 81.38 | 77.31 | 0.7103 | 0.6429 | 129 | 149 |
| E2 | 64 → 32 | 0.001 | áno | 90.73 | 89.56 | 89.04 | 82.11 | 77.45 | 0.7115 | 0.6442 | 14 | 34 |
| E3 | 64 → 32 | 0.01 | áno | 90.95 | 89.33 | 88.93 | 79.90 | 75.13 | 0.6983 | 0.6190 | 5 | 25 |
| E4 | 32 | 0.001 | áno | 91.20 | 89.22 | 88.47 | 81.25 | 75.70 | 0.7136 | 0.6183 | 46 | 66 |
| E5 | 128 → 64 → 32 | 0.001 | áno | 90.41 | 88.70 | 88.70 | 84.44 | 80.32 | 0.7223 | 0.6644 | 8 | 28 |

### Pretrénovanie a jeho obmedzenie

Bez EarlyStopping dosiahla veľká sieť po 300 epochách train accuracy 99,98 %, ale test accuracy iba 86,85 %. Tréningová strata klesla z 0,2085 v 6. epoche na 0,00125 na konci, zatiaľ čo validačná strata sa z minima 0,2578 zvýšila na 1,1467. To jasne ukazuje pretrénovanie: tréning sa zlepšuje, validácia sa zhoršuje. Loss ukazuje problém výraznejšie než accuracy, pretože zohľadňuje aj istotu nesprávnych predikcií.

S EarlyStopping tréning skončil po 26 epochách a obnovil váhy zo 6. epochy. Train accuracy bola 90,52 % a test accuracy 88,87 %. Rozdiel train-test klesol z 13,12 na 1,65 percentuálneho bodu, test accuracy sa zvýšila o 2,02 percentuálneho bodu. Test balanced accuracy vzrástla zo 73,89 % na 79,31 % a F1 nákupov z 0,5778 na 0,6584.

Zhoršovanie validačnej krivky po 6. epoche je viditeľné aj pri EarlyStopping, pretože patience umožňuje ďalších 20 epoch. Hodnotený model však používa skoršie obnovené váhy. Graf preto neinterpretujem tak, že boli ponechané posledné váhy z pretrénovanej časti behu.

### Porovnanie E1–E5

Najlepší podľa validačnej accuracy bol E2: val accuracy 89,56 %, train accuracy 90,73 % a test accuracy 89,04 %. E1 dosiahol rovnakú test accuracy, no vykonal 149 epoch oproti 34 pri E2. E3 s vyšším learning rate potreboval 25 epoch, ale test accuracy bola mierne nižšia, 88,93 %. Menší learning rate teda vyžadoval dlhší tréning a vyšší learning rate automaticky nezlepšil výsledok.

Najhorší podľa validačnej accuracy bol E5: val aj test accuracy 88,70 %. Zároveň však dosiahol najvyššiu test balanced accuracy spomedzi E1–E5 (80,32 %) a najvyššie test F1 nákupov (0,6644). E2 mal balanced accuracy 77,45 % a F1 0,6442. Väčšia sieť tak nepriniesla najvyššiu accuracy, no lepšie zachytávala nákupy. Označenie „najhorší“ sa preto vzťahuje na zvolené validačné kritérium, nie na všetky metriky.

E4 mal najnižšiu test accuracy spomedzi E1–E5 (88,47 %) a najväčší rozdiel train-test (2,73 percentuálneho bodu). Nie je však najhorším podľa validácie; tým zostáva E5. Rozdiely test accuracy medzi piatimi experimentmi sú malé, približne od 88,47 % do 89,04 %. Všetky prekračujú väčšinový baseline, ale výsledky pochádzajú z jedného rozdelenia a seedu, preto nepreukazujú univerzálnu nadradenosť jednej architektúry.

Konfúzna matica E2 mala na tréningu 6 414 správnych návštev bez nákupu, 925 správne predikovaných nákupov, 339 falošne predikovaných nákupov a 411 nezachytených nákupov. Na teste bolo 1 372 správnych návštev bez nákupu, 172 správne predikovaných nákupov, 76 falošne predikovaných nákupov a 114 nezachytených nákupov. Zachytil teda približne 60,14 % zo skutočných nákupov. Accuracy blízka 90 % neznamená, že model zachytáva takmer všetky nákupy.

Za výsledný experiment pri zvolenom kritériu považujem E2. Pri praktickom cieli zachytiť viac nákupov by bolo potrebné vopred určiť zodpovedajúcu validačnú metriku. Testové výsledky slúžia na opis, nie na dodatočnú zmenu kritéria výberu.

## Reprodukovateľnosť a uloženie výstupov

Overenie bežalo na CPU so seedom 42. Prostredie: Python 3.14.6, PyTorch 2.14.1, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 a Matplotlib 3.11.2. Iné verzie, zariadenie alebo seed môžu mierne zmeniť hodnoty.

Po vložení buniek ulož notebook aj s tabuľkou, grafmi a train/test konfúznymi maticami. Pri novom behu aktualizuj namerané pozorovania podľa aktuálnych výstupov. Samotné použitie EarlyStopping nie je dôkazom odstránenia pretrénovania: výsledok treba posúdiť podľa kriviek, úspešností a oboch tried v konfúznej matici.
