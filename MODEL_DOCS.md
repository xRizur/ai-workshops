# Dokumentacja Modeli ML — pIC50 Prediction (ChEMBL)

## Spis treści
1. [Przegląd projektu](#1-przegląd-projektu)
2. [Pipeline danych](#2-pipeline-danych)
3. [MLP — Wielowarstwowy Perceptron](#3-mlp--wielowarstwowy-perceptron)
4. [GIN — Graph Isomorphism Network](#4-gin--graph-isomorphism-network)
5. [Kluczowe koncepty ML](#5-kluczowe-koncepty-ml)
6. [Strategia trenowania](#6-strategia-trenowania)
7. [Metryki oceny](#7-metryki-oceny)
8. [Jak uruchomić](#8-jak-uruchomić)

---

## 1. Przegląd projektu

**Zadanie:** Regresja — przewidywanie `pIC50` cząsteczki chemicznej na podstawie jej struktury.

**pIC50** to miara aktywności biologicznej leku:
```
pIC50 = -log10(IC50 [mol/L])
     = 9 - log10(IC50 [nM])
```
- IC50 = stężenie potrzebne do 50% zahamowania aktywności białka
- Wyższe pIC50 → silniejszy lek (np. pIC50 = 9 oznacza IC50 = 1 nM)
- Typowy zakres: 4–10

**Dane:** ChEMBL 36 — baza ~2.5M pomiarów IC50, filtrowana do:
- `standard_relation = '='` (dokładne pomiary)
- `standard_type = 'IC50'`
- organizmy: Homo sapiens / Mus musculus / Rattus norvegicus
- jeden wybrany cel białkowy (protein target) — max 10k próbek

---

## 2. Pipeline danych

```
ChEMBL SQLite
     ↓
Bronze Parquets       (surowe tabele: activities, assays, targets, molecules)
     ↓ filtracja IC50, standaryzacja jednostek, obliczenie pIC50
Silver Parquets       (oczyszczone: activities_silver, assays_silver, targets_silver, molecules_silver)
     ↓ JOIN + RDKit descriptors + one-hot organism + outlier removal + StandardScaler
Silver Final          (silver_final.parquet — gotowy do trenowania, ale bez tid)
```

**Dlaczego silver a nie final w modelach?**  
Silver final nie zawiera `tid` (identyfikator białka), więc nie możemy filtrować po białku.
Modele ładują silver parquets i dołączają bronze targets żeby uzyskać `pref_name` białka.

### Cechy wejściowe (MLP)

| Cecha | Opis | Typowy zakres |
|-------|------|---------------|
| MW | Masa cząsteczkowa | 100–800 Da |
| LogP | Lipofilność | -3 do 7 |
| HBD | Liczba donorów wiązań H | 0–5 |
| HBA | Liczba akceptorów wiązań H | 0–10 |
| TPSA | Polarna powierzchnia cząsteczki | 0–200 Å² |
| QED | Quantitative Estimate of Druglikeness | 0–1 |
| RotBonds | Obrotowe wiązania | 0–12 |
| AromaticRings | Pierścienie aromatyczne | 0–5 |

**Reguła Lipińskiego (Ro5):** lek doustny powinien mieć MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.

---

## 3. MLP — Wielowarstwowy Perceptron

### Architektura

```
Wejście (8 cech)
       ↓
┌─────────────────────────────────┐
│ Linear(8 → 256)                 │
│ BatchNorm1d(256)                │  ← stabilizacja
│ LeakyReLU(0.01)                 │  ← zapobiega dying neurons
│ Dropout(0.3)                    │  ← regularyzacja
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ Linear(256 → 128)               │
│ BatchNorm1d(128)                │
│ LeakyReLU(0.01)                 │
│ Dropout(0.3)                    │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│ Linear(128 → 64)                │
│ BatchNorm1d(64)                 │
│ LeakyReLU(0.01)                 │
│ Dropout(0.15)                   │
└─────────────────────────────────┘
       ↓
  Linear(64 → 1)
       ↓
    pIC50
```

### Inicjalizacja wag — He (Kaiming)

Zamiast losowej inicjalizacji, używamy inicjalizacji **Kaiminga** (He):

```python
nn.init.kaiming_normal_(weight, a=0.01, nonlinearity="leaky_relu")
```

**Dlaczego?**  
Przy głębokiej sieci z ReLU/LeakyReLU, losowa inicjalizacja powoduje, że wariancja aktywacji
maleje lub rośnie wykładniczo z liczbą warstw → gradienty znikają lub eksplodują.

He initialization ustawia wagi tak, że wariancja jest zachowana między warstwami:
```
Var(w) = 2 / fan_in
```

---

## 4. GIN — Graph Isomorphism Network

### Dlaczego grafy dla cząsteczek?

Cząsteczka to naturalnie **graf**:
- **Węzły** = atomy (z cechami: symbol, stopień, ładunek, hybrydyzacja, aromatyczność)
- **Krawędzie** = wiązania chemiczne (z cechami: typ wiązania, koniugacja, pierścień)

Deskryptory RDKit (MW, LogP, ...) tracą informację strukturalną — GNN uczy się jej bezpośrednio.

### Konwersja SMILES → Graf

```
"c1ccccc1" (benzen)          Graf: 6 węzłów (C), 6 krawędzi (wiązania aromatyczne)
                              każdy węzeł: 42 cechy
```

**Cechy węzła (42 dim):**
```
[symbol_one_hot (16)] + [stopień_one_hot (11)] + [ładunek_one_hot (7)] + 
[hybrydyzacja_one_hot (7)] + [aromatyczność (1)]
```

**Cechy krawędzi (6 dim):**
```
[typ_wiązania_one_hot (4)] + [czy_sprzężone (1)] + [czy_w_pierścieniu (1)]
```

### Architektura GIN

GIN opiera się na **teście Weisfeiler-Leman** — algorytmie sprawdzania izomorfizmu grafów.

**Aktualizacja węzła w GIN:**
```
h_v^(k) = MLP^(k)( (1 + ε) · h_v^(k-1) + Σ_{u ∈ N(v)} h_u^(k-1) )
```

Gdzie:
- `h_v^(k)` — reprezentacja węzła `v` po k-tej warstwie
- `N(v)` — sąsiedzi węzła `v`
- `ε` — uczony parametr (learnable epsilon)
- MLP — dwuwarstwowa sieć z BatchNorm i ReLU

```
Wejście: graf (X, edge_index)
         ↓
  Input Projection: Linear(42 → 128) + BatchNorm + ReLU
         ↓
  ┌─ GIN Layer 1: GINConv(MLP(128→256→128)) ─┐
  │  GIN Layer 2: GINConv(MLP(128→256→128))  │  × 3 warstwy
  └─ GIN Layer 3: GINConv(MLP(128→256→128)) ─┘
         ↓
  Global Mean Pool  ─┐
  Global Sum Pool   ─┴→ concat → [256 dim]
         ↓
  Readout MLP: Linear(256→128→64→1)
         ↓
       pIC50
```

**Dlaczego Mean + Sum pooling?**  
- Mean pool: unormowana reprezentacja → skala niezależna od rozmiaru grafu
- Sum pool: zachowuje informację o rozmiarze cząsteczki
- Konkatenacja daje bogatszą reprezentację niż każde osobno

---

## 5. Kluczowe koncepty ML

### 5.1 Dying Neurons (Wygaszające Neurony)

**Problem:** ReLU `max(0, x)` dla `x < 0` zwraca dokładnie 0 — brak gradientu.
Jeśli wagi wejściowe do neuronu są bardzo ujemne, neuron "umiera" — nigdy się nie aktywuje
i przestaje się uczyć (gradient = 0 zawsze).

**Detektujemy:** Forward hook mierzący `% aktywacji ≤ 0` w każdej warstwie.

```python
# Alarm gdy dead > 50%: "layer.net.2: 73.4%"  ← problem!
```

**Rozwiązania zastosowane:**
- **LeakyReLU(0.01):** zamiast `max(0,x)` → `x if x>0 else 0.01*x` — neuron zawsze ma gradient
- **BatchNorm przed aktywacją:** normalizuje wejście, zmniejsza szansę ujemnych wartości
- **He initialization:** właściwy rozkład wag na starcie

### 5.2 BatchNorm (Batch Normalization)

Normalizuje aktywacje w obrębie batcha:
```
z = (x - μ_batch) / (σ_batch + ε)
y = γ·z + β    ← γ, β to uczone parametry
```

**Korzyści:**
- Stabilizuje trening (mniejsza czułość na lr)
- Redukuje internal covariate shift
- Działa jako lekka regularyzacja
- Pozwala na wyższe learning rate

**Uwaga:** Zachowuj się inaczej w `model.train()` vs `model.eval()` —
w eval używa running mean/std z treningu, nie z batcha.

### 5.3 Dropout

Podczas trenowania losowo zeruje `p%` aktywacji:
```python
Dropout(p=0.3)  →  każdy neuron z p=30% szansą = 0
```

**Dlaczego to działa?** Model nie może polegać na żadnym konkretnym neuronie — uczy się
redundantnych reprezentacji → lepsza generalizacja.

**Ważne:** Wyłączone podczas `model.eval()` (inaczej predykcje byłyby losowe).

### 5.4 Monitorowanie Gradientów

```python
grad_norm = √(Σ ||∂L/∂w_i||²)   # L2 norm wszystkich gradientów
Δgrad     = |grad_norm_epoch_t - grad_norm_epoch_{t-1}|
```

**Interpretacja:**
| Sytuacja | Co oznacza |
|----------|------------|
| `grad_norm` → 0 | Vanishing gradients — model przestaje się uczyć |
| `grad_norm` bardzo duże | Exploding gradients — trening niestabilny |
| `Δgrad` → 0 | Konwergencja — gradient stabilizuje się |
| `Δgrad` oscyluje | Za duże lr lub problemy ze skalą danych |

**Gradient clipping** (`max_norm=5.0`): jeśli gradient jest za duży, skalujemy go w dół.
Zapobiega eksplodującym gradientom bez zatrzymywania treningu.

### 5.5 Learning Rate Scheduler

```python
ReduceLROnPlateau(patience=5, factor=0.5)
```

Jeśli `val_loss` nie spada przez 5 epok → zmniejsz `lr` o 50%.  
Pozwala na agresywny start (duże lr) i fine-tuning na końcu.

### 5.6 Early Stopping

```python
patience = 15  # epok bez poprawy val_loss
```

Zatrzymuje trening gdy val_loss przestaje spadać.  
Zapisujemy `best_state` = wagi z najlepszej epoki → brak overfittingu na końcu treningu.

---

## 6. Strategia trenowania

### Krok 1: Tiny dataset (N=50–200) — sprawdzenie że model "działa"

**Cel:** Overfit — model powinien osiągnąć R² ≈ 0.99 na danych treningowych.  
Jeśli nie — problem z architekturą lub danymi.

```
N=50, 200 epok → oczekiwane: train_loss → ~0.0, val_loss rośnie
                              train_R² → ~0.99
```

**Co sprawdzamy:**
- Czy loss maleje? (model w ogóle się uczy)
- Czy train << val? (overfit = model "zapamiętuje")
- Czy gradienty nie znikają?

### Krok 2: Pełny dataset jednego białka (N≤10k) — generalizacja

```
N=10k, 100 epok, early stopping
→ oczekiwane: train_R² ≈ val_R² ≈ 0.3–0.6 (realnie dla 8 deskryptorów)
```

**Dlaczego R² może być niskie (0.3–0.6)?**  
8 deskryptorów RDKit to uproszczenie — struktura 3D, kieszeń wiązania białka,
konformacje — tego nie mamy. GNN powinien dać lepsze wyniki bo uczy się struktury.

---

## 7. Metryki oceny

### R² (Coefficient of Determination)

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y_pred - y_true)² / Σ(y_true - ȳ)²
```

| R² | Interpretacja |
|----|---------------|
| 1.0 | Idealne przewidywania |
| 0.6 | Model wyjaśnia 60% wariancji — dobry dla pIC50 z deskryptorami |
| 0.3 | Słaby ale informatywny |
| ≤ 0 | Model gorszy niż przewidywanie średniej |

### RMSE (Root Mean Squared Error)

```
RMSE = √(Σ(y_pred - y_true)² / N)
```

Dla pIC50: RMSE < 1.0 = dobry, < 0.5 = bardzo dobry (w jednostkach pIC50).

### MSE vs RMSE w trenowaniu

MSE używamy jako **loss** (różniczkowalna, numerycznie stabilna).  
RMSE raportujemy (w tych samych jednostkach co pIC50 → łatwiejsza interpretacja).

---

## 8. Jak uruchomić

### Lokalnie (demo z syntetycznymi danymi)

```bash
python train_models.py --mode demo
```

### Na Colab z prawdziwymi danymi

```bash
# Dane są lokalnie (chembl_work/) lub na GCS
python train_models.py --mode real --silver_dir chembl_work/parquet_silver
```

### Notebooki na Colab

1. Wgraj `mlp_model.ipynb` na Colab
2. W cell "Config" ustaw ścieżki (dane są na GCS — bucket `project-0c281f60-a9e8-4b55-949-chembl-data`)
3. Uruchom wszystkie komórki

### Wymagane biblioteki

```bash
pip install torch scikit-learn rdkit matplotlib pandas pyarrow
# Dla GNN:
pip install torch-geometric
```

---

## Słowniczek

| Term | PL | Opis |
|------|-----|------|
| Epoch | Epoka | Jedno przejście przez cały dataset |
| Batch | Paczka | Podzbiór danych przetwarzany naraz |
| Loss | Strata | Funkcja błędu minimalizowana przez model |
| Gradient | Gradient | Pochodna loss względem wag — kierunek poprawy |
| Overfit | Przeuczenie | Model "pamięta" dane treningowe, nie generalizuje |
| Regularization | Regularyzacja | Techniki zapobiegające overfittingowi (dropout, weight decay) |
| Forward pass | Przejście w przód | Obliczenie predykcji |
| Backward pass | Przejście wstecz | Obliczenie gradientów (backpropagation) |
| Pooling | Agregacja | Łączenie reprezentacji węzłów grafu w jedną |
| SMILES | SMILES | Tekstowa reprezentacja struktury cząsteczki |
