# Documentation — Weather Prevision

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Pipeline complet](#3-pipeline-complet)
4. [Modules](#4-modules)
   - [data.py](#41-datapy)
   - [parser.py](#42-parserpy)
   - [check_data.py](#43-check_datapy)
   - [train.py](#44-trainpy)
   - [predict_gui.py](#45-predict_guipy)
   - [app_gui.py](#46-app_guipy)
   - [ann.py](#47-annpy-legacy)
   - [nn_tf.py](#48-nn_tfpy-legacy)
5. [Format des données](#5-format-des-données)
6. [Architecture du modèle](#6-architecture-du-modèle)
7. [Améliorations apportées](#7-améliorations-apportées)
8. [Guide de démarrage](#8-guide-de-démarrage)

---

## 1. Vue d'ensemble

**Objectif** : prédire la **température** (en °C) et la **présence de pluie** (0/1) heure par heure sur les 24 prochaines heures, à partir d'observations météo historiques de la station **Bordeaux-Mérignac** (code Infoclimat `07510`).

**Approche** : réseau de neurones récurrent (LSTM) entraîné sur des séries temporelles horaires. Le modèle reçoit 48 heures d'observations en entrée et produit 24 heures de prévisions en sortie.

**Stack technique** :

| Composant | Technologie |
|-----------|-------------|
| Réseau de neurones | TensorFlow / Keras |
| Manipulation de données | pandas, numpy |
| Interface graphique | tkinter, matplotlib |
| Source de données | API OpenData Infoclimat v2 |
| Langage | Python 3.13 |

---

## 2. Architecture du projet

```
weather-prevision/
├── codes/
│   ├── app_gui.py        # Interface centrale (point d'entrée utilisateur)
│   ├── data.py           # Collecte des données via l'API Infoclimat
│   ├── parser.py         # Nettoyage et segmentation des CSV
│   ├── check_data.py     # Audit de qualité des CSV
│   ├── train.py          # Entraînement du modèle LSTM
│   ├── predict_gui.py    # Visualisation des prédictions
│   ├── ann.py            # [legacy] Réseau de neurones custom numpy
│   └── nn_tf.py          # [legacy] Premier prototype TensorFlow
│
├── downloads/            # CSV bruts téléchargés depuis Infoclimat
├── artifacts/            # Modèle entraîné + paramètres de normalisation
│   ├── weather_model.keras
│   └── norm_params.pkl
│
└── DOCUMENTATION.md
```

---

## 3. Pipeline complet

```
┌─────────────────────────────────────────────────────────────────┐
│                         app_gui.py                              │
│  Orchestrateur graphique — enchaîne les 4 étapes ci-dessous     │
└──────────┬──────────────────┬──────────────┬───────────────────┘
           │                  │              │
           ▼                  ▼              ▼
     ┌──────────┐     ┌──────────────┐  ┌──────────────┐
     │ data.py  │────▶│  parser.py   │  │ check_data   │
     │          │     │              │  │ (optionnel)  │
     │ API      │     │ Nettoyage    │  └──────────────┘
     │ Infoclimat     │ 72h consec.  │
     └──────────┘     └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  train.py    │
                      │              │
                      │ LSTM         │─────▶ artifacts/
                      │ 48h → 24h   │       weather_model.keras
                      └──────────────┘       norm_params.pkl
                             │
                             ▼
                      ┌──────────────┐
                      │predict_gui.py│
                      │              │
                      │ Dashboard    │
                      │ comparaison  │
                      └──────────────┘
```

**Étape 1 — Collecte** (`data.py`) : appel à l'API Infoclimat sur une plage de dates, par blocs de 7 jours maximum, consolidation en un CSV unique.

**Étape 2 — Nettoyage** (`parser.py`) : suppression des jours incomplets (< 24 mesures), découpe en segments horaires consécutifs, conservation des segments ≥ 72 heures, insertion de séparateurs `*` entre segments.

**Étape 3 — Entraînement** (`train.py`) : normalisation des features, construction des séquences X/Y, entraînement du LSTM avec early stopping, sauvegarde du modèle et des paramètres de normalisation.

**Étape 4 — Prédiction** (`predict_gui.py`) : chargement du modèle, inférence sur le jeu de test, dashboard interactif avec métriques globales et navigation par échantillon.

---

## 4. Modules

### 4.1 `data.py`

**Rôle** : télécharger des observations météo depuis l'API OpenData Infoclimat v2 et les exporter en CSV.

#### Variables d'environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `INFOCLIMAT_TOKEN` | Oui | Clé API Infoclimat (liée à une IP publique) |

#### Fonctions principales

```python
build_url(station_id, start_date, end_date, fmt="csv") -> str
```
Construit l'URL d'appel à l'API. Paramètres : identifiant station, dates au format `YYYY-MM-DD`, format de sortie (`csv` ou `json`).

---

```python
fetch_csv_from_infoclimat(url) -> str
```
Effectue la requête HTTP GET. Gère les erreurs courantes :
- `wrong ip adress` → IP publique non autorisée pour ce token
- HTTP 400/403/429 → diagnostics détaillés dans le message d'erreur

---

```python
parse_infoclimat_csv(csv_text) -> pd.DataFrame
```
Parse le CSV brut (séparateur `;`, commentaires `#`, ligne d'unités `YYYY/MM/DD`). Détecte automatiquement les colonnes disponibles selon la station. Produit un DataFrame normalisé avec les colonnes :

| Colonne | Type | Description |
|---------|------|-------------|
| `station_id` | str | Identifiant de la station |
| `datetime_utc` | datetime | Horodatage UTC |
| `hour_utc` | int | Heure UTC (0–23) |
| `temp_C` | float | Température en °C |
| `pressure_hPa` | float | Pression en hPa |
| `humidity_pct` | float | Humidité relative en % |
| `wind_avg` | float | Vitesse moyenne du vent (km/h ou m/s selon station) |
| `wind_dir_deg` | float | Direction du vent en degrés (0–360) |
| `sunshine` | float | Ensoleillement ou rayonnement (unité variable selon station) |
| `rain_mm` | float | Précipitations en mm |

---

```python
generate_chunks(start_date, end_date, max_days=7) -> Iterator[(date, date)]
```
Découpe une plage de dates en blocs de `max_days` jours maximum pour respecter la limite de l'API Infoclimat.

---

```python
download_infoclimat_range(start_date, end_date, output_dir, filename, station_id, max_days_per_request) -> (str, pd.DataFrame)
```
Fonction principale réutilisable depuis une autre application. Télécharge, parse, concatène, déduplique et sauvegarde le CSV final. Retourne `(chemin_fichier, dataframe)`.

---

#### Interface graphique (`DateFileDialog`)

Boîte de dialogue Tkinter pour saisir date début/fin, nom de fichier et clé API optionnelle. La clé saisie ici écrase la variable d'environnement pour la session.

---

### 4.2 `parser.py`

**Rôle** : nettoyer un CSV brut pour ne conserver que des segments horaires continus d'au moins N heures.

**Problème traité** : les données réelles comportent des trous (heures manquantes, journées incomplètes). Le modèle LSTM suppose une série temporelle strictement régulière à 1h d'intervalle — les ruptures doivent être signalées explicitement.

#### Fonctions principales

```python
clean_csv_72h(csv_path, min_hours=72) -> (str, str)
```
Pipeline complet de nettoyage. Retourne `(chemin_fichier_sortie, rapport_texte)`.

Étapes internes :

1. **Détection automatique de la colonne datetime** (`detect_datetime_column`) — cherche en priorité `datetime_utc`, `dh_utc`, puis toute colonne dont le nom contient `date`, `time` ou `dh`.

2. **Suppression des doublons de timestamps** — un même horodatage ne peut apparaître qu'une seule fois.

3. **Filtrage des jours incomplets** (`keep_only_complete_days`) — un jour est valide si et seulement si les 24 heures 0h–23h sont toutes présentes (au moins une mesure par heure).

4. **Segmentation en blocs consécutifs** (`build_consecutive_segments`) — découpe le fichier à chaque rupture de continuité horaire (différence > 1h entre deux timestamps consécutifs).

5. **Filtrage par durée minimale** — seuls les segments ≥ `min_hours` heures sont conservés (défaut : 72h).

6. **Insertion de séparateurs** — une ligne contenant `*` dans la colonne datetime est insérée entre deux segments. Cette convention est utilisée par `train.py` et `predict_gui.py` pour délimiter les blocs lors du chargement.

**Format de sortie** : `{nom_fichier_original}_clean72h.csv`

---

### 4.3 `check_data.py`

**Rôle** : outil d'audit autonome — produit un rapport de qualité sur n'importe quel CSV horaire sans modifier le fichier.

```python
analyze_csv(csv_path) -> str
```
Retourne un rapport texte comprenant :
- Période couverte (date min → date max)
- Nombre de lignes totales / conservées / invalides
- Nombre de jours complets (24h sans doublons) / incomplets / avec doublons
- Détail ligne par ligne des anomalies (heures manquantes, heures dupliquées)

```python
show_report_window(report_text, title) -> None
```
Affiche le rapport dans une fenêtre `ScrolledText` Tkinter. Fermeture par `Echap` ou fermeture de fenêtre.

**Usage en ligne de commande** :
```bash
python check_data.py chemin/vers/fichier.csv
```
Sans argument, une boîte de dialogue de sélection de fichier s'ouvre.

---

### 4.4 `train.py`

**Rôle** : construire le pipeline de données, entraîner le modèle LSTM, sauvegarder le modèle et les paramètres de normalisation.

#### Constantes

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `lookback` | 48 | Nombre d'heures d'historique en entrée |
| `window_size` | 24 | Horizon de prédiction en heures |
| `n_inputs` | 12 | Nombre de features par timestep |
| `rain_mm_threshold` | 0.1 | Seuil pluie/sec en mm/h |

#### Feature layout (12 colonnes par timestep)

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0 | `hour_sin` | `sin(2π × heure / 24)` |
| 1 | `hour_cos` | `cos(2π × heure / 24)` |
| 2 | `day_sin` | `sin(2π × jour_année / 365)` |
| 3 | `day_cos` | `cos(2π × jour_année / 365)` |
| 4 | `humidity` | valeur / 100 → [0, 1] |
| 5 | `temp` | `(T − μ_T) / σ_T` |
| 6 | `pressure` | `(P − μ_P) / σ_P` |
| 7 | `rain_log` | `(log(1+r) − μ_r) / σ_r` |
| 8 | `press_tend_3h` | `press_norm[t] − press_norm[t−3]` |
| 9 | `wind_avg` | `(W − μ_W) / σ_W` |
| 10 | `wind_dir_sin` | `sin(2π × direction / 360)` |
| 11 | `wind_dir_cos` | `cos(2π × direction / 360)` |

> **Note encodage circulaire** : l'heure, le jour de l'année et la direction du vent sont des variables circulaires. Un encodage direct (valeur brute) créerait une discontinuité artificielle (ex: 23h → 0h semblerait un saut de 23 unités). Le couple sin/cos élimine cette discontinuité.

> **Note tendance de pression** : `press_tend_3h[t] = press_norm[t] − press_norm[t−3]`. Une valeur négative (pression en chute) est un signal fort de front pluvieux imminent.

#### Fonctions principales

```python
get_mean_std(filename, i_col) -> (float, float)
```
Calcule μ et σ d'une colonne CSV en ignorant les lignes séparateurs `*` et les valeurs vides. Pour la colonne 9 (pluie), applique `log(1+r)` avant le calcul.

---

```python
get_norm_data_from_file(filename, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W)
    -> (chunks, raw_rain_chunks)
```
Lit le CSV, segmente sur les `*`, normalise, calcule la tendance de pression et l'encodage de la direction du vent. Retourne :
- `chunks` : liste de tableaux `(N, 12)` — features normalisées
- `raw_rain_chunks` : liste de tableaux `(N,)` — pluie brute en mm (utilisée pour la cible binaire)

---

```python
build_xy_from_chunks(chunks, raw_rain_chunks, lookback=48, window_size=24, rain_mm_threshold=0.1)
    -> (X, Y_temp, Y_rain)
```
Construit les tenseurs d'entraînement :
- `X` : shape `(N_samples, 48, 12)` — fenêtres d'entrée
- `Y_temp` : shape `(N_samples, 24)` — températures normalisées cibles
- `Y_rain` : shape `(N_samples, 24)` — cible binaire pluie (0/1), seuil physique sur mm bruts

---

```python
make_weighted_bce(pos_weight) -> loss_fn
```
Retourne une fonction de perte Binary Cross-Entropy pondérée. Chaque pixel de pluie (`y=1`) est pénalisé `pos_weight` fois plus qu'un pixel sec. Compense le déséquilibre de classes sans modifier les données.

```
loss = mean( weights × BCE(y_true, y_pred) )
weights = y_true × pos_weight + (1 − y_true)
```

---

```python
class F1Score(tf.keras.metrics.Metric)
```
Métrique Keras custom calculant le F1-score à partir de `Precision` et `Recall` internes :
```
F1 = 2 × Précision × Rappel / (Précision + Rappel + ε)
```
Réinitialisation correcte à chaque epoch via `reset_state()`.

---

```python
build_model(lookback=48, n_inputs=12, window_size=24, rain_pos_weight=1.0) -> tf.keras.Model
```
Construit et compile le modèle. Voir [Section 6](#6-architecture-du-modèle) pour le détail de l'architecture.

---

#### Artefacts produits

| Fichier | Contenu |
|---------|---------|
| `artifacts/weather_model.keras` | Modèle complet (architecture + poids) |
| `artifacts/norm_params.pkl` | Tuple `(μ_T, σ_T, μ_P, σ_P, μ_r, σ_r, μ_W, σ_W)` |

> **Important** : les paramètres de normalisation sont calculés **uniquement sur le jeu d'entraînement** et réutilisés à l'identique sur le test pour éviter toute fuite d'information.

---

### 4.5 `predict_gui.py`

**Rôle** : charger le modèle entraîné, inférer sur le jeu de test et afficher un dashboard interactif de visualisation et d'évaluation.

#### Pipeline de données

Identique à `train.py` (`get_norm_data_and_dates_from_file` + `build_xy_and_sample_dates`), avec en plus la conservation des dates brutes pour l'affichage.

#### Métriques calculées

```python
compute_metrics(y_true, y_pred) -> dict
```
Calcule sur des tableaux 1D :

| Clé | Formule |
|-----|---------|
| `mae` | `mean(|y_pred − y_true|)` |
| `mape` | `mean(|y_pred − y_true| / max(ε, |y_true|)) × 100` |
| `rmse` | `sqrt(mean((y_pred − y_true)²))` |
| `r2` | `1 − SS_res / SS_tot` |

```python
compute_per_sample_mae(y_true_real, y_pred_real) -> np.ndarray
```
MAE par échantillon (shape `[N]`) — utilisée pour les boutons "Meilleur cas / Pire cas".

#### Interface `ForecastNavigator`

Fenêtre principale de navigation. Paramètres à la construction :

| Paramètre | Type | Description |
|-----------|------|-------------|
| `idx_init` | int | Index de départ |
| `sample_dates` | list[str] | Dates de chaque échantillon |
| `y_temp_pred_real` | ndarray (N,24) | Températures prédites en °C |
| `y_temp_true_real` | ndarray (N,24) | Températures réelles en °C |
| `y_rain_pred` | ndarray (N,24) | Probabilités de pluie [0,1] |
| `y_rain_true` | ndarray (N,24) | Pluie réelle binaire |
| `global_metrics_temp` | dict | MAE/RMSE/R² global température |
| `global_metrics_rain` | dict | Métriques globales pluie |
| `per_sample_mae_temp` | ndarray (N,) | MAE locale par échantillon |

**Onglet Température** — courbe prédite vs réelle + ombrage de l'écart, métriques de l'échantillon dans le titre.

**Onglet Pluie** — probabilité prédite vs cible binaire, seuil 0.5 visible, accuracy de l'échantillon.

**Onglet Vue globale** — 3 graphiques calculés une seule fois :
1. **Scatter prédit vs réel** (température) avec R² et RMSE annotés
2. **Histogramme des résidus** `(prédit − réel)` avec moyenne indiquée — un biais positif signifie une surestimation systématique
3. **MAE par heure de prédiction h+1…h+24** — montre si l'erreur augmente avec l'horizon

**Bandeau métriques globales** (en haut) :
- Température : MAE (°C), RMSE (°C), R²
- Pluie : F1, Précision, Rappel — métriques pertinentes pour une classification déséquilibrée

**Navigation** :
- Slider sur toute la plage de test
- Boutons ◀ / ▶ (pas de 1 échantillon)
- "Meilleur cas ↑" → `argmin(per_sample_mae_temp)`
- "Pire cas ↓" → `argmax(per_sample_mae_temp)`
- Saut par date (YYYY-MM-DD) ou par index

---

### 4.6 `app_gui.py`

**Rôle** : interface centrale intégrant les 4 étapes du pipeline dans une seule fenêtre.

#### Sections de l'interface

**Téléchargement** :
- Date début / fin (YYYY-MM-DD)
- Clé API Infoclimat (masquée)
- Sélection du dossier de destination
- Bouton "Télécharger" → appelle `download_infoclimat_range` de `data.py`

**Parser** :
- Paramètre `min_hours` (nombre minimum d'heures consécutives, défaut 72)
- Bouton "Nettoyer" → appelle `clean_csv_72h` de `parser.py`

**Modèle** :
- Bouton "Entraîner" → lance `train.py` en sous-processus
- Bouton "Prédire" → lance `predict_gui.py` en sous-processus
- Les sous-processus tournent dans un thread séparé, le log est mis à jour en temps réel

**Log** : zone de texte affichant la sortie standard des scripts lancés.

#### Gestion du token API

Ordre de priorité :
1. Champ de saisie dans l'interface
2. Variable d'environnement `INFOCLIMAT_TOKEN`
3. Dialogue de saisie en dernier recours

Le token saisi est injecté dans `os.environ` et dans `data.API_TOKEN` pour que le module soit cohérent en cours d'exécution.

---

### 4.7 `ann.py` (legacy)

**Rôle** : implémentation manuelle d'un réseau de neurones dense en numpy pur. Fichier d'exploration pédagogique, non utilisé dans le pipeline de production.

**Fonctionnalités** :
- `create_nn(layers)` — crée un réseau avec les dimensions spécifiées (liste d'entiers)
- `fill_nn_rand` / `fill_nn` / `heetal_init` — initialisations des poids
- `forward_pass(inputs, nn)` — propagation avant avec ReLU (couches cachées) et sigmoid (sortie)
- `backprop(nn, inputs, expected)` — rétropropagation, calcul des deltas
- `grad_descent(nn, inputs, expected, batch_size)` — descente de gradient par mini-batches
- `loss` — erreur quadratique MSE

---

### 4.8 `nn_tf.py` (legacy)

**Rôle** : premier prototype du modèle LSTM sous TensorFlow. Fichier script exécutable directement (pas de `if __name__ == "__main__"`), chemins hardcodés (`train_data.csv`, `test_data.csv`). Non utilisé dans le pipeline actuel — supplanté par `train.py`.

**Différences avec `train.py`** :
- Chemins hardcodés, pas de sélection de fichier
- Pas de sauvegarde des paramètres de normalisation
- Binarisation sur la valeur normalisée log-rain (bug corrigé dans `train.py`)
- Architecture partagée (pas de branche dédiée pluie)
- Métriques : accuracy uniquement (pas de F1/Précision/Rappel)

---

## 5. Format des données

### CSV brut (sortie de `data.py`)

```
station_id,datetime_utc,hour_utc,temp_C,pressure_hPa,humidity_pct,wind_avg,wind_dir_deg,sunshine,rain_mm
07510,2026-01-01 00:00:00,0,8.2,1021.4,87.0,15.3,220.0,,0.0
07510,2026-01-01 01:00:00,1,7.9,1021.2,88.0,14.1,215.0,,0.0
...
```

Colonnes par index (utilisé dans les scripts CSV reader) :

| Index | Colonne | Notes |
|-------|---------|-------|
| 0 | `station_id` | |
| 1 | `datetime_utc` | Format ISO ou `*` pour séparateur |
| 2 | `hour_utc` | 0–23 |
| 3 | `temp_C` | Peut être vide |
| 4 | `pressure_hPa` | Peut être vide |
| 5 | `humidity_pct` | Peut être vide |
| 6 | `wind_avg` | Peut être vide |
| 7 | `wind_dir_deg` | 0–360, peut être vide |
| 8 | `sunshine` | Non utilisé dans le modèle |
| 9 | `rain_mm` | Peut être vide |

### CSV nettoyé (sortie de `parser.py`)

Même structure, avec :
- Jours incomplets supprimés
- Lignes séparateurs `*` dans la colonne datetime entre les segments consécutifs
- Nom de fichier : `{original}_clean72h.csv`

### Valeurs manquantes

Stratégie de remplacement dans `get_norm_data_from_file` : **forward fill** — toute valeur manquante est remplacée par la dernière valeur valide connue. Cette approche est conservative et évite les artefacts d'interpolation.

---

## 6. Architecture du modèle

```
Input (48, 12)
      │
      ▼
  LSTM(64, return_sequences=True)        ← encodeur partagé
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
  LSTM(32)                                       LSTM(64)
      │                                          Dropout(0.3)
  Dense(32, relu)                                Dense(64, relu)
      │                                              │
  Dense(24)                                     Dense(24, sigmoid)
      │                                              │
  temperature output                            rain output
  shape: (24,)                                  shape: (24,)
  loss: MSE                                     loss: weighted BCE
  metric: MAE                                   metrics: Precision, Recall, F1
```

**Encodeur partagé** (`LSTM(64, return_sequences=True)`) : extrait les dépendances temporelles communes aux deux tâches. `return_sequences=True` transmet la séquence complète aux deux branches.

**Branche température** (`LSTM(32)` → `Dense(32, relu)` → `Dense(24)`) : plus légère, la prévision de température est une tâche de régression relativement régulière. Activation linéaire en sortie — pas de contrainte sur la plage de valeurs.

**Branche pluie** (`LSTM(64)` → `Dropout(0.3)` → `Dense(64, relu)` → `Dense(24, sigmoid)`) : plus profonde, la détection de pluie est une tâche de classification binaire avec fort déséquilibre. Activation sigmoid en sortie — probabilité dans [0, 1]. Seuil de décision : 0.5.

**Paramètres d'entraînement** :

| Paramètre | Valeur |
|-----------|--------|
| Optimiseur | Adam |
| Epochs max | 30 |
| Batch size | 32 |
| Validation split | 20% |
| Early stopping | `patience=10`, `restore_best_weights=True` |

**Dimensionnement** : avec `lookback=48` et `window_size=24`, chaque CSV de N heures produit `N − 48 − 24 = N − 72` échantillons d'entraînement.

---

## 7. Améliorations apportées

Six améliorations ont été implémentées sur le code initial en deux sessions. Elles sont listées ci-dessous par ordre d'implémentation.

### P1 — Correction du seuil de binarisation de la pluie

**Problème** : dans le code original, la binarisation de la cible `Y_rain` était appliquée sur la valeur **normalisée** de `log(1+rain_mm)` avec un seuil de `0.5`. Ce seuil correspond à `rain > exp(μ_r + 0.5×σ_r) − 1` — une quantité variable et non interprétable physiquement. Selon les données d'entraînement, cela pouvait catégoriser comme "sec" la majorité des heures effectivement pluvieuses.

**Correction** : les valeurs brutes `rain_mm` sont désormais conservées séparément dans `raw_rain_chunks`. La binarisation est effectuée directement sur ces valeurs : `1 si rain_mm > 0.1 sinon 0`. Le seuil de 0.1 mm/h est une convention météorologique standard pour distinguer "précipitation" de "humidité résiduelle".

**Fichiers modifiés** : `train.py`, `predict_gui.py`

---

### P2 — Pondération de la classe positive (pluie)

**Problème** : avec `binary_crossentropy` standard et un jeu déséquilibré (ex: 25% d'heures pluvieuses), le modèle minimise la loss en prédisant "toujours sec" — atteignant 75% d'accuracy sans jamais détecter une goutte de pluie.

**Correction** : la fonction de loss `make_weighted_bce(pos_weight)` pondère chaque pixel de pluie `pos_weight` fois plus qu'un pixel sec :

```
loss = mean((y_true × pos_weight + (1 − y_true)) × BCE(y_true, y_pred))
```

`pos_weight` est calculé automatiquement depuis `Y_rain_train` : `n_négatifs / n_positifs`. Si 25% du temps il pleut, `pos_weight ≈ 3.0` — le modèle est pénalisé 3× plus fortement pour chaque heure de pluie manquée.

**Fichiers modifiés** : `train.py`

---

### P3 — Tendance de pression comme feature

**Problème** : la pression brute à l'instant t est un indicateur de conditions météo actuelles, mais ne renseigne pas sur leur évolution. Un modèle qui ne voit que `P(t)` ne peut pas distinguer une pression de 1010 hPa en hausse (amélioration) d'une pression de 1010 hPa en chute (dégradation imminente).

**Correction** : ajout de `press_tend_3h[t] = press_norm[t] − press_norm[t−3]` comme 9ème feature. Une valeur négative indique une dépression en approche.

**Fichiers modifiés** : `train.py`, `predict_gui.py`

---

### P4 — Métriques F1 / Précision / Rappel

**Problème** : l'accuracy est une métrique inadaptée pour une classification déséquilibrée. Un modèle qui dit "jamais de pluie" obtient ~75% d'accuracy — score qui semble bon mais masque un modèle inutile.

**Correction** :
- Ajout de la classe `F1Score` (métrique Keras custom) compilée dans le modèle — visible à chaque epoch dans les logs d'entraînement
- `Precision` et `Recall` Keras ajoutés également
- Dans `predict_gui.py`, le bandeau global affiche F1, Précision, Rappel calculés sur tout X_test

**Interprétation** :
- **Précision** : parmi les heures où le modèle dit "pluie", combien étaient vraiment pluvieuses ?
- **Rappel** : parmi les heures réellement pluvieuses, combien le modèle les a-t-il détectées ?
- **F1** : moyenne harmonique des deux — résumé en un seul chiffre

**Fichiers modifiés** : `train.py`, `predict_gui.py`

---

### P5 — Vent (vitesse + direction) dans les features

**Problème** : les 8 features originales ignoraient le vent, qui est l'un des meilleurs prédicteurs de pluie à Bordeaux (vent de sud-ouest = front atlantique pluvieux).

**Correction** : ajout de 3 features (indices 9–11) :
- `wind_avg` normalisé : `(W − μ_W) / σ_W`
- `wind_dir_sin` = `sin(2π × direction / 360)`
- `wind_dir_cos` = `cos(2π × direction / 360)`

La direction est encodée en sin/cos car c'est une variable circulaire : 350° et 10° sont séparés de 20° et non de 340°.

`μ_W` et `σ_W` sont calculés sur TRAIN et sauvegardés dans `norm_params.pkl`.

**Fichiers modifiés** : `train.py`, `predict_gui.py`

---

### P6 — Branche LSTM dédiée à la pluie

**Problème** : dans l'architecture originale, température et pluie partageaient **tout** le trunk LSTM → Dense, forçant la même représentation interne à servir deux tâches très différentes (régression lisse vs classification binaire déséquilibrée).

**Correction** : architecture bifurquée après le premier LSTM :

- Le LSTM partagé (`return_sequences=True`) produit la séquence complète
- La **branche température** utilise un LSTM léger (32 unités) — suffisant pour une régression régulière
- La **branche pluie** utilise un LSTM dédié (64 unités) avec Dropout(0.3) et Dense(64) — plus de capacité pour capturer les patterns de précipitation

**Fichiers modifiés** : `train.py`

---

### Résumé des modifications par fichier

| Fichier | Modifications |
|---------|--------------|
| `train.py` | P1 (seuil), P2 (weighted BCE + `F1Score`), P3 (tendance pression), P4 (métriques), P5 (vent), P6 (branche dédiée) |
| `predict_gui.py` | P1 (seuil), P3 (tendance pression), P4 (F1/Prec/Recall dans bandeau), P5 (vent), amélioration UI (scatter, histogramme résidus, MAE par heure, slider, meilleur/pire cas) |

---

## 8. Guide de démarrage

### Prérequis

```bash
pip install tensorflow pandas numpy matplotlib requests scikit-learn
```

### Configuration de la clé API

```bash
export INFOCLIMAT_TOKEN="votre_token_infoclimat"
```

Ou saisie directement dans l'interface lors du téléchargement.

### Lancement de l'application complète

```bash
cd codes/
python app_gui.py
```

### Exécution étape par étape

```bash
# 1. Télécharger les données
python data.py

# 2. Vérifier la qualité (optionnel)
python check_data.py ../downloads/observations_2025-01-01_2025-12-31.csv

# 3. Nettoyer les données
python parser.py ../downloads/observations_2025-01-01_2025-12-31.csv

# 4. Entraîner le modèle (ouvre une boîte de dialogue de sélection de fichier)
python train.py

# 5. Visualiser les prédictions
python predict_gui.py
```

### Fichiers attendus pour `train.py`

- **Fichier d'entraînement** : sélectionné via dialogue (CSV nettoyé par `parser.py`)
- **Fichier de test** : `codes/test_data.csv` ou `test_data.csv` à la racine

### Fichiers attendus pour `predict_gui.py`

- `artifacts/weather_model.keras`
- `artifacts/norm_params.pkl`
- `codes/test_data.csv` ou `test_data.csv` à la racine

> **Note** : après toute modification du pipeline de données (P1–P5–P6), il est nécessaire de réentraîner le modèle. Le modèle précédent n'est plus compatible (changement de `n_inputs` : 8 → 12, nouvelle architecture).
