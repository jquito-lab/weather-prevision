"""weather-prevision

Télécharge des observations météo depuis l'API OpenData Infoclimat (v2) au format CSV,
nettoie les données, puis consolide le tout dans un seul fichier CSV exploitable.

Points clés :
- Limitation API : 7 jours maximum par requête → découpage automatique en blocs.
- Les colonnes disponibles dépendent de la station (capteurs) : certaines grandeurs peuvent être absentes.
- Le champ `sunshine` peut représenter soit une durée d'ensoleillement, soit un rayonnement (W/m²), selon la station.
"""

import io
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import simpledialog, messagebox

# =============================
#  Paramètres Infoclimat
# =============================

# ⚠️ Ne stocke pas la clé en dur dans le code (elle est liée à ton compte/usage et peut fuiter).
# Exporter la variable d'environnement avant d'exécuter le script :
#   export INFOCLIMAT_TOKEN="..."
API_TOKEN = os.getenv("INFOCLIMAT_TOKEN", "")


def mask_token_in_url(url: str) -> str:
    """Masque la valeur du paramètre `token=` dans une URL pour éviter de logguer une clé API."""
    if "token=" not in url:
        return url
    head, tail = url.split("token=", 1)
    # tail may contain other params after token; keep them but mask token value
    if "&" in tail:
        _token, rest = tail.split("&", 1)
        return head + "token=***" + "&" + rest
    return head + "token=***"


STATION_ID = "07510"  # Identifiant de la station Bordeaux-Mérignac sur InfoClimat


BASE_URL = "https://www.infoclimat.fr/opendata/"
# Dossier par défaut où une app peut déposer les CSV téléchargés
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "downloads"))


def build_url(station_id: str, start_date: str, end_date: str, fmt: str = "csv") -> str:
    """Construit l'URL d'appel à l'API OpenData Infoclimat v2 (export CSV ou JSON).

    Args:
        station_id: Identifiant Infoclimat de la station (ex: "07510").
        start_date: Date de début au format YYYY-MM-DD (incluse).
        end_date: Date de fin au format YYYY-MM-DD (incluse).
        fmt: format de sortie ("csv" ou "json").

    Returns:
        URL complète prête à être appelée via HTTP GET.

    Exemple :
        https://www.infoclimat.fr/opendata/?version=2&method=get&format=csv&stations[]=07510&start=2026-01-12&end=2026-01-14&token=...
    """
    if not API_TOKEN:
        raise RuntimeError(
            "Clé API manquante : définis INFOCLIMAT_TOKEN dans tes variables d'environnement."
        )
    return (
        f"{BASE_URL}"
        f"?version=2"
        f"&method=get"
        f"&format={fmt}"
        f"&stations[]={station_id}"
        f"&start={start_date}"
        f"&end={end_date}"
        f"&token={API_TOKEN}"
    )


def fetch_csv_from_infoclimat(url: str) -> str:
    """Télécharge le CSV brut depuis l'API Infoclimat et retourne le texte.

    Améliorations :
    - timeout réseau
    - diagnostic détaillé quand l'API renvoie 400/403/etc.
    """
    print("Requête :", mask_token_in_url(url))

    # Un User-Agent explicite évite parfois des refus côté serveur/proxy.
    headers = {"User-Agent": "weather-prevision/1.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Erreur réseau lors de l'appel Infoclimat: {e}") from e

    print("Status HTTP:", resp.status_code)
    txt = resp.text or ""

    # L'API peut renvoyer un message d'erreur dans le corps.
    lowered = txt.lower()
    if "wrong ip" in lowered or "wrong ip adress" in lowered:
        raise RuntimeError(
            "Infoclimat renvoie 'wrong ip adress'. "
            "Vérifie que l'IP publique utilisée pour lancer le script correspond bien à celle déclarée "
            "lors de la génération du token."
        )

    # Si l'API renvoie une erreur HTTP, affiche le corps pour aider au debug.
    if resp.status_code >= 400:
        # On tronque pour éviter d'afficher des pavés immenses.
        body_preview = txt.strip().replace("\r", "")
        if len(body_preview) > 800:
            body_preview = body_preview[:800] + "..."

        raise RuntimeError(
            "Erreur Infoclimat (HTTP %s).\n" % resp.status_code
            + "Corps de réponse (extrait):\n"
            + body_preview
            + "\n\nPistes courantes :\n"
            + "- token invalide/expiré ou mal copié\n"
            + "- token généré pour une autre IP publique (cas fréquent)\n"
            + "- station non autorisée pour ton type d'usage (commercial/non-commercial)\n"
            + "- paramètres start/end refusés (période > 7 jours, format, etc.)\n"
        )

    return txt

def parse_infoclimat_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse le CSV Infoclimat en DataFrame et nettoie les lignes parasites.

    - Ignore les lignes commençant par '#'
    - Supprime la ligne d’unités (dh_utc = 'YYYY-MM-DD hh:mm:ss')
    - Supprime les éventuelles lignes où 'station_id' == 'station_id' (header dupliqué)
    """
    buffer = io.StringIO(csv_text)

    # Lecture du CSV brut renvoyé par Infoclimat.
    # - séparateur ';'
    # - lignes de commentaires commencent par '#'
    # - certaines lignes peuvent être mal formées → on les ignore pour éviter un crash.
    df_raw = pd.read_csv(
        buffer,
        sep=";",
        comment="#",
        engine="python",
        on_bad_lines="skip",  # saute les lignes mal formées -> entraîne un crash du script
    )

    # Garde-fou : si le CSV est vide ou ne contient pas la colonne attendue 'dh_utc', on retourne un DataFrame vide.
    if df_raw.empty or "dh_utc" not in df_raw.columns:
        return pd.DataFrame()

    # Infoclimat ajoute parfois une ligne d'unités (dh_utc = "YYYY-MM-DD hh:mm:ss") : on la supprime.
    mask_units = df_raw["dh_utc"].astype(str).str.contains("YYYY/MM/DD", na=False)
    df = df_raw.loc[~mask_units].copy()

    # Par sécurité : suppression d'un éventuel header dupliqué au milieu du fichier.
    df = df[df["dh_utc"] != "dh_utc"]

    # Conversion de la colonne date/heure (UTC) en datetime.
    df["dh_utc"] = pd.to_datetime(df["dh_utc"], errors="coerce")

    # Colonnes candidates à convertir en numérique (les stations n'ont pas toutes les mêmes capteurs).
    numeric_cols = [
        "temperature",
        "pression",
        "humidite",
        "pluie_1h",
        "pluie_3h",
        "pluie_6h",
        "pluie_12h",
        "pluie_24h",
        # Vent
        "vent_moyen",
        "vent_direction",
        # Ensoleillement / rayonnement (selon stations)
        "ensoleillement",
        "ensoleillement_1h",
        "rayonnement",
        "rayonnement_solaire",
    ]
    # Conversion en float (valeurs non numériques → NaN).
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pluie : on choisit la meilleure colonne disponible (selon ce que fournit la station).
    rain_col = None
    for candidate in ["pluie_1h", "pluie_3h", "pluie_24h"]:
        if candidate in df.columns:
            rain_col = candidate
            break

    # Utilitaire : retourne le premier nom de colonne existant dans df parmi une liste de candidats.
    def first_existing(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Vent / soleil : les noms varient selon les stations → on tente plusieurs colonnes possibles.
    wind_avg_col = first_existing(["vent_moyen", "vent_moyen_10m", "vent_moyen_kmh"])
    wind_dir_col = first_existing(["vent_direction", "direction_vent", "vent_dir"])
    sun_col = first_existing([
        "ensoleillement",
        "ensoleillement_1h",
        "rayonnement",
        "rayonnement_solaire",
    ])

    out = pd.DataFrame()
    if "station_id" in df.columns:
        out["station_id"] = df["station_id"].astype(str)
    else:
        out["station_id"] = STATION_ID

    out["datetime_utc"] = df["dh_utc"]
    out["temp_C"] = df.get("temperature")
    out["pressure_hPa"] = df.get("pression")
    out["humidity_pct"] = df.get("humidite")

    # Colonnes optionnelles : si la station ne fournit pas la grandeur, on met NA.
    # NOTE : 'sunshine' peut être :
    # - une durée d'ensoleillement (ex: minutes sur l'heure / la journée)
    # - OU un rayonnement solaire (W/m²)
    # selon la colonne réellement trouvée dans le CSV.
    out["wind_avg"] = df[wind_avg_col] if wind_avg_col is not None else pd.NA
    out["wind_dir_deg"] = df[wind_dir_col] if wind_dir_col is not None else pd.NA
    out["sunshine"] = df[sun_col] if sun_col is not None else pd.NA

    if rain_col is not None:
        out["rain_mm"] = df[rain_col]
    else:
        out["rain_mm"] = 0.0

    out["hour_utc"] = out["datetime_utc"].dt.hour

    # On garde un ordre de colonnes fixe
    out = out[
        [
            "station_id",
            "datetime_utc",
            "hour_utc",
            "temp_C",
            "pressure_hPa",
            "humidity_pct",
            "wind_avg",
            "wind_dir_deg",
            "sunshine",
            "rain_mm",
        ]
    ]

    # On enlève les lignes sans datetime
    out = out[out["datetime_utc"].notna()]

    return out


class DateFileDialog(simpledialog.Dialog):
    """Boîte de dialogue Tkinter : saisie date début/fin + nom du fichier CSV de sortie."""

    def body(self, master):
        tk.Label(master, text="Date de début (YYYY/MM/DD) :").grid(row=0, column=0, sticky="w")
        tk.Label(master, text="Date de fin (YYYY/MM/DD) :").grid(row=1, column=0, sticky="w")
        tk.Label(master, text="Nom du fichier CSV de sortie :").grid(row=2, column=0, sticky="w")
        tk.Label(master, text="Clé API Infoclimat (optionnel) :").grid(row=3, column=0, sticky="w")

        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        # Nom de fichier proposé par défaut (modifiable par l'utilisateur).
        self.file_var = tk.StringVar(value="observations_infoclimat_full.csv")
        self.token_var = tk.StringVar(value=os.getenv("INFOCLIMAT_TOKEN", ""))

        self.start_entry = tk.Entry(master, textvariable=self.start_var)
        self.end_entry = tk.Entry(master, textvariable=self.end_var)
        self.file_entry = tk.Entry(master, textvariable=self.file_var)
        self.token_entry = tk.Entry(master, textvariable=self.token_var, show="*")

        self.start_entry.grid(row=0, column=1, padx=5, pady=2)
        self.end_entry.grid(row=1, column=1, padx=5, pady=2)
        self.file_entry.grid(row=2, column=1, padx=5, pady=2)
        self.token_entry.grid(row=3, column=1, padx=5, pady=2)

        return self.start_entry  # focus initial

    def apply(self):
        start_str = self.start_var.get().strip()
        end_str = self.end_var.get().strip()
        filename = self.file_var.get().strip()
        token = self.token_var.get().strip()
        # Résultat renvoyé au code appelant (tuple).
        self.result = (start_str, end_str, filename, token)


def ask_date_and_filename_via_popup():
    """
    Demande date début / fin et nom de fichier via une seule pop-up Tkinter,
    retourne (date_debut, date_fin, nom_fichier).
    """
    root = tk.Tk()
    root.withdraw()  # pas de fenêtre principale visible

    dialog = DateFileDialog(root, "Paramètres de téléchargement")
    if dialog.result is None:
        # L'utilisateur a fermé/annulé la fenêtre.
        raise SystemExit("Saisie annulée")

    start_str, end_str, filename, token = dialog.result

    # Si l'utilisateur fournit une clé via la pop-up, elle prime sur la variable d'environnement.
    global API_TOKEN
    if token:
        API_TOKEN = token

    if not filename:
        messagebox.showerror("Erreur", "Le nom de fichier ne peut pas être vide.")
        raise SystemExit(1)

    # Normalisation : ajoute une extension .csv si l'utilisateur n'en a pas mis.
    if "." not in filename:
        filename = filename + ".csv"

    # Validation : format des dates (accepte YYYY-MM-DD ou DD/MM/YYYY).
    def _parse_date(s):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        raise ValueError("invalid date")

    try:
        start_date = _parse_date(start_str)
        end_date = _parse_date(end_str)
    except ValueError:
        messagebox.showerror(
            "Erreur",
            "Format de date invalide. Utilise YYYY-MM-DD ou YYYY/MM/DD.",
        )
        raise SystemExit(1)

    # Validation : cohérence des dates.
    if end_date < start_date:
        messagebox.showerror("Erreur", "La date de fin est avant la date de début.")
        raise SystemExit(1)

    return start_date, end_date, filename


def generate_chunks(start_date, end_date, max_days=7):
    """Découpe une période en blocs pour respecter la limite Infoclimat (7 jours max par requête).

    Args:
        start_date: date de début (incluse)
        end_date: date de fin (incluse)
        max_days: taille maximale d'un bloc (7 par défaut)

    Yields:
        Tuples (chunk_start, chunk_end) inclusifs.

    Exemple :
        start=2026-01-01, end=2026-01-10, max_days=7 →
        (2026-01-01..2026-01-07), puis (2026-01-08..2026-01-10)
    """
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ----------------------------------------------------------
# Fonction réutilisable pour une app (Tkinter ou autre)
# ----------------------------------------------------------
def download_infoclimat_range(
    start_date,
    end_date,
    output_dir: str | None = None,
    filename: str | None = None,
    station_id: str = STATION_ID,
    max_days_per_request: int = 7,
) -> tuple[str, pd.DataFrame]:
    """Télécharge une plage de dates Infoclimat, consolide et sauvegarde un CSV.

    Cette fonction est pensée pour être appelée depuis une application (Tkinter ou autre)
    sans afficher de popups.

    Args:
        start_date: datetime.date (incluse)
        end_date: datetime.date (incluse)
        output_dir: dossier cible (créé si absent). Par défaut: ../downloads
        filename: nom de fichier (optionnel). Si None, un nom est généré.
        station_id: ID de station Infoclimat
        max_days_per_request: limite Infoclimat (7 jours max)

    Returns:
        (output_path, df_all)
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"infoclimat_{station_id}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    output_path = os.path.join(output_dir, filename)

    all_dfs: list[pd.DataFrame] = []

    for chunk_start, chunk_end in generate_chunks(start_date, end_date, max_days=max_days_per_request):
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"\nTéléchargement du bloc {start_str} -> {end_str}")

        url = build_url(station_id, start_str, end_str, fmt="csv")
        csv_text = fetch_csv_from_infoclimat(url)
        df_chunk = parse_infoclimat_csv(csv_text)

        if df_chunk.empty:
            print("⚠️  Aucune donnée dans ce bloc.")
        else:
            print(f"  -> {len(df_chunk)} lignes")
            all_dfs.append(df_chunk)

    if not all_dfs:
        raise RuntimeError("Aucune donnée récupérée sur toute la période demandée.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values("datetime_utc")
    df_all = df_all.drop_duplicates(subset=["datetime_utc", "station_id"])

    df_all.to_csv(output_path, index=False)
    return output_path, df_all


def main():
    # 1) Demande des paramètres à l'utilisateur (dates + fichier de sortie).
    start_date, end_date, output_file = ask_date_and_filename_via_popup()
    print(f"Période demandée : {start_date} -> {end_date}")
    print(f"Fichier de sortie : {output_file}")

    # 2) Téléchargement + consolidation + export
    # Ici on écrit dans le dossier courant (comportement identique à avant),
    # mais la fonction est réutilisable par une app.
    out_path, df_all = download_infoclimat_range(
        start_date=start_date,
        end_date=end_date,
        output_dir=os.getcwd(),
        filename=output_file,
        station_id=STATION_ID,
        max_days_per_request=7,
    )

    print(f"\nDonnées consolidées sauvegardées dans {out_path}")
    print(f"Nombre total de lignes : {len(df_all)}")


if __name__ == "__main__":
    main()
