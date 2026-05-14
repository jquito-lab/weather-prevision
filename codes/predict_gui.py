# -*- coding: utf-8 -*-
import argparse
import os
import sys
import csv
import pickle

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk, filedialog
from tensorflow.keras.models import load_model

# ── Palette (identique à app_gui.py) ─────────────────────────────────── #
BG       = "#0D1117"
SURFACE  = "#161B22"
SURFACE2 = "#21262D"
SURFACE3 = "#30363D"
BORDER   = "#30363D"
BLUE     = "#1F6FEB"
BLUE_H   = "#388BFD"
GREEN    = "#238636"
GREEN_H  = "#3FB950"
AMBER    = "#BB8009"
RED      = "#DA3633"
RED_H    = "#F85149"
TEXT     = "#E6EDF3"
TEXT2    = "#8B949E"
TEXT3    = "#6E7681"
ACCENT   = "#58A6FF"

C_PRED   = "#F778BA"   # rose — prédiction
C_TRUE   = "#3FB950"   # vert — réel
C_FILL   = "#58A6FF"   # bleu (fill_between)

_FF  = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"
_FM  = "SF Mono"        if sys.platform == "darwin" else "Consolas"
_F9  = (_FF, 9)
_F10 = (_FF, 10)
_F11B = (_FF, 11, "bold")
_FM9  = (_FM, 9)


# ── Helpers UI ────────────────────────────────────────────────────────── #

class Btn(tk.Label):
    def __init__(self, parent, text, command,
                 bg=SURFACE3, fg=TEXT, hover=None,
                 padx=12, pady=5, font=None, **kw):
        super().__init__(parent, text=text, bg=bg, fg=fg,
                         cursor="hand2", padx=padx, pady=pady,
                         font=font or _F10, **kw)
        self._bg = bg
        self._hover = hover or SURFACE2
        self.bind("<Enter>",    lambda e: self.config(bg=self._hover))
        self.bind("<Leave>",    lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: command())


def _dark_ax(ax, fig=None):
    if fig is not None:
        fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE2)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


def _divider(parent, color=BORDER):
    tk.Frame(parent, bg=color, height=1).pack(fill="x")


def _setup_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(".",           background=BG, foreground=TEXT, font=_F10)
    style.configure("TFrame",      background=BG)
    style.configure("TNotebook",   background=SURFACE, borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=SURFACE2, foreground=TEXT2,
                    padding=[16, 7], font=_F10)
    style.map("TNotebook.Tab",
              background=[("selected", SURFACE3), ("active", SURFACE3)],
              foreground=[("selected", TEXT),      ("active", TEXT)])
    style.configure("TScale",
                    background=BG, troughcolor=SURFACE3,
                    sliderlength=14)
    style.configure("TEntry",
                    fieldbackground=SURFACE2, foreground=TEXT,
                    insertcolor=TEXT, bordercolor=BORDER,
                    focuscolor=BLUE, relief="flat", padding=5)
    style.map("TEntry", bordercolor=[("focus", BLUE)])


# -------------------- Utils paths -------------------- #

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model",    default=None)
_ap.add_argument("--norm",     default=None)
_ap.add_argument("--n-inputs", type=int, default=None, dest="n_inputs")
_cli, _ = _ap.parse_known_args()

MODEL_PATH = _cli.model if _cli.model else first_existing([
    os.path.join(ARTIFACTS_DIR, "weather_model.keras"),
    os.path.join(PROJECT_DIR, "weather_model.keras"),
    os.path.join(BASE_DIR, "weather_model.keras"),
])

NORM_PATH = _cli.norm if _cli.norm else first_existing([
    os.path.join(ARTIFACTS_DIR, "norm_params.pkl"),
    os.path.join(PROJECT_DIR, "norm_params.pkl"),
    os.path.join(BASE_DIR, "norm_params.pkl"),
])

DOWNLOAD_DIR = os.path.join(PROJECT_DIR, "downloads")


def pick_test_csv():
    root = tk.Tk()
    root.withdraw()
    initial = DOWNLOAD_DIR if os.path.isdir(DOWNLOAD_DIR) else BASE_DIR
    path = filedialog.askopenfilename(
        title="Choisir le fichier CSV de prédiction",
        initialdir=initial,
        filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*")],
    )
    root.destroy()
    if not path:
        raise RuntimeError("Aucun fichier de prédiction sélectionné.")
    return path


if MODEL_PATH is None:
    raise FileNotFoundError("weather_model.keras introuvable.")
if NORM_PATH is None:
    raise FileNotFoundError("norm_params.pkl introuvable.")

TEST_CSV = pick_test_csv()

print("MODEL:", MODEL_PATH)
print("NORM :", NORM_PATH)
print("TEST :", TEST_CSV)


# -------------------- Data pipeline -------------------- #

def calculate_date(str_date):
    month = int(str_date[0:2])
    n_days = 0
    for i in range(month - 1):
        if i + 1 in [1, 3, 5, 7, 8, 10, 12]:
            n_days += 31
        elif i + 1 == 2:
            n_days += 28
        else:
            n_days += 30
    day = int(str_date[3:5])
    if month == 2 and day == 29:
        n_days += 28
    else:
        n_days += day
    return n_days - 1


def get_norm_data_and_dates_from_file(filename, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W):
    """Retourne (chunks, raw_rain_chunks, date_chunks).

    Feature layout (12 colonnes) — identique à train.py :
        0-3  encodages temporels (hour/day sin-cos)
        4    humidity (/100)
        5    temp      (normalisé)
        6    press     (normalisé)
        7    rain_log  (normalisé)
        8    press_tend_3h                   [PRIORITÉ 3]
        9    wind_avg  (normalisé)           [PRIORITÉ 5]
       10    wind_dir_sin                    [PRIORITÉ 5]
       11    wind_dir_cos                    [PRIORITÉ 5]
    """
    chunks = []
    raw_rain_chunks = []
    date_chunks = []
    sep_found = False

    humidity, temp, rain_log, press, rain_raw = [], [], [], [], []
    wind_avg_raw, wind_dir_raw = [], []
    day_sin, day_cos, hour_sin, hour_cos = [], [], [], []
    dates = []
    n = 0

    def _flush(n):
        humidity_arr = np.array(humidity) / 100.0
        temp_arr     = (np.array(temp)         - mu_T) / sig_T
        press_arr    = (np.array(press)        - mu_P) / sig_P
        rain_arr     = (np.array(rain_log)     - mu_r) / sig_r
        wind_arr     = (np.array(wind_avg_raw) - mu_W) / sig_W

        press_tend = np.zeros(n)
        for i in range(n):
            press_tend[i] = press_arr[i] - press_arr[max(0, i - 3)]

        dir_rad    = np.array(wind_dir_raw) * (2.0 * np.pi / 360.0)
        wind_d_sin = np.sin(dir_rad)
        wind_d_cos = np.cos(dir_rad)

        chunk = np.array([
            [hour_sin[i], hour_cos[i], day_sin[i], day_cos[i],
             humidity_arr[i], temp_arr[i], press_arr[i], rain_arr[i], press_tend[i],
             wind_arr[i], wind_d_sin[i], wind_d_cos[i]]
            for i in range(n)
        ], dtype=np.float32)

        return chunk, np.array(rain_raw, dtype=np.float32)

    with open(filename, mode="r") as file:
        csvfile = csv.reader(file)
        next(csvfile)

        for line in csvfile:
            if line[1] != "*":
                n += 1
                dates.append(line[1])

                date = calculate_date(line[1][5:10])
                day_sin.append(np.sin(2 * np.pi * date / 365))
                day_cos.append(np.cos(2 * np.pi * date / 365))

                hour = float(line[2])
                hour_sin.append(np.sin(2 * np.pi * hour / 24))
                hour_cos.append(np.cos(2 * np.pi * hour / 24))

                if line[3] == "":
                    temp.append(temp[-1])
                else:
                    temp.append(float(line[3]))

                if line[4] == "":
                    press.append(press[-1])
                else:
                    press.append(float(line[4]))

                if line[5] == "":
                    humidity.append(humidity[-1])
                else:
                    humidity.append(float(line[5]))

                if line[9] == "":
                    rain_log.append(rain_log[-1])
                    rain_raw.append(rain_raw[-1] if rain_raw else 0.0)
                else:
                    r = float(line[9])
                    rain_log.append(np.log(1 + r))
                    rain_raw.append(r)

                if line[6] == "":
                    wind_avg_raw.append(wind_avg_raw[-1] if wind_avg_raw else 0.0)
                else:
                    wind_avg_raw.append(float(line[6]))

                if line[7] == "":
                    wind_dir_raw.append(wind_dir_raw[-1] if wind_dir_raw else 0.0)
                else:
                    wind_dir_raw.append(float(line[7]))

            else:
                sep_found = True
                chunk, raw_rain = _flush(n)
                chunks.append(chunk)
                raw_rain_chunks.append(raw_rain)
                date_chunks.append(list(dates))

                humidity, temp, rain_log, press, rain_raw = [], [], [], [], []
                wind_avg_raw, wind_dir_raw = [], []
                day_sin, day_cos, hour_sin, hour_cos = [], [], [], []
                dates = []
                n = 0

        if not sep_found:
            chunk, raw_rain = _flush(n)
            chunks.append(chunk)
            raw_rain_chunks.append(raw_rain)
            date_chunks.append(list(dates))

    return chunks, raw_rain_chunks, date_chunks


def build_xy_and_sample_dates(chunks, raw_rain_chunks, date_chunks, lookback=48, window_size=24, rain_mm_threshold=0.1):
    """[PRIORITÉ 1] Binarisation sur mm bruts avec seuil physique 0.1 mm/h."""
    X, Y_rain, Y_temp, sample_dates = [], [], [], []
    for chunk, raw_rain, dates in zip(chunks, raw_rain_chunks, date_chunks):
        for i in range(lookback, len(chunk) - window_size):
            X.append(chunk[i - lookback:i])
            sample_dates.append(dates[i])

        for j in range(lookback, len(chunk) - window_size):
            rain_target = np.array(
                [1.0 if raw_rain[k] > rain_mm_threshold else 0.0 for k in range(j, j + window_size)],
                dtype=np.float32,
            )
            temp_input = np.array([chunk[k][5] for k in range(j, j + window_size)], dtype=np.float32)
            Y_rain.append(rain_target)
            Y_temp.append(temp_input)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y_temp, dtype=np.float32),
        np.array(Y_rain, dtype=np.float32),
        sample_dates
    )


def denormalize_temp(tab, mu_T, sig_T):
    return tab * sig_T + mu_T


# -------------------- Metrics -------------------- #

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> dict:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "r2": np.nan}

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(eps, np.abs(yt))
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else np.nan

    return {"mae": mae, "mape": mape, "rmse": rmse, "r2": r2}


def fmt_metrics(m: dict) -> str:
    def f(x, nd=2):
        return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"
    return f"MAE={f(m['mae'])}  MAPE={f(m['mape'])}%  RMSE={f(m['rmse'])}  R²={f(m['r2'])}"


def compute_per_sample_mae(y_true_real: np.ndarray, y_pred_real: np.ndarray) -> np.ndarray:
    """MAE par échantillon (shape: [N])."""
    return np.mean(np.abs(y_pred_real - y_true_real), axis=1)


# -------------------- Dialogs -------------------- #

class ForecastDayDialog(simpledialog.Dialog):
    def body(self, master):
        master.configure(bg=BG)
        for r, (lbl, attr) in enumerate([
            ("Jour à prévoir (YYYY-MM-DD) :", "day_var"),
            ("Index (optionnel) :",           "idx_var"),
        ]):
            setattr(self, attr, tk.StringVar())
            tk.Label(master, text=lbl, bg=BG, fg=TEXT2,
                     font=_F9).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            e = tk.Entry(master, textvariable=getattr(self, attr),
                         bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                         relief="flat", bd=0,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=BLUE,
                         font=_F10)
            e.grid(row=r, column=1, padx=8, pady=4, sticky="ew", ipady=5)
            if r == 0:
                self._first = e
        return self._first

    def apply(self):
        self.result = (self.day_var.get().strip(), self.idx_var.get().strip())


def find_idx_for_day(sample_dates, day_str):
    day_str = day_str.replace("/", "-")
    for i, d in enumerate(sample_dates):
        if d.replace("/", "-").startswith(day_str):
            return i
    return None


# -------------------- Main Navigator Window -------------------- #

class ForecastNavigator(tk.Toplevel):
    def __init__(
        self,
        master,
        *,
        idx_init: int,
        sample_dates,
        y_temp_pred_real,
        y_temp_true_real,
        y_rain_pred,
        y_rain_true,
        model,
        x_test,
        y_temp_test,
        y_rain_test,
        global_metrics_temp: dict,
        global_metrics_rain: dict,
        per_sample_mae_temp: np.ndarray,
    ):
        super().__init__(master)
        self.title("Prévisions météo — Résultats & Comparaison")
        self.geometry("1280x860")
        self.minsize(900, 640)
        self.configure(bg=BG)
        _setup_style(self)

        self.sample_dates = sample_dates
        self.y_temp_pred_real = y_temp_pred_real
        self.y_temp_true_real = y_temp_true_real
        self.y_rain_pred = y_rain_pred
        self.y_rain_true = y_rain_true
        self.model = model
        self.x_test = x_test
        self.y_temp_test = y_temp_test
        self.y_rain_test = y_rain_test
        self.global_metrics_temp = global_metrics_temp
        self.global_metrics_rain = global_metrics_rain
        self.per_sample_mae_temp = per_sample_mae_temp

        self.idx = max(0, min(idx_init, len(sample_dates) - 1))

        self._build_ui()
        self._build_global_tab()
        self._build_global_rain_tab()
        self.refresh()

    # ---- UI construction ----

    def _build_ui(self):
        def fv(x):
            return "—" if (x is None or not np.isfinite(x)) else f"{x:.3f}"

        m_t = self.global_metrics_temp

        # Métriques pluie
        y_pred_bin = (self.y_rain_pred > 0.5).ravel()
        y_true_bin = self.y_rain_true.ravel()
        tp = float(np.sum((y_pred_bin >= 0.5) & (y_true_bin >= 0.5)))
        fp = float(np.sum((y_pred_bin >= 0.5) & (y_true_bin < 0.5)))
        fn = float(np.sum((y_pred_bin < 0.5)  & (y_true_bin >= 0.5)))
        prec_val = tp / (tp + fp + 1e-7)
        rec_val  = tp / (tp + fn + 1e-7)
        f1_val   = 2.0 * prec_val * rec_val / (prec_val + rec_val + 1e-7)

        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=SURFACE)
        hdr.pack(fill="x")
        _divider(hdr, BORDER)
        inner = tk.Frame(hdr, bg=SURFACE)
        inner.pack(fill="x", padx=16, pady=10)
        tk.Label(inner, text="⛅", bg=SURFACE, fg=ACCENT,
                 font=(_FF, 18)).pack(side="left")
        tk.Label(inner, text=" Weather Prevision", bg=SURFACE, fg=TEXT,
                 font=(_FF, 15, "bold")).pack(side="left")
        tk.Label(inner, text="  —  Résultats & Comparaison",
                 bg=SURFACE, fg=TEXT2, font=_F9).pack(side="left")
        _divider(self, BORDER)

        # ── Bannière métriques ────────────────────────────────────────
        mb = tk.Frame(self, bg=BG)
        mb.pack(fill="x", padx=16, pady=8)

        def _mcard(parent, label, val, color):
            f = tk.Frame(parent, bg=SURFACE2)
            f.pack(side="left", padx=(0, 5))
            tk.Label(f, text=label, bg=SURFACE2, fg=TEXT3,
                     font=(_FF, 8)).pack(padx=10, pady=(5, 0))
            tk.Label(f, text=val, bg=SURFACE2, fg=color,
                     font=(_FM, 12, "bold")).pack(padx=10, pady=(0, 5))

        tk.Label(mb, text="TEMP", bg=BG, fg=TEXT3,
                 font=(_FF, 8, "bold")).pack(side="left", padx=(0, 5))
        _mcard(mb, "MAE",  f"{fv(m_t['mae'])} °C",  C_PRED)
        _mcard(mb, "RMSE", f"{fv(m_t['rmse'])} °C", C_PRED)
        _mcard(mb, "R²",   fv(m_t["r2"]),            C_TRUE)

        tk.Frame(mb, bg=SURFACE3, width=1).pack(side="left", fill="y",
                                                padx=10, pady=2)
        tk.Label(mb, text="PLUIE", bg=BG, fg=TEXT3,
                 font=(_FF, 8, "bold")).pack(side="left", padx=(0, 5))
        _mcard(mb, "F1",      f"{f1_val:.1%}",   C_TRUE)
        _mcard(mb, "Préc.",   f"{prec_val:.1%}", ACCENT)
        _mcard(mb, "Rappel",  f"{rec_val:.1%}",  "#D29922")

        _divider(self, BORDER)

        # ── Navigation ────────────────────────────────────────────────
        nav = tk.Frame(self, bg=SURFACE2)
        nav.pack(fill="x")

        # Date courante
        self.date_label = tk.Label(nav, text="", bg=SURFACE2, fg=TEXT,
                                   font=(_FF, 10, "bold"))
        self.date_label.pack(side="left", padx=14, pady=8)

        # Boutons navigation à droite
        for txt, cmd, bg, hv in [
            ("Pire ↓",     self._go_worst, RED,     RED_H),
            ("Meilleur ↑", self._go_best,  GREEN,   GREEN_H),
            ("▶",          self._next,     SURFACE3, BORDER),
            ("◀",          self._prev,     SURFACE3, BORDER),
        ]:
            Btn(nav, txt, cmd, bg=bg, fg="#FFFFFF" if bg != SURFACE3 else TEXT,
                hover=hv, padx=12, pady=5, font=_F9).pack(
                side="right", padx=4, pady=6)

        _divider(self, BORDER)

        # ── Slider ────────────────────────────────────────────────────
        sl = tk.Frame(self, bg=BG)
        sl.pack(fill="x", padx=16, pady=4)
        self.slider = ttk.Scale(
            sl, from_=0, to=len(self.sample_dates) - 1,
            orient="horizontal", command=self._on_slider)
        self.slider.pack(fill="x")

        # ── Barre de saut rapide ──────────────────────────────────────
        jump = tk.Frame(self, bg=BG)
        jump.pack(fill="x", padx=16, pady=(0, 6))

        for lbl, attr, cmd, w in [
            ("Aller au jour :", "jump_var",     self._jump_to_day, 13),
            ("Index :",         "jump_idx_var", self._jump_to_idx,  6),
        ]:
            tk.Label(jump, text=lbl, bg=BG, fg=TEXT3,
                     font=_F9).pack(side="left", padx=(0, 3))
            setattr(self, attr, tk.StringVar())
            e = tk.Entry(jump, textvariable=getattr(self, attr),
                         width=w, bg=SURFACE2, fg=TEXT,
                         insertbackground=TEXT, relief="flat", bd=0,
                         highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=BLUE,
                         font=_F9)
            e.pack(side="left", padx=(0, 2), ipady=4)
            Btn(jump, "Go", cmd, bg=BLUE, fg="#FFFFFF", hover=BLUE_H,
                padx=8, pady=4, font=_F9).pack(side="left", padx=(0, 16))

        _divider(self, BORDER)

        # ── Notebook ──────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_temp        = tk.Frame(nb, bg=BG)
        self.tab_rain        = tk.Frame(nb, bg=BG)
        self.tab_global      = tk.Frame(nb, bg=BG)
        self.tab_global_rain = tk.Frame(nb, bg=BG)
        nb.add(self.tab_temp,        text="  🌡  Température  ")
        nb.add(self.tab_rain,        text="  🌧  Pluie  ")
        nb.add(self.tab_global,      text="  📊  Vue globale  ")
        nb.add(self.tab_global_rain, text="  🌧  Vue globale pluie  ")

        # Temp figure
        self.fig_temp = Figure(figsize=(9, 4.5), dpi=100, facecolor=SURFACE)
        self.ax_temp  = self.fig_temp.add_subplot(111)
        _dark_ax(self.ax_temp)
        self.canvas_temp = FigureCanvasTkAgg(self.fig_temp, master=self.tab_temp)
        self.canvas_temp.get_tk_widget().pack(fill="both", expand=True)

        # Rain figure
        self.fig_rain = Figure(figsize=(9, 4.5), dpi=100, facecolor=SURFACE)
        self.ax_rain  = self.fig_rain.add_subplot(111)
        _dark_ax(self.ax_rain)
        self.canvas_rain = FigureCanvasTkAgg(self.fig_rain, master=self.tab_rain)
        self.canvas_rain.get_tk_widget().pack(fill="both", expand=True)

        self._init_sample_lines()

    def _init_sample_lines(self):
        hours = list(range(1, 25))
        _lg = dict(framealpha=0.2, facecolor=SURFACE2, edgecolor=BORDER,
                   labelcolor=TEXT2, fontsize=8)

        self.ax_temp.set_xlim(0, 25)
        self.ax_temp.set_xlabel("Heure h+1 … h+24")
        self.ax_temp.set_ylabel("T (°C)")
        (self.temp_pred_line,) = self.ax_temp.plot(
            hours, [0] * 24, color=C_PRED, linewidth=2, label="Prédiction")
        (self.temp_true_line,) = self.ax_temp.plot(
            hours, [0] * 24, color=C_TRUE, linewidth=2, label="Réel")
        self.ax_temp.fill_between(
            hours, [0] * 24, [0] * 24, alpha=0.12, color=C_FILL, label="Écart")
        self.ax_temp.legend(**_lg)
        self.ax_temp.grid(True, alpha=0.12, color=BORDER)
        self.fig_temp.tight_layout(pad=1.8)

        self.ax_rain.set_ylim(-0.1, 1.1)
        self.ax_rain.set_xlabel("Heure h+1 … h+24")
        self.ax_rain.set_ylabel("Probabilité de pluie")
        (self.rain_pred_line,) = self.ax_rain.plot(
            hours, [0] * 24, color=C_PRED, linewidth=2, label="Prédiction (proba)")
        (self.rain_true_line,) = self.ax_rain.plot(
            hours, [0] * 24, color=C_TRUE, linewidth=2.5, label="Réel (0/1)")
        self.ax_rain.axhline(0.5, color=TEXT3, linestyle="--",
                              linewidth=1, label="Seuil 0.5")
        self.ax_rain.legend(**_lg)
        self.ax_rain.grid(True, alpha=0.12, color=BORDER)
        self.fig_rain.tight_layout(pad=1.8)

    def _build_global_tab(self):
        """Onglet Vue globale : scatter temp + histogramme résidus + MAE par heure."""
        fig = Figure(figsize=(13, 5), dpi=100, facecolor=SURFACE)
        _lg = dict(framealpha=0.2, facecolor=SURFACE2, edgecolor=BORDER,
                   labelcolor=TEXT2, fontsize=8)

        y_true_flat = self.y_temp_true_real.ravel()
        y_pred_flat = self.y_temp_pred_real.ravel()
        m           = self.global_metrics_temp

        # --- Scatter pred vs true ---
        ax_s = fig.add_subplot(131)
        _dark_ax(ax_s)
        step = max(1, len(y_true_flat) // 3000)
        ax_s.scatter(y_true_flat[::step], y_pred_flat[::step],
                     s=5, alpha=0.4, color=ACCENT)
        lims = [min(y_true_flat.min(), y_pred_flat.min()) - 1,
                max(y_true_flat.max(), y_pred_flat.max()) + 1]
        ax_s.plot(lims, lims, color=RED_H, linestyle="--",
                  linewidth=1, label="y=x")
        ax_s.set_xlim(lims); ax_s.set_ylim(lims)
        ax_s.set_xlabel("T réelle (°C)")
        ax_s.set_ylabel("T prédite (°C)")
        ax_s.set_title("Prédit vs Réel")
        ax_s.legend(**_lg)
        ax_s.grid(True, alpha=0.1, color=BORDER)
        ax_s.text(0.05, 0.95,
                  f"R²  = {m['r2']:.3f}\nRMSE= {m['rmse']:.2f} °C",
                  transform=ax_s.transAxes, va="top", fontsize=8.5,
                  color=TEXT,
                  bbox=dict(boxstyle="round,pad=0.4",
                             facecolor=SURFACE2, edgecolor=BORDER, alpha=0.9))

        # --- Histogramme des résidus ---
        ax_h = fig.add_subplot(132)
        _dark_ax(ax_h)
        residuals = y_pred_flat - y_true_flat
        ax_h.hist(residuals, bins=60, color=ACCENT, alpha=0.7,
                  edgecolor=SURFACE)
        ax_h.axvline(0, color=RED_H,    linestyle="--", linewidth=1.3)
        ax_h.axvline(residuals.mean(), color="#D29922", linestyle="-",
                     linewidth=1.3,
                     label=f"Moy. = {residuals.mean():.2f} °C")
        ax_h.set_xlabel("Résidu (°C)")
        ax_h.set_ylabel("Fréquence")
        ax_h.set_title("Distribution des résidus")
        ax_h.legend(**_lg)
        ax_h.grid(True, alpha=0.1, color=BORDER, axis="y")

        # --- MAE par heure ---
        ax_hr = fig.add_subplot(133)
        _dark_ax(ax_hr)
        mae_per_hour = [
            float(np.mean(np.abs(
                self.y_temp_pred_real[:, h] - self.y_temp_true_real[:, h]
            )))
            for h in range(24)
        ]
        hours   = list(range(1, 25))
        max_mae = max(mae_per_hour) or 1
        bar_colors = []
        for v in mae_per_hour:
            t = v / max_mae
            # vert → rouge au fur et à mesure que l'erreur augmente
            r = int(0x3F + t * (0xF8 - 0x3F))
            g = int(0xB9 - t * (0xB9 - 0x51))
            b = int(0x50 - t * 0x50)
            bar_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        ax_hr.bar(hours, mae_per_hour, color=bar_colors, alpha=0.9)
        ax_hr.set_xlabel("Heure h+1 … h+24")
        ax_hr.set_ylabel("MAE (°C)")
        ax_hr.set_title("MAE par heure de prédiction")
        ax_hr.grid(True, alpha=0.1, color=BORDER, axis="y")

        fig.tight_layout(pad=2.2)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_global)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _build_global_rain_tab(self):
        """Onglet Vue globale pluie : confusion matrix + distribution probas + F1 par heure."""
        fig = Figure(figsize=(13, 5), dpi=100, facecolor=SURFACE)
        _lg = dict(framealpha=0.2, facecolor=SURFACE2, edgecolor=BORDER,
                   labelcolor=TEXT2, fontsize=8)

        y_pred_proba = self.y_rain_pred.ravel()
        y_true_bin   = self.y_rain_true.ravel().astype(int)
        y_pred_bin   = (y_pred_proba >= 0.5).astype(int)

        tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
        fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
        fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
        tn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 0)))

        # --- Matrice de confusion ---
        ax_cm = fig.add_subplot(131)
        _dark_ax(ax_cm)

        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-7)  # normalisation par ligne (recall)

        im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        labels = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                count = int(cm[i, j])
                pct   = cm_norm[i, j]
                color = "white" if pct > 0.5 else TEXT
                ax_cm.text(j, i, f"{labels[i][j]}\n{count}\n({pct:.1%})",
                           ha="center", va="center", fontsize=9,
                           color=color, fontweight="bold")

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Prédit sec", "Prédit pluie"], fontsize=8)
        ax_cm.set_yticklabels(["Réel sec", "Réel pluie"], fontsize=8)
        ax_cm.set_title("Matrice de confusion", color=TEXT)

        total = tp + fp + fn + tn
        acc   = (tp + tn) / (total + 1e-7)
        prec  = tp / (tp + fp + 1e-7)
        rec   = tp / (tp + fn + 1e-7)
        f1    = 2 * prec * rec / (prec + rec + 1e-7)
        ax_cm.text(0.5, -0.22,
                   f"Acc={acc:.1%}  F1={f1:.1%}  Préc={prec:.1%}  Rappel={rec:.1%}",
                   transform=ax_cm.transAxes, ha="center", fontsize=7.5,
                   color=TEXT2)

        # --- Distribution des probabilités prédites ---
        ax_d = fig.add_subplot(132)
        _dark_ax(ax_d)

        proba_sec   = y_pred_proba[y_true_bin == 0]
        proba_pluie = y_pred_proba[y_true_bin == 1]
        bins = np.linspace(0, 1, 30)

        ax_d.hist(proba_sec,   bins=bins, alpha=0.65, color=ACCENT,  label=f"Réel sec   (n={len(proba_sec)})",   density=True)
        ax_d.hist(proba_pluie, bins=bins, alpha=0.65, color=C_PRED,  label=f"Réel pluie (n={len(proba_pluie)})", density=True)
        ax_d.axvline(0.5, color=TEXT3, linestyle="--", linewidth=1.2, label="Seuil 0.5")
        ax_d.set_xlabel("Probabilité prédite")
        ax_d.set_ylabel("Densité")
        ax_d.set_title("Distribution des probabilités")
        ax_d.legend(**_lg)
        ax_d.grid(True, alpha=0.1, color=BORDER, axis="y")

        # --- F1 / Précision / Rappel par heure h+1 → h+24 ---
        ax_hr = fig.add_subplot(133)
        _dark_ax(ax_hr)

        hours  = list(range(1, 25))
        f1s, precs, recs = [], [], []
        for h in range(24):
            yt_h = self.y_rain_true[:, h].ravel().astype(int)
            yp_h = (self.y_rain_pred[:, h].ravel() >= 0.5).astype(int)
            _tp = int(np.sum((yp_h == 1) & (yt_h == 1)))
            _fp = int(np.sum((yp_h == 1) & (yt_h == 0)))
            _fn = int(np.sum((yp_h == 0) & (yt_h == 1)))
            _p  = _tp / (_tp + _fp + 1e-7)
            _r  = _tp / (_tp + _fn + 1e-7)
            _f1 = 2 * _p * _r / (_p + _r + 1e-7)
            f1s.append(_f1); precs.append(_p); recs.append(_r)

        ax_hr.plot(hours, f1s,   color=C_TRUE,  linewidth=2,   label="F1")
        ax_hr.plot(hours, precs, color=ACCENT,  linewidth=1.5, linestyle="--", label="Précision")
        ax_hr.plot(hours, recs,  color="#D29922", linewidth=1.5, linestyle=":",  label="Rappel")
        ax_hr.set_ylim(0, 1.05)
        ax_hr.set_xlabel("Heure h+1 … h+24")
        ax_hr.set_ylabel("Score")
        ax_hr.set_title("F1 / Précision / Rappel par heure")
        ax_hr.legend(**_lg)
        ax_hr.grid(True, alpha=0.1, color=BORDER)

        fig.tight_layout(pad=2.2)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_global_rain)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    # ---- Refresh current sample ----

    def refresh(self):
        self.idx = max(0, min(self.idx, len(self.sample_dates) - 1))
        date_str = self.sample_dates[self.idx]
        mae_here = float(self.per_sample_mae_temp[self.idx])
        self.date_label.config(text=f"Index {self.idx}/{len(self.sample_dates)-1} — {date_str}   (MAE locale: {mae_here:.2f}°C)")

        # Sync slider without triggering _on_slider callback
        self.slider.set(self.idx)

        hours = list(range(1, 25))

        # --- Température ---
        y_pred_t = np.asarray(self.y_temp_pred_real[self.idx], dtype=float)
        y_true_t = np.asarray(self.y_temp_true_real[self.idx], dtype=float)

        self.temp_pred_line.set_ydata(y_pred_t)
        self.temp_true_line.set_ydata(y_true_t)

        # Ombrage de l'écart entre pred et vrai
        for coll in self.ax_temp.collections:
            coll.remove()
        self.ax_temp.fill_between(hours, y_pred_t, y_true_t,
                                  alpha=0.15, color=C_FILL)

        y_min = float(np.nanmin([y_pred_t.min(), y_true_t.min()]))
        y_max = float(np.nanmax([y_pred_t.max(), y_true_t.max()]))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = -5.0, 35.0
        else:
            pad = max(1.0, 0.1 * (y_max - y_min))
            y_min -= pad
            y_max += pad
        self.ax_temp.set_ylim(y_min, y_max)

        m_t = compute_metrics(y_true_t, y_pred_t)
        self.ax_temp.set_title(
            f"Température — {date_str}   {fmt_metrics(m_t)}",
            color=TEXT, fontsize=8.5)
        self.canvas_temp.draw_idle()

        # --- Pluie ---
        y_pred_r = np.asarray(self.y_rain_pred[self.idx], dtype=float)
        y_true_r = np.asarray(self.y_rain_true[self.idx], dtype=float)

        self.rain_pred_line.set_ydata(y_pred_r)
        self.rain_true_line.set_ydata(y_true_r)

        m_r = compute_metrics(y_true_r, (y_pred_r > 0.5).astype(float))
        acc = float(np.mean((y_pred_r > 0.5).astype(float) == y_true_r))
        self.ax_rain.set_title(
            f"Pluie — {date_str}   Accuracy: {acc:.1%}",
            color=TEXT, fontsize=8.5)
        self.canvas_rain.draw_idle()

    # ---- Navigation ----

    def _prev(self):
        self.idx = max(0, self.idx - 1)
        self.refresh()

    def _next(self):
        self.idx = min(len(self.sample_dates) - 1, self.idx + 1)
        self.refresh()

    def _on_slider(self, val):
        new_idx = int(float(val))
        if new_idx != self.idx:
            self.idx = new_idx
            self.refresh()

    def _go_best(self):
        self.idx = int(np.argmin(self.per_sample_mae_temp))
        self.refresh()

    def _go_worst(self):
        self.idx = int(np.argmax(self.per_sample_mae_temp))
        self.refresh()

    def _jump_to_day(self):
        day_str = (self.jump_var.get() or "").strip()
        if not day_str:
            messagebox.showerror("Erreur", "Saisis un jour (YYYY-MM-DD).")
            return
        idx = find_idx_for_day(self.sample_dates, day_str)
        if idx is None:
            messagebox.showerror("Erreur", f"Jour introuvable : {day_str}")
            return
        self.idx = idx
        self.refresh()

    def _jump_to_idx(self):
        s = (self.jump_idx_var.get() or "").strip()
        if not s:
            messagebox.showerror("Erreur", "Saisis un index.")
            return
        try:
            idx = int(s)
        except ValueError:
            messagebox.showerror("Erreur", "Index invalide.")
            return
        if idx < 0 or idx >= len(self.sample_dates):
            messagebox.showerror("Erreur", f"Index hors limites. Max = {len(self.sample_dates)-1}")
            return
        self.idx = idx
        self.refresh()


# -------------------- Entry point -------------------- #

def main():
    lookback          = 48
    window_size       = 24
    rain_mm_threshold = 0.1  # seuil physique, identique à train.py

    # compile=False : on ne recharge pas la loss/métriques custom (WeightedBCE, F1Score)
    # car predict_gui.py n'a besoin que de model.predict(), pas d'entraîner ni d'évaluer.
    model = load_model(MODEL_PATH, compile=False)
    with open(NORM_PATH, "rb") as f:
        norm = pickle.load(f)

    if isinstance(norm, dict):
        mu_T  = norm["mu_T"];  sig_T = norm["sig_T"]
        mu_P  = norm["mu_P"];  sig_P = norm["sig_P"]
        mu_r  = norm["mu_r"];  sig_r = norm["sig_r"]
        mu_W  = norm.get("mu_W", 0.0); sig_W = norm.get("sig_W", 1.0)
        n_inputs_model = norm.get("n_inputs", 8)
    else:
        if len(norm) == 8:
            mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W = norm
            n_inputs_model = 12
        else:
            mu_T, sig_T, mu_P, sig_P, mu_r, sig_r = norm
            mu_W, sig_W = 0.0, 1.0
            n_inputs_model = 8

    if _cli.n_inputs is not None:
        n_inputs_model = _cli.n_inputs
        print(f"FEATURES (forcé CLI) : {n_inputs_model}")
    else:
        print(f"FEATURES : {n_inputs_model}")

    test_chunks, test_raw_rain, date_chunks = get_norm_data_and_dates_from_file(
        TEST_CSV, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W
    )
    X_test, Y_temp_test, Y_rain_test, sample_dates = build_xy_and_sample_dates(
        test_chunks, test_raw_rain, date_chunks,
        lookback=lookback, window_size=window_size, rain_mm_threshold=rain_mm_threshold
    )

    if len(X_test) == 0:
        raise RuntimeError("Impossible de construire X_test: test_data.csv trop court (il faut lookback + window_size).")

    X_test = X_test[:, :, :n_inputs_model]
    print(f"X_test shape après slicing : {X_test.shape}")

    Y_temp_pred, Y_rain_pred = model.predict(X_test, verbose=0)
    Y_temp_pred_real = denormalize_temp(Y_temp_pred, mu_T, sig_T)
    Y_temp_test_real = denormalize_temp(Y_temp_test, mu_T, sig_T)

    # Métriques globales calculées une seule fois
    global_metrics_temp = compute_metrics(Y_temp_test_real.ravel(), Y_temp_pred_real.ravel())
    global_metrics_rain = compute_metrics(Y_rain_test.ravel(), (Y_rain_pred > 0.5).astype(float).ravel())
    per_sample_mae_temp = compute_per_sample_mae(Y_temp_test_real, Y_temp_pred_real)

    root = tk.Tk()
    root.withdraw()
    _setup_style(root)
    dialog = ForecastDayDialog(root, "Choisir le point de départ")

    if dialog.result is None:
        return

    day_str, idx_str = dialog.result

    if idx_str:
        try:
            idx = int(idx_str)
        except ValueError:
            messagebox.showerror("Erreur", "Index invalide.")
            return
    else:
        if not day_str:
            idx = 0
        else:
            idx = find_idx_for_day(sample_dates, day_str)
            if idx is None:
                messagebox.showerror("Erreur", f"Jour introuvable : {day_str}")
                return

    if idx < 0 or idx >= len(X_test):
        messagebox.showerror("Erreur", f"Index hors limites. Max = {len(X_test)-1}")
        return

    nav = ForecastNavigator(
        root,
        idx_init=idx,
        sample_dates=sample_dates,
        y_temp_pred_real=Y_temp_pred_real,
        y_temp_true_real=Y_temp_test_real,
        y_rain_pred=Y_rain_pred,
        y_rain_true=Y_rain_test,
        model=model,
        x_test=X_test,
        y_temp_test=Y_temp_test,
        y_rain_test=Y_rain_test,
        global_metrics_temp=global_metrics_temp,
        global_metrics_rain=global_metrics_rain,
        per_sample_mae_temp=per_sample_mae_temp,
    )
    nav.grab_set()
    root.mainloop()


if __name__ == "__main__":
    main()
