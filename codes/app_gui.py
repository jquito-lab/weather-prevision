# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

import data
from data import download_infoclimat_range
from parser import clean_csv_72h


PROJECT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOWNLOAD_DIR  = os.path.join(PROJECT_DIR, "downloads")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CODES_DIR      = os.path.abspath(os.path.dirname(__file__))
TRAIN_SCRIPT   = os.path.join(CODES_DIR, "train.py")
PREDICT_SCRIPT = os.path.join(CODES_DIR, "predict_gui.py")
PYTHON_EXE     = sys.executable

PRESET_MODELS = [
    {
        "label": "12 entrées  —  weather_model.keras",
        "model": os.path.join(ARTIFACTS_DIR, "weather_model.keras"),
        "norm":  os.path.join(ARTIFACTS_DIR, "norm_params.pkl"),
        "extra": [],
    },
    {
        "label": "8 entrées  —  8_inputs_model_2015-2026.keras",
        "model": os.path.join(ARTIFACTS_DIR, "8_inputs_model_2015-2026.keras"),
        "norm":  os.path.join(ARTIFACTS_DIR, "norm_params.pkl"),
        "extra": ["--n-inputs", "8"],
    },
]

# ── Palette ───────────────────────────────────────────────────────────── #
BG       = "#0D1117"
SURFACE  = "#161B22"
SURFACE2 = "#21262D"
SURFACE3 = "#30363D"
BORDER   = "#30363D"
PRIMARY  = "#238636"   # GitHub green
PRIMARY_H= "#2EA043"
BLUE     = "#1F6FEB"
BLUE_H   = "#388BFD"
AMBER    = "#BB8009"
AMBER_H  = "#D29922"
RED      = "#DA3633"
RED_H    = "#F85149"
TEXT     = "#E6EDF3"
TEXT2    = "#8B949E"
TEXT3    = "#6E7681"
ACCENT   = "#58A6FF"   # blue link

_FF  = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"
_FM  = "SF Mono"        if sys.platform == "darwin" else "Consolas"
_F10 = (_FF, 10)
_F9  = (_FF, 9)
_F11B = (_FF, 11, "bold")
_FM9  = (_FM, 9)


def parse_date(s: str):
    s = s.strip().replace("/", "-")
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── Widgets réutilisables ─────────────────────────────────────────────── #

class Btn(tk.Label):
    """Bouton plat avec effet hover."""
    def __init__(self, parent, text, command,
                 bg=SURFACE3, fg=TEXT, hover=None,
                 padx=16, pady=7, font=None, **kw):
        super().__init__(
            parent, text=text, bg=bg, fg=fg,
            cursor="hand2", padx=padx, pady=pady,
            font=font or _F10, **kw)
        self._bg    = bg
        self._hover = hover or SURFACE3
        self._cmd   = command
        self.bind("<Enter>",    lambda e: self.config(bg=self._hover))
        self.bind("<Leave>",    lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._cmd())


def _field(parent, label_text, var, show=None):
    """Label + Entry empilés verticalement dans parent."""
    tk.Label(parent, text=label_text, bg=SURFACE, fg=TEXT2,
             font=_F9, anchor="w").pack(fill="x", padx=16, pady=(8, 2))
    e = tk.Entry(parent, textvariable=var,
                 bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, font=_F10,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BLUE)
    if show:
        e.config(show=show)
    e.pack(fill="x", padx=16, ipady=6)
    return e


def _divider(parent, color=BORDER):
    tk.Frame(parent, bg=color, height=1).pack(fill="x")


class Card(tk.Frame):
    """Carte avec bande colorée sur la gauche."""
    def __init__(self, parent, *, num, color, title, desc, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.pack(fill="x", padx=16, pady=(0, 10))

        # Ligne de titre
        top = tk.Frame(self, bg=SURFACE)
        top.pack(fill="x", padx=0, pady=0)

        # Bande colorée
        tk.Frame(top, bg=color, width=3).pack(side="left", fill="y")

        # Badge numéroté
        tk.Label(top, text=f" {num} ", bg=color, fg="#FFFFFF",
                 font=(_FF, 10, "bold")).pack(side="left", padx=(0, 0), pady=10)

        # Titre + description
        tk.Label(top, text=f"  {title}", bg=SURFACE, fg=TEXT,
                 font=_F11B).pack(side="left", pady=10)
        tk.Label(top, text=desc, bg=SURFACE, fg=TEXT2,
                 font=_F9).pack(side="right", padx=16, pady=10)

        _divider(self, SURFACE3)

        # Zone de contenu retournée
        self.body = tk.Frame(self, bg=SURFACE)
        self.body.pack(fill="x", pady=(4, 12))


# ── Sélecteur de modèle ───────────────────────────────────────────────── #

class ModelPickerDialog(tk.Toplevel):
    """Dialog modal pour choisir le modèle et le fichier CSV de test."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Choisir un modèle")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self.result = None  # (model_path, norm_path) ou None si annulé

        self._choice = tk.IntVar(value=0)
        self._custom_model = tk.StringVar()
        self._custom_norm  = tk.StringVar()

        self._build()
        self.update_idletasks()
        # centrage sur le parent
        px = parent.winfo_rootx() + parent.winfo_width()  // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"+{px - 230}+{py - 180}")
        self.wait_window()

    def _build(self):
        # ── Titre ──
        hdr = tk.Frame(self, bg=SURFACE)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Choisir un modèle de prédiction",
                 bg=SURFACE, fg=TEXT, font=(_FF, 11, "bold")).pack(
                 side="left", padx=12, pady=10)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="x", padx=20, pady=12)

        # ── Presets ──
        tk.Label(body, text="Modèle prédéfini", bg=BG, fg=TEXT2,
                 font=(_FF, 9, "bold")).pack(anchor="w", pady=(0, 6))

        for i, p in enumerate(PRESET_MODELS):
            exists = os.path.exists(p["model"])
            color  = TEXT if exists else TEXT3
            note   = "" if exists else "  (non entraîné)"
            rb = tk.Radiobutton(
                body, text=p["label"] + note,
                variable=self._choice, value=i,
                bg=BG, fg=color, selectcolor=SURFACE2,
                activebackground=BG, activeforeground=TEXT,
                font=_F10, state="normal" if exists else "disabled",
                command=self._on_choice)
            rb.pack(anchor="w", pady=2)

        # Séparateur
        tk.Frame(body, bg=SURFACE3, height=1).pack(fill="x", pady=10)

        # ── Personnalisé ──
        tk.Radiobutton(
            body, text="Modèle personnalisé",
            variable=self._choice, value=len(PRESET_MODELS),
            bg=BG, fg=TEXT, selectcolor=SURFACE2,
            activebackground=BG, activeforeground=TEXT,
            font=_F10, command=self._on_choice).pack(anchor="w")

        custom_grid = tk.Frame(body, bg=BG)
        custom_grid.pack(fill="x", pady=(6, 0))

        for row, (lbl, var, btn_cmd) in enumerate([
            ("Modèle (.keras)", self._custom_model, self._browse_model),
            ("Norm params (.pkl)", self._custom_norm,  self._browse_norm),
        ]):
            tk.Label(custom_grid, text=lbl, bg=BG, fg=TEXT3,
                     font=_F9, width=18, anchor="w").grid(
                     row=row, column=0, sticky="w", pady=3)
            tk.Entry(custom_grid, textvariable=var,
                     bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=0, font=_F9, width=28,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=BLUE).grid(
                     row=row, column=1, padx=(4, 4), ipady=4)
            Btn(custom_grid, "…", btn_cmd,
                bg=SURFACE3, fg=TEXT, hover=SURFACE2,
                padx=6, pady=3, font=_F9).grid(row=row, column=2)

        self._custom_frame = custom_grid
        self._refresh_custom_state()

        # ── Boutons ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(8, 0))
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=20, pady=10)
        Btn(btn_row, "Annuler", self.destroy,
            bg=SURFACE3, fg=TEXT, hover=SURFACE2,
            padx=12, pady=6, font=_F10).pack(side="right", padx=(6, 0))
        Btn(btn_row, "▶  Prédire", self._confirm,
            bg=AMBER, fg="#FFFFFF", hover=AMBER_H,
            padx=12, pady=6, font=_F10).pack(side="right")

    def _on_choice(self):
        self._refresh_custom_state()

    def _refresh_custom_state(self):
        is_custom = (self._choice.get() == len(PRESET_MODELS))
        state = "normal" if is_custom else "disabled"
        for w in self._custom_frame.winfo_children():
            try:
                w.config(state=state)
            except tk.TclError:
                pass

    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Choisir le modèle (.keras)",
            initialdir=ARTIFACTS_DIR,
            filetypes=[("Keras model", "*.keras"), ("Tous", "*.*")])
        if p:
            self._custom_model.set(p)

    def _browse_norm(self):
        p = filedialog.askopenfilename(
            title="Choisir les paramètres de normalisation (.pkl)",
            initialdir=ARTIFACTS_DIR,
            filetypes=[("Pickle", "*.pkl"), ("Tous", "*.*")])
        if p:
            self._custom_norm.set(p)

    def _confirm(self):
        choice = self._choice.get()
        if choice < len(PRESET_MODELS):
            p = PRESET_MODELS[choice]
            self.result = (p["model"], p["norm"], p.get("extra", []))
        else:
            model = self._custom_model.get().strip()
            norm  = self._custom_norm.get().strip()
            if not model or not os.path.exists(model):
                messagebox.showerror("Erreur", "Modèle introuvable.", parent=self)
                return
            if not norm or not os.path.exists(norm):
                messagebox.showerror("Erreur", "Fichier .pkl introuvable.", parent=self)
                return
            self.result = (model, norm, [])
        self.destroy()


# ── App principale ────────────────────────────────────────────────────── #

class App(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, bg=BG)
        self.master = master
        self.pack(fill="both", expand=True)

        master.title("Weather Prevision")
        master.configure(bg=BG)
        master.resizable(True, True)

        self.start_var     = tk.StringVar(value="2026-01-01")
        self.end_var       = tk.StringVar(value="2026-01-10")
        self.min_hours_var = tk.StringVar(value="72")
        self.api_token_var = tk.StringVar(value=os.getenv("INFOCLIMAT_TOKEN", ""))
        self._download_dir = DOWNLOAD_DIR
        self.downloaded_path = None
        self.cleaned_path    = None

        self._build_header()
        self._build_pipeline()
        self._build_log()

    # ── Header ──────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=SURFACE)
        hdr.pack(fill="x")
        _divider(hdr, BORDER)

        inner = tk.Frame(hdr, bg=SURFACE)
        inner.pack(fill="x", padx=20, pady=12)

        # Logo + titre
        tk.Label(inner, text="⛅", bg=SURFACE, fg=ACCENT,
                 font=(_FF, 20)).pack(side="left")
        tk.Label(inner, text=" Weather Prevision", bg=SURFACE, fg=TEXT,
                 font=(_FF, 17, "bold")).pack(side="left")
        tk.Label(inner, text="  —  Pipeline de données météo",
                 bg=SURFACE, fg=TEXT2, font=_F9).pack(side="left")

        # Dot de statut
        self._status_lbl = tk.Label(inner, text="● Prêt",
                                    bg=SURFACE, fg=TEXT3, font=_F9)
        self._status_lbl.pack(side="right")

        _divider(self, BORDER)

    # ── Pipeline (3 cartes) ──────────────────────────────────────────────

    def _build_pipeline(self):
        # Conteneur scrollable
        wrapper = tk.Frame(self, bg=BG)
        wrapper.pack(fill="both", expand=False, pady=10)

        # ── Carte 1 — Collecte ───────────────────────────────────────────
        c1 = Card(wrapper, num="1", color=BLUE, title="Collecte des données",
                  desc="Téléchargement via API InfoClimat")

        _field(c1.body, "Date de début  (YYYY-MM-DD)", self.start_var)
        _field(c1.body, "Date de fin    (YYYY-MM-DD)", self.end_var)
        _field(c1.body, "Clé API InfoClimat", self.api_token_var, show="*")

        # Sélection dossier
        dir_row = tk.Frame(c1.body, bg=SURFACE)
        dir_row.pack(fill="x", padx=16, pady=(8, 0))
        Btn(dir_row, "📁  Choisir dossier", self.choose_download_dir,
            bg=SURFACE2, hover=SURFACE3, font=_F9).pack(side="left")
        self.download_dir_label = tk.Label(
            dir_row, text=self._download_dir,
            bg=SURFACE, fg=ACCENT, font=(_FM, 8),
            wraplength=480, justify="left")
        self.download_dir_label.pack(side="left", padx=(10, 0))

        # Bouton principal
        btn_row1 = tk.Frame(c1.body, bg=SURFACE)
        btn_row1.pack(fill="x", padx=16, pady=(10, 0))
        Btn(btn_row1, "⬇  Télécharger", self.on_download,
            bg=BLUE, fg="#FFFFFF", hover=BLUE_H, font=_F10).pack(side="left")

        # ── Carte 2 — Nettoyage ──────────────────────────────────────────
        c2 = Card(wrapper, num="2", color=PRIMARY, title="Nettoyage",
                  desc="Découpage des séquences consécutives ≥ N heures")

        _field(c2.body, "Min heures consécutives", self.min_hours_var)

        btn_row2 = tk.Frame(c2.body, bg=SURFACE)
        btn_row2.pack(fill="x", padx=16, pady=(10, 0))
        Btn(btn_row2, "✓  Nettoyer (parser)", self.on_clean,
            bg=PRIMARY, fg="#FFFFFF", hover=PRIMARY_H, font=_F10).pack(side="left")

        # ── Carte 3 — Modèle ─────────────────────────────────────────────
        c3 = Card(wrapper, num="3", color=AMBER, title="Modèle LSTM",
                  desc="Entraînement ou prédiction sur données de test")

        btn_row3 = tk.Frame(c3.body, bg=SURFACE)
        btn_row3.pack(fill="x", padx=16, pady=(6, 0))
        Btn(btn_row3, "⚙  Entraîner", self.on_train,
            bg=AMBER, fg="#FFFFFF", hover=AMBER_H, font=_F10).pack(side="left")
        tk.Frame(btn_row3, bg=SURFACE, width=8).pack(side="left")
        Btn(btn_row3, "▶  Prédire", self.on_predict,
            bg=SURFACE3, fg=TEXT, hover=SURFACE2, font=_F10).pack(side="left")

    # ── Zone de log ──────────────────────────────────────────────────────

    def _build_log(self):
        _divider(self, BORDER)

        log_hdr = tk.Frame(self, bg=SURFACE)
        log_hdr.pack(fill="x")
        tk.Label(log_hdr, text="  LOG", bg=SURFACE, fg=TEXT2,
                 font=(_FF, 9, "bold")).pack(side="left", pady=6)
        Btn(log_hdr, "Effacer", self._clear_log,
            bg=SURFACE, fg=TEXT3, hover=SURFACE2,
            padx=8, pady=4, font=_F9).pack(side="right", padx=6, pady=4)

        _divider(self, BORDER)

        txt_frame = tk.Frame(self, bg=BG)
        txt_frame.pack(fill="both", expand=True)

        sb = tk.Scrollbar(txt_frame, bg=SURFACE2,
                          troughcolor=BG, relief="flat", width=10)
        sb.pack(side="right", fill="y")

        self.log = tk.Text(
            txt_frame,
            bg="#010409", fg=TEXT, font=_FM9,
            insertbackground=TEXT,
            selectbackground=SURFACE3,
            relief="flat", bd=0,
            padx=14, pady=10,
            yscrollcommand=sb.set,
            state="disabled",
            height=10,
        )
        self.log.pack(fill="both", expand=True)
        sb.config(command=self.log.yview)

        self.log.tag_config("ts",      foreground=TEXT3)
        self.log.tag_config("info",    foreground=ACCENT)
        self.log.tag_config("success", foreground="#3FB950")
        self.log.tag_config("error",   foreground=RED_H)
        self.log.tag_config("warn",    foreground="#D29922")
        self.log.tag_config("cmd",     foreground="#A5D6FF")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def append_log(self, txt: str, level: str = "info"):
        def _insert():
            self.log.config(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            self.log.insert("end", f" {ts}  ", "ts")
            self.log.insert("end", txt + "\n", level)
            self.log.see("end")
            self.log.config(state="disabled")
        self.log.after(0, _insert)

    def _set_status(self, text: str, color: str = TEXT3):
        self._status_lbl.config(text=f"● {text}", fg=color)

    # ── Actions ──────────────────────────────────────────────────────────

    def choose_download_dir(self):
        d = filedialog.askdirectory(
            title="Choisir un dossier", initialdir=self._download_dir)
        if d:
            self._download_dir = d
            self.download_dir_label.config(text=d)

    def ensure_api_token(self) -> bool:
        token = (self.api_token_var.get() or "").strip()
        if not token:
            token = simpledialog.askstring(
                "Clé API InfoClimat",
                "Entrez votre clé API InfoClimat :",
                show="*", parent=self.master)
            token = (token or "").strip()
            if token:
                self.api_token_var.set(token)
        if not token:
            messagebox.showerror("Erreur",
                "Clé API manquante (INFOCLIMAT_TOKEN).")
            return False
        os.environ["INFOCLIMAT_TOKEN"] = token
        data.API_TOKEN = token
        return True

    def on_download(self):
        try:
            start_date = parse_date(self.start_var.get())
            end_date   = parse_date(self.end_var.get())
            if end_date < start_date:
                raise ValueError("Date de fin avant date de début.")
        except Exception as e:
            messagebox.showerror("Dates invalides", str(e))
            return
        if not self.ensure_api_token():
            return

        filename = (f"observations_"
                    f"{start_date.isoformat()}_{end_date.isoformat()}.csv")
        try:
            self._set_status("Téléchargement…", AMBER_H)
            self.append_log(f"Téléchargement {start_date} → {end_date}", "info")
            csv_path, _ = download_infoclimat_range(
                start_date, end_date,
                output_dir=self._download_dir, filename=filename)
            self.downloaded_path = csv_path
            self.append_log(f"Fichier sauvegardé : {csv_path}", "success")
            self._set_status("Téléchargement OK", "#3FB950")
            messagebox.showinfo("Téléchargement OK", f"Fichier :\n{csv_path}")
        except Exception as e:
            self._set_status("Erreur", RED_H)
            self.append_log(str(e), "error")
            messagebox.showerror("Erreur téléchargement", str(e))

    def on_clean(self):
        if not self.downloaded_path or not os.path.exists(self.downloaded_path):
            messagebox.showwarning("Étape manquante",
                "Télécharge d'abord un fichier CSV (étape 1).")
            return
        try:
            min_hours = int(self.min_hours_var.get().strip())
        except ValueError:
            messagebox.showerror("Erreur", "min_hours doit être un entier.")
            return
        try:
            self._set_status("Nettoyage…", AMBER_H)
            self.append_log(f"Parser — min_hours={min_hours}", "info")
            out_path, report = clean_csv_72h(
                self.downloaded_path, min_hours=min_hours)
            self.cleaned_path = out_path
            self.append_log(report, "success")
            self._set_status("Nettoyage OK", "#3FB950")
            messagebox.showinfo("Parser OK", report)
        except Exception as e:
            self._set_status("Erreur", RED_H)
            self.append_log(str(e), "error")
            messagebox.showerror("Erreur parser", str(e))

    def run_script_async(self, script_path: str, title: str, extra_args: list = None):
        if not os.path.exists(script_path):
            messagebox.showerror("Script introuvable", script_path)
            return

        cmd = [PYTHON_EXE, script_path] + (extra_args or [])

        def _worker():
            self.log.after(0, self._set_status, f"{title}…", AMBER_H)
            self.append_log(f"─── {title} ───", "cmd")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=CODES_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    s = line.rstrip("\n")
                    lvl = ("error" if any(k in s.lower()
                            for k in ("error", "erreur", "traceback"))
                           else "warn" if "warning" in s.lower()
                           else "info")
                    self.append_log(s, lvl)
                rc = proc.wait()
                lvl = "success" if rc == 0 else "error"
                self.append_log(f"─── {title} terminé (exit {rc}) ───", lvl)
                self.log.after(0, self._set_status,
                               f"{title} {'OK' if rc == 0 else 'ERREUR'}",
                               "#3FB950" if rc == 0 else RED_H)
            except Exception as e:
                self.append_log(f"Impossible de lancer {title}: {e}", "error")
                self.log.after(0, self._set_status, "Erreur", RED_H)

        threading.Thread(target=_worker, daemon=True).start()

    def on_train(self):
        self.run_script_async(TRAIN_SCRIPT, "TRAIN")

    def on_predict(self):
        dlg = ModelPickerDialog(self.master)
        if dlg.result is None:
            return
        model_path, norm_path, extra = dlg.result
        model_name = os.path.basename(model_path)
        self.append_log(f"Modèle sélectionné : {model_name}", "info")
        self.run_script_async(
            PREDICT_SCRIPT, "PREDICT",
            extra_args=["--model", model_path, "--norm", norm_path] + extra
        )


# ── Entry point ───────────────────────────────────────────────────────── #

def main():
    root = tk.Tk()
    root.geometry("780x780")
    root.minsize(640, 600)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
