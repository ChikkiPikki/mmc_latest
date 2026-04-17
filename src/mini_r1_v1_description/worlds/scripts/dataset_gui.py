#!/usr/bin/env python3
"""
dataset_gui.py — Tkinter front-end for the dataset capture pipeline.

- Scans worlds/symbols/*.png and shows a checkbox per symbol.
- "Import…" lets you pick any PNG from disk; it is center-cropped to a
  square and resized to SYMBOL_SIZE×SYMBOL_SIZE so all symbols stay
  uniform, then saved as symbols/symbol_<next_id>.png.
- Lets you set # texture combos, toggle negatives + count, toggle clear.
- "Generate Dataset" shells out to run_dataset_capture.sh with the right
  flags and streams output into the log pane.

Run inside the container (X11 is forwarded by dev.sh):
  python3 src/mini_r1_v1_description/worlds/scripts/dataset_gui.py
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2


SYMBOL_SIZE = 512
HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.abspath(os.path.join(HERE, ".."))
SYMBOLS_DIR = os.path.join(WORLDS_DIR, "symbols")
CAPTURE_SH = os.path.join(HERE, "run_dataset_capture.sh")
DEFAULT_OUTPUT = os.path.expanduser("~/dataset")


SYMBOL_RE = re.compile(r"^symbol_(\d+)\.png$")


def list_symbols():
    if not os.path.isdir(SYMBOLS_DIR):
        return []
    ids = []
    for f in os.listdir(SYMBOLS_DIR):
        m = SYMBOL_RE.match(f)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def next_symbol_id():
    existing = list_symbols()
    return (max(existing) + 1) if existing else 1


def center_crop_resize(src_path, size=SYMBOL_SIZE):
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {src_path}")
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return resized


class App:
    def __init__(self, root):
        self.root = root
        root.title("Mini R1 — Dataset Capture")
        root.geometry("760x700")

        self.symbol_vars = {}

        self._build_ui()
        self._refresh_symbols()

    # ── UI layout ─────────────────────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        top = ttk.Frame(self.root); top.pack(fill="x", **pad)
        ttk.Label(top, text="Symbols to capture:",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        self.symbol_frame = ttk.Frame(self.root)
        self.symbol_frame.pack(fill="x", padx=20)

        btns = ttk.Frame(self.root); btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Import symbol…",
                   command=self._import_symbol).pack(side="left")
        ttk.Button(btns, text="Refresh",
                   command=self._refresh_symbols).pack(side="left", padx=6)
        ttk.Button(btns, text="Select all",
                   command=lambda: self._set_all(True)).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear all",
                   command=lambda: self._set_all(False)).pack(side="left", padx=6)

        sep = ttk.Separator(self.root, orient="horizontal"); sep.pack(fill="x", pady=6)

        opts = ttk.Frame(self.root); opts.pack(fill="x", **pad)

        row = 0
        ttk.Label(opts, text="# texture combos:").grid(row=row, column=0, sticky="w")
        self.combos_var = tk.IntVar(value=5)
        ttk.Spinbox(opts, from_=1, to=20, width=6,
                    textvariable=self.combos_var).grid(row=row, column=1, sticky="w")

        row += 1
        self.negatives_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Capture wall-only negatives",
                        variable=self.negatives_var).grid(row=row, column=0, sticky="w")
        self.neg_count_var = tk.IntVar(value=40)
        ttk.Spinbox(opts, from_=10, to=500, width=6,
                    textvariable=self.neg_count_var).grid(row=row, column=1, sticky="w")
        ttk.Label(opts, text="positions").grid(row=row, column=2, sticky="w")

        row += 1
        self.obstacles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Inject random obstacles each combo",
                        variable=self.obstacles_var).grid(row=row, column=0, sticky="w")
        self.obs_count_var = tk.IntVar(value=6)
        ttk.Spinbox(opts, from_=0, to=30, width=6,
                    textvariable=self.obs_count_var).grid(row=row, column=1, sticky="w")
        ttk.Label(opts, text="obstacles").grid(row=row, column=2, sticky="w")

        row += 1
        self.target_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Cap images per symbol (stop when reached)",
                        variable=self.target_enabled_var).grid(row=row, column=0, sticky="w")
        self.target_images_var = tk.IntVar(value=400)
        ttk.Spinbox(opts, from_=50, to=10000, width=6, increment=50,
                    textvariable=self.target_images_var).grid(row=row, column=1, sticky="w")
        ttk.Label(opts, text="images").grid(row=row, column=2, sticky="w")

        row += 1
        self.clear_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Delete previous dataset files before running",
                        variable=self.clear_var).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        self.headless_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Headless (no Gazebo GUI / RViz — faster)",
                        variable=self.headless_var).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        ttk.Label(opts, text="Output dir:").grid(row=row, column=0, sticky="w")
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT)
        ttk.Entry(opts, textvariable=self.output_var, width=40).grid(
            row=row, column=1, columnspan=2, sticky="we")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=6)

        run_row = ttk.Frame(self.root); run_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_row, text="Generate Dataset",
                                  command=self._on_generate)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(run_row, text="Stop",
                                   command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(run_row, textvariable=self.status_var).pack(side="left", padx=12)

        prog_row = ttk.Frame(self.root); prog_row.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_label_var = tk.StringVar(value="")
        ttk.Label(prog_row, textvariable=self.progress_label_var, width=38,
                  anchor="w").pack(side="left", padx=8)

        log_frame = ttk.Frame(self.root); log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=18, bg="#1e1e1e", fg="#d4d4d4",
                           insertbackground="#d4d4d4", font=("monospace", 9))
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self.proc = None

    def _refresh_symbols(self):
        for w in self.symbol_frame.winfo_children():
            w.destroy()
        self.symbol_vars = {}

        ids = list_symbols()
        if not ids:
            ttk.Label(self.symbol_frame,
                      text="(no symbols in worlds/symbols/ — import one)").pack(anchor="w")
            return

        cols = 4
        for i, sid in enumerate(ids):
            # Default OFF — user must explicitly pick. Prevents the case
            # where "selecting 3" actually leaves all 7 checked and the
            # subset filter falls back to "all".
            var = tk.BooleanVar(value=False)
            self.symbol_vars[sid] = var
            cb = ttk.Checkbutton(self.symbol_frame, text=f"symbol_{sid}",
                                 variable=var)
            cb.grid(row=i // cols, column=i % cols, sticky="w", padx=4, pady=2)

    def _set_all(self, on):
        for v in self.symbol_vars.values():
            v.set(on)

    def _import_symbol(self):
        path = filedialog.askopenfilename(
            title="Select a PNG/JPG to import as a new symbol",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            img = center_crop_resize(path)
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return
        os.makedirs(SYMBOLS_DIR, exist_ok=True)
        new_id = next_symbol_id()
        dest = os.path.join(SYMBOLS_DIR, f"symbol_{new_id}.png")
        cv2.imwrite(dest, img)
        self._log(f"Imported {path} → {dest} ({SYMBOL_SIZE}x{SYMBOL_SIZE})\n")
        self._refresh_symbols()

    # ── Run pipeline ──────────────────────────────────────────────────
    def _selected_ids(self):
        return [sid for sid, v in self.symbol_vars.items() if v.get()]

    def _on_generate(self):
        ids = self._selected_ids()
        if not ids:
            messagebox.showwarning("Pick a symbol",
                                   "Select at least one symbol (or import one first).")
            return

        symbol_arg = str(ids[0]) if len(ids) == 1 else ",".join(str(i) for i in ids)
        if set(ids) == set(list_symbols()):
            symbol_arg = "all"

        cmd = ["bash", CAPTURE_SH, symbol_arg,
               "--output", self.output_var.get(),
               "--combos", str(self.combos_var.get())]
        if self.negatives_var.get():
            cmd.append(f"--negatives:{self.neg_count_var.get()}")
        if self.obstacles_var.get() and self.obs_count_var.get() > 0:
            cmd.extend(["--obstacles", str(self.obs_count_var.get())])
        if self.target_enabled_var.get() and self.target_images_var.get() > 0:
            cmd.extend(["--target-images", str(self.target_images_var.get())])
        if self.clear_var.get():
            cmd.append("--clear")
        if not self.headless_var.get():
            cmd.append("--no-headless")

        self._log(f"\n$ {' '.join(cmd)}\n")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running…")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in self.proc.stdout:
                    self.root.after(0, self._log, line)
                rc = self.proc.wait()
                self.root.after(0, self._log, f"\n[exit {rc}]\n")
            except Exception as e:
                self.root.after(0, self._log, f"\n[ERROR] {e}\n")
            finally:
                self.proc = None
                self.root.after(0, self._done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self):
        if self.proc is None:
            return
        self._log("\n[stopping…]\n")
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _done(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Idle.")
        self.progress["value"] = 100
        self.progress_label_var.set("Done.")

    def _log(self, text):
        self.log.insert("end", text)
        self.log.see("end")
        self._parse_progress(text)

    _PROG_RE = re.compile(
        r"\[progress\] (\S+) (\d+)/(\d+) \((\d+)%\)\s+images=(\d+)\s+elapsed=(\S+)\s+ETA=(\S+)"
    )

    def _parse_progress(self, line):
        m = self._PROG_RE.search(line)
        if not m:
            return
        stage, done, total, pct, imgs, elapsed, eta = m.groups()
        self.progress["value"] = int(pct)
        self.progress_label_var.set(
            f"{stage} {done}/{total}  {imgs} imgs  elapsed {elapsed}  ETA {eta}"
        )


def main():
    if not os.path.exists(CAPTURE_SH):
        print(f"ERROR: {CAPTURE_SH} not found", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
