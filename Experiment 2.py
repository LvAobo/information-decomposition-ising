#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ising_pid_sim.py  ——  Ising model + PID analysis batch processing script
===============================================================
Unified with Experiment 1 environment:
- Glauber dynamics (random-order sweep visiting each site once per sweep)
- Parameters match Experiment 1 exactly
- PID tasks: adjacent source spins + coarse-grained |mean| in local windows / global as target
- Writes outputs to a timestamped folder (figures, Excel, PDF report)
"""

from __future__ import annotations

import time, datetime
from collections import Counter
from math import log2
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  16,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
})

# -----------------------------------------------------------------------------
# 0 ◇ Global tunable parameters (MATCH Experiment 1)
# -----------------------------------------------------------------------------
CONFIG: dict = {
    'L': 128,                 # lattice size
    'J': 1.0,                 # coupling
    'PREHEAT_STEPS': 20_000,  # burn-in
    'N_FRAMES':    80_000,    # recorded sweeps (every step)
    'T_MIN': 2.00,
    'T_MAX': 2.80,
    'N_T':   50,              # temperature points
    'N_POINTS': 50,           # observation pairs per task
    'OBS_SEED': 6463,
    'RNG_SEED': 635546,
    'WITH_SHIFT': False,      # kept for compatibility (unused here)
}

# —— Create output directory based on timestamp ——
_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = Path(f"ising_pid_out_{_ts}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG['OUT_DIR'] = OUT_DIR  # for convenient reference later

# -----------------------------------------------------------------------------
# 1 ◇ Utility functions
# -----------------------------------------------------------------------------
rng = np.random.default_rng(CONFIG['RNG_SEED'])

def safe_log2(p: float) -> float:
    return 0.0 if p <= 0 else log2(p)

# -----------------------------------------------------------------------------
# 2 ◇ Glauber dynamics (matches Experiment 1 description)
# -----------------------------------------------------------------------------
def glauber_sweep(spins: np.ndarray, T: float):
    """
    One Glauber sweep:
      - visit each site exactly once in a random order
      - flip with probability 1/(1+exp(ΔE/T))
    """
    L = spins.shape[0]
    order = rng.permutation(L * L)
    J = CONFIG['J']

    for idx in order:
        i = idx // L
        j = idx % L

        s = spins[i, j]
        nn = (
            spins[(i + 1) % L, j] +
            spins[(i - 1) % L, j] +
            spins[i, (j + 1) % L] +
            spins[i, (j - 1) % L]
        )
        dE = 2.0 * J * s * nn
        # Glauber flip probability
        if rng.random() < 1.0 / (1.0 + np.exp(dE / T)):
            spins[i, j] = -s

# -----------------------------------------------------------------------------
# 3 ◇ PID tools (your existing package)
# -----------------------------------------------------------------------------
from pidtools import (
    two_source_pid, mutual_information as mi_multi,
    total_syn_effect,
    multi_source_un,
    multi_source_red,
)

# -----------------------------------------------------------------------------
# 4 ◇ Task geometry: random adjacent pairs + windows
# -----------------------------------------------------------------------------
def generate_window_pairs(L: int, n_pairs: int, W: int, rng_local: np.random.Generator):
    """
    Generate n_pairs of adjacent source spins inside the lattice,
    where the FIRST source is used as the 'center' of a W×W window.
    The window is fully contained inside the lattice.
    """
    radius = W // 2
    pairs = []

    rs = rng_local.integers(radius, L - radius, size=n_pairs)
    cs = rng_local.integers(radius, L - radius, size=n_pairs)

    for r, c in zip(rs, cs):
        direction = rng_local.integers(0, 4)
        if direction == 0:      # up
            r2, c2 = r - 1, c
        elif direction == 1:    # down
            r2, c2 = r + 1, c
        elif direction == 2:    # left
            r2, c2 = r, c - 1
        else:                   # right
            r2, c2 = r, c + 1

        top = int(r - radius)
        left = int(c - radius)
        pairs.append({
            's1': (int(r), int(c)),
            's2': (int(r2), int(c2)),
            'top': top,
            'left': left,
            'size': W,
        })
    return pairs

def generate_global_pairs(L: int, n_pairs: int, rng_local: np.random.Generator):
    """
    Generate n_pairs of adjacent source spins anywhere in the lattice.
    The target window is the whole lattice (global mean).
    """
    pairs = []
    for _ in range(n_pairs):
        if rng_local.random() < 0.5:
            r = int(rng_local.integers(0, L))
            c = int(rng_local.integers(0, L - 1))
            r2, c2 = r, c + 1
        else:
            r = int(rng_local.integers(0, L - 1))
            c = int(rng_local.integers(0, L))
            r2, c2 = r + 1, c

        pairs.append({
            's1': (r, c),
            's2': (r2, c2),
            'top': 0,
            'left': 0,
            'size': L,
        })
    return pairs

def build_tasks_spec(L: int, n_points: int, seed: int):
    """
    Build the specification of the 5 PID tasks:
      - W4, W6, W8, W10: local windows
      - GLOBAL: whole lattice
    Each task has `n_points` random adjacent source pairs.
    """
    rng_local = np.random.default_rng(seed)
    return {
        'W4':     generate_window_pairs(L, n_points, 4,  rng_local),
        'W6':     generate_window_pairs(L, n_points, 6,  rng_local),
        'W8':     generate_window_pairs(L, n_points, 8,  rng_local),
        'W10':    generate_window_pairs(L, n_points, 10, rng_local),
        'GLOBAL': generate_global_pairs(L, n_points,      rng_local),
    }

def quantize_abs_mean(window: np.ndarray) -> int:
    """
    Compute |mean(spins)| over the given window (spins in {-1,+1}),
    then discretize into 3 bins:
      - < 0.33       -> 0
      - 0.33–0.66    -> 1
      - > 0.66       -> 2
    """
    abs_mean = abs(window.mean())
    if abs_mean < 0.33:
        return 0
    elif abs_mean <= 0.66:
        return 1
    else:
        return 2

def counter3_to_df(counter: Counter, total: int) -> pd.DataFrame:
    """
    Convert a Counter over (S1,S2,T) into a DataFrame with columns
    ['S1','S2','T','Pr'], making sure all 2×2×3 states are present.
    """
    if total <= 0:
        raise ValueError("total must be positive")

    rows = [[s1, s2, t, cnt / total] for (s1, s2, t), cnt in counter.items()]
    df = pd.DataFrame(rows, columns=['S1', 'S2', 'T', 'Pr'])

    if len(df) < 12:
        all_states = pd.MultiIndex.from_product(
            [[0, 1], [0, 1], [0, 1, 2]],
            names=['S1', 'S2', 'T'],
        )
        df = (
            df.set_index(['S1', 'S2', 'T'])
              .reindex(all_states, fill_value=0.0)
              .reset_index()
        )
    return df

# -----------------------------------------------------------------------------
# 5 ◇ Sampling and PID computation per temperature
# -----------------------------------------------------------------------------
def sample_one_temperature(T: float, tasks_spec: dict):
    """
    For a single temperature T:
      - run Glauber dynamics with burn-in (PREHEAT_STEPS sweeps)
      - record N_FRAMES sweeps
      - for each recorded sweep, for each task & pair, update P(S1,S2,Tbin)
        where Tbin is coarse bin of |mean(spins)| in window (or global)
      - compute two-source PID per pair and average within each task
      - compute physical observables: |M|, chi (from Var(M)), Cv (from Var(E))
    """
    L = CONFIG['L']
    N_FRAMES = CONFIG['N_FRAMES']
    J = CONFIG['J']

    # initialize spins randomly in {-1,+1}
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    # burn-in
    for _ in range(CONFIG['PREHEAT_STEPS']):
        glauber_sweep(spins, T)

    mags = np.empty(N_FRAMES, dtype=np.float64)
    enes = np.empty(N_FRAMES, dtype=np.float64)

    # one Counter per pair per task
    pid_counters: dict[str, list[Counter]] = {
        task_name: [Counter() for _ in pairs]
        for task_name, pairs in tasks_spec.items()
    }

    # record sweeps
    for t in range(N_FRAMES):
        glauber_sweep(spins, T)

        # physical observables: magnetization
        mags[t] = spins.mean()

        # energy density (per-spin), using nearest-neighbor sum once
        # E = -J * sum_{<i,j>} s_i s_j ; implement via right/down neighbors
        s_right = spins * np.roll(spins, -1, axis=1)
        s_down  = spins * np.roll(spins, -1, axis=0)
        enes[t] = -J * (s_right.sum() + s_down.sum()) / (L * L)

        # sources as bits in {0,1}
        bits = (spins + 1) // 2

        # update PID counters
        for task_name, pairs in tasks_spec.items():
            counters = pid_counters[task_name]
            for idx, p in enumerate(pairs):
                (r1, c1) = p['s1']
                (r2, c2) = p['s2']
                top = p['top']
                left = p['left']
                W = p['size']

                s1 = int(bits[r1, c1])
                s2 = int(bits[r2, c2])

                window = spins[top:top + W, left:left + W]
                t_bin = quantize_abs_mean(window)  # 0 / 1 / 2

                counters[idx][(s1, s2, t_bin)] += 1

    # physical quantities
    mag_mean = float(np.abs(mags).mean())
    varM = float(mags.var())
    varE = float(enes.var())

    # PID per task: average over random pairs
    pid_avgs: dict[str, dict[str, float]] = {}
    for task_name, counters in pid_counters.items():
        metrics_list = []
        for c in counters:
            df = counter3_to_df(c, total=N_FRAMES)
            pid = two_source_pid(df[['S1', 'S2', 'T', 'Pr']], ['S1', 'S2'], 'T')
            metrics_list.append({
                'redundancy': pid['redundancy'],
                'synergy':    pid['synergy'],
                'unique_S1':  pid['unique_src1'],
                'unique_S2':  pid['unique_src2'],
            })

        keys = metrics_list[0].keys()
        pid_avgs[task_name] = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    return mag_mean, varM, varE, pid_avgs

def process_temperature(args):
    """Wrapper for multiprocessing."""
    T, tasks_spec = args
    mag, varM, varE, pid_avgs = sample_one_temperature(T, tasks_spec)
    return T, mag, varM, varE, pid_avgs

# -----------------------------------------------------------------------------
# 6 ◇ Main procedure
# -----------------------------------------------------------------------------
def main():
    start = time.time()

    L = CONFIG['L']
    n_points = CONFIG['N_POINTS']
    tasks_spec = build_tasks_spec(L, n_points, CONFIG['OBS_SEED'])
    task_names = list(tasks_spec.keys())

    # Experiment 1 temperature sweep: 2.0 to 2.8 with 50 points
    T_arr = np.linspace(CONFIG['T_MIN'], CONFIG['T_MAX'], CONFIG['N_T'])

    params = [(float(T), tasks_spec) for T in T_arr]

    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_temperature, params),
                total=len(params),
                desc='Temperature scan'
            )
        )

    # Collect results in T-order
    results.sort(key=lambda x: x[0])

    phys_mag, phys_chi_num, phys_cv_num = [], [], []
    vals: dict[str, list[dict]] = {name: [] for name in task_names}

    for T, mag, varM, varE, pid_avgs in results:
        phys_mag.append(mag)
        phys_chi_num.append(varM)
        phys_cv_num.append(varE)
        for task_name, metrics in pid_avgs.items():
            vals[task_name].append(metrics)

    phys_mag = np.asarray(phys_mag, dtype=np.float64)
    T_arr = np.asarray(T_arr, dtype=np.float64)

    # susceptibility and specific heat
    chi = np.asarray(phys_chi_num, dtype=np.float64) * (L**2) / T_arr
    cv  = np.asarray(phys_cv_num, dtype=np.float64) / (T_arr**2)

    # Build DataFrames for each task
    task_df: dict[str, pd.DataFrame] = {}
    for task_name, lst in vals.items():
        keys = list(lst[0].keys())
        df = pd.DataFrame([{k: d[k] for k in keys} for d in lst], index=T_arr)
        task_df[task_name] = df

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig_ids = []

    def plot_curve(y, label, fname):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(T_arr, y, 'o-')
        plt.axvline(2.269, ls='--', c='k', lw=0.8)
        plt.xlabel('Temperature $T$')
        plt.ylabel(label)
        plt.title(f'{label} vs. Temperature $T$')
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=300)
        fig_ids.append(fig)
        plt.close(fig)

    plot_curve(phys_mag, 'Magnetization $|M|$', 'fig1_mag.png')
    plot_curve(chi, 'Magnetic susceptibility $\\chi$', 'fig2_chi.png')
    plot_curve(cv, 'Specific heat $C_v$', 'fig3_cv.png')

    def plot_df(df, title, fname):
        fig = plt.figure(figsize=(6, 4))
        for c in df.columns:
            plt.plot(df.index, df[c], 'o-', label=c)
        plt.axvline(2.269, ls='--', c='k', lw=0.8)
        plt.xlabel('Temperature $T$')
        plt.ylabel('bits')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=300)
        fig_ids.append(fig)
        plt.close(fig)

    for task_name, df in task_df.items():
        plot_df(df, f'PID task {task_name}', f'fig4_{task_name}.png')

    # -------------------------------------------------------------------------
    # Save numeric results and correlations to Excel
    # -------------------------------------------------------------------------
    df_phys = pd.DataFrame({'|M|': phys_mag, 'chi': chi, 'C_v': cv}, index=T_arr)

    all_pid_df = pd.concat(
        [df.add_prefix(f'{task_name}_') for task_name, df in task_df.items()],
        axis=1
    )

    df_all = pd.concat([df_phys, all_pid_df], axis=1)

    with pd.ExcelWriter(OUT_DIR / 'ising_pid_results_makeup.xlsx', engine='xlsxwriter') as w:
        df_all.to_excel(w, sheet_name='data')

        corr = np.corrcoef(df_phys.values.T, all_pid_df.values.T)[:3, 3:]
        corr_df = pd.DataFrame(corr, index=df_phys.columns, columns=all_pid_df.columns)
        corr_df.to_excel(w, sheet_name='correlation')

    # -------------------------------------------------------------------------
    # Correlation table figure
    # -------------------------------------------------------------------------
    corr = np.corrcoef(df_phys.values.T, all_pid_df.values.T)[:3, 3:]
    fig_tbl = plt.figure(figsize=(18, 4))
    ax_tbl = fig_tbl.add_subplot(111)
    ax_tbl.axis('off')
    tbl = ax_tbl.table(
        cellText=np.round(corr, 3),
        rowLabels=df_phys.columns,
        colLabels=all_pid_df.columns,
        loc='center',
    )
    tbl.scale(1, 1.5)
    fig_ids.insert(0, fig_tbl)

    # -------------------------------------------------------------------------
    # Parameter & runtime page
    # -------------------------------------------------------------------------
    runtime = time.time() - start
    fig_param = plt.figure(figsize=(8.5, 4))
    plt.axis('off')
    txt = f"Run timestamp: {_ts}\nRuntime: {runtime:.1f} s\n"
    for k, v in CONFIG.items():
        if k != 'OUT_DIR':
            txt += f"{k}: {v}\n"
    plt.text(0.01, 0.99, txt, va='top', ha='left', family='monospace')
    fig_ids.insert(0, fig_param)

    # -------------------------------------------------------------------------
    # Save all figures into a PDF
    # -------------------------------------------------------------------------
    with PdfPages(OUT_DIR / 'ising_pid_report.pdf') as pdf:
        for fig in fig_ids:
            pdf.savefig(fig)

    print('PDF saved to', OUT_DIR / 'ising_pid_report.pdf')
    print('Excel saved to', OUT_DIR / 'ising_pid_results_makeup.xlsx')
    print('Output folder:', OUT_DIR)

if __name__ == '__main__':
    main()
