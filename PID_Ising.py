#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ising_pid_sim.py  ——  Ising model + PID analysis batch processing script (including sequences A2/B2/C2)
===============================================================
Automatically writes all outputs to a timestamped folder, and lists key parameters and runtime on the first page of the PDF.
"""

from __future__ import annotations
import itertools, json, time, datetime
from collections import Counter
from math import log2
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

plt.rcParams.update({
    'font.size':       12,   # Base font size
    'axes.titlesize':  16,   # Title size
    'axes.labelsize':  14,   # Axis label size
    'xtick.labelsize': 12,   # X tick size
    'ytick.labelsize': 12,   # Y tick size
    'legend.fontsize': 10,   # Legend font size
})

# -----------------------------------------------------------------------------
# 0 ◇ Global tunable parameters
# -----------------------------------------------------------------------------
CONFIG: dict = {
    'L': 128,                 # lattice size
    'J': 1.0,                # coupling
    'PREHEAT_STEPS': 20_000,  # burn-in steps
    'N_FRAMES':    80_000,   # number of recorded sweeps (every step)
    'T_MIN': 2.00,
    'T_MAX': 2.80,
    'N_T':   50,              # number of temperature points
    'N_POINTS': 50,          # number of observation sites
    'OBS_SEED': 6463,
    'RNG_SEED': 635546,
    'WITH_SHIFT': False,
}

# —— Create output directory based on timestamp ——
_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = Path(f"ising_pid_out_{_ts}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG['OUT_DIR'] = OUT_DIR  # For convenient reference later

# -----------------------------------------------------------------------------
# 1 ◇ Utility functions
# -----------------------------------------------------------------------------

rng = np.random.default_rng(CONFIG['RNG_SEED'])

def safe_log2(p: float) -> float: return 0.0 if p <= 0 else log2(p)

def choose_observation_points(L: int, n: int, seed: int):
    rng_local = np.random.default_rng(seed)
    cand = [(r, c) for r in range(1, L - 1) for c in range(1, L - 1)]
    idx = rng_local.choice(len(cand), size=n, replace=False)
    return [cand[i] for i in idx]

def counter_to_df(counter: Counter, total: int) -> pd.DataFrame:
    rows = [[*key, cnt / total] for key, cnt in counter.items()]
    df = pd.DataFrame(rows, columns=['U', 'D', 'L', 'R', 'C', 'Pr'])
    if len(df) < 32:
        all_states = pd.MultiIndex.from_product([[0, 1]] * 5, names=['U', 'D', 'L', 'R', 'C'])
        df = df.set_index(['U', 'D', 'L', 'R', 'C']).reindex(all_states, fill_value=0).reset_index()
    return df

def pairwise_mi(df: pd.DataFrame, x: str, y: str) -> float:
    joint = df.groupby([x, y], sort=False)['Pr'].sum().reset_index()
    px = joint.groupby(x, sort=False)['Pr'].sum(); py = joint.groupby(y, sort=False)['Pr'].sum()
    return sum(p * safe_log2(p / (px[row[x]] * py[row[y]])) for _, row in joint.iterrows() if (p := row['Pr']) > 0)

# -----------------------------------------------------------------------------
# 2 ◇ Metropolis and sampling
# -----------------------------------------------------------------------------

def mc_step(spins: np.ndarray, beta: float):
    L = spins.shape[0]
    for _ in range(L * L):
        i, j = rng.integers(0, L, size=2)
        s = spins[i, j]
        nn = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
        dE = 2 * CONFIG['J'] * s * nn
        if dE <= 0 or rng.random() < np.exp(-beta * dE): spins[i, j] = -s

from pidtools import (
    two_source_pid, mutual_information as mi_multi,
    total_syn_effect,
    multi_source_un,
    multi_source_red)


def sample_one_temperature(T: float, obs_points):
    L = CONFIG['L']; beta = 1.0 / T
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    for _ in range(CONFIG['PREHEAT_STEPS']): mc_step(spins, beta)

    mags, enes = [], []
    cnt_static = [Counter() for _ in obs_points]
    cnt_shift  = [Counter() for _ in obs_points]
    prev_neigh = [None]*len(obs_points)

    for _ in range(CONFIG['N_FRAMES']):
        mc_step(spins, beta)
        mags.append(spins.mean())
        enes.append(-CONFIG['J']*(np.roll(spins,1,0)*spins+np.roll(spins,-1,0)*spins+np.roll(spins,1,1)*spins+np.roll(spins,-1,1)*spins).sum()/2/L**2)
        for idx,(r,c) in enumerate(obs_points):
            u,d,l,rgt = (spins[r-1,c]+1)//2,(spins[r+1,c]+1)//2,(spins[r,c-1]+1)//2,(spins[r,c+1]+1)//2
            cen=(spins[r,c]+1)//2
            cnt_static[idx][(u,d,l,rgt,cen)]+=1
            if prev_neigh[idx] is not None:
                cnt_shift[idx][(*prev_neigh[idx],cen)]+=1
            prev_neigh[idx]=(u,d,l,rgt)
    mags=np.asarray(mags); enes=np.asarray(enes)
    df_static=[counter_to_df(c, CONFIG['N_FRAMES']) for c in cnt_static]
    # print(cnt_static)
    df_shift =[counter_to_df(c, CONFIG['N_FRAMES']-1) for c in cnt_shift]
    # print(cnt_shift)
    return (np.abs(mags).mean(), mags.var(), enes.var(), df_static, df_shift)

# -----------------------------------------------------------------------------
# 3 ◇ PID task wrappers (shared by A/B/C and A2/B2/C2)
# -----------------------------------------------------------------------------

def task_A(df):
    pid=two_source_pid(df[['L','R','C','Pr']],[ 'L','R'],'C');d=df.copy();d['LR']=d['L'].astype(int).astype(str)+','+d['R'].astype(int).astype(str)
    return {'redundancy':pid['redundancy'],'synergy':pid['synergy'],'unique_L':pid['unique_src1'],'unique_R':pid['unique_src2'],'mi_C_LR':pairwise_mi(d[['LR','C','Pr']],'LR','C')}

def task_B(df):
    d=df.copy();d['UD']=d['U'].astype(int).astype(str)+','+d['D'].astype(int).astype(str);d['LR']=d['L'].astype(int).astype(str)+','+d['R'].astype(int).astype(str);d['UDLR']=d['UD']+','+d['LR']
    pid=two_source_pid(d[['UD','LR','C','Pr']],[ 'UD','LR'],'C');
    return {'redundancy':pid['redundancy'],'synergy':pid['synergy'],'unique_UD':pid['unique_src1'],'unique_LR':pid['unique_src2'],'mi_C_UDLR':pairwise_mi(d[['UDLR','C','Pr']],'UDLR','C')}

def task_C(df):
    tse=total_syn_effect(df[['U','D','L','R','C','Pr']],[ 'U','D','L','R'],'C');
    un=multi_source_un(df[['U','D','L','R','C','Pr']],[ 'U','D','L','R'],'C')
    return {'TSE':tse,'Un_U':un.iloc[0],'Un_D':un.iloc[1],'Un_R':un.iloc[2],'Un_L':un.iloc[3]}

def process_temperature(args):
    T, obs_pts = args
    mag, varM, varE, dfs, dfs2 = sample_one_temperature(T, obs_pts)
    task_results = {'A': [], 'B': [], 'C': []}
    for df in dfs:
        task_results['A'].append(task_A(df))
        task_results['B'].append(task_B(df))
        task_results['C'].append(task_C(df))
    return T, mag, varM, varE, task_results


# -----------------------------------------------------------------------------
# 4 ◇ Main procedure
# -----------------------------------------------------------------------------

def main():
    start = time.time()
    obs_pts = choose_observation_points(CONFIG['L'], CONFIG['N_POINTS'], CONFIG['OBS_SEED'])
    T_arr = np.linspace(CONFIG['T_MIN'], CONFIG['T_MAX'], CONFIG['N_T'])

    params = [(T, obs_pts) for T in T_arr]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_temperature, params),
                            total=len(params), desc='温度扫描'))

    phys_mag, phys_chi_num, phys_cv_num = [], [], []
    vals = {'A': [], 'B': [], 'C': []}

    for T, mag, varM, varE, task_res in results:
        phys_mag.append(mag)
        phys_chi_num.append(varM)
        phys_cv_num.append(varE)
        for tag in ['A', 'B', 'C']:
            vals[tag].extend(task_res[tag])
    def avg_list(lst,keylist): return {k:np.mean([d[k] for d in lst]) for k in keylist}
    n_pts=CONFIG['N_POINTS']
    keysA=list(vals['A'][0].keys())
    task_df={}
    for tag in ['A','B','C']:
        grouped=[vals[tag][i*n_pts:(i+1)*n_pts] for i in range(CONFIG['N_T'])]
        task_df[tag]=pd.DataFrame([avg_list(g,keysA if 'A' in tag else list(g[0].keys())) for g in grouped],index=T_arr)
    phys_mag=np.asarray(phys_mag);chi=np.asarray(phys_chi_num)*CONFIG['L']**2/T_arr;cv=np.asarray(phys_cv_num)/T_arr**2

    fig_ids=[]
    def plot_curve(y,label,fname):
        fig=plt.figure(figsize=(6,4));plt.plot(T_arr,y,'o-');plt.axvline(2.269,ls='--',c='k',lw=0.8);
        plt.xlabel('Temperature $T$');plt.ylabel(label);
        plt.title(f'{label} vs. Temperature $T$');
        plt.tight_layout();
        plt.savefig(OUT_DIR/fname,dpi=300);
        fig_ids.append(fig);plt.close(fig)
    plot_curve(phys_mag,'Magnetization $|M|$','fig1_mag.png');
    plot_curve(chi,'Magnetic susceptibility $χ$','fig2_chi.png');
    plot_curve(cv,'Specific heat $C_v$','fig3_cv.png')
    def plot_df(df,title,fname):
        fig=plt.figure(figsize=(6,4));
        for c in df.columns: plt.plot(df.index,df[c],'o-',label=c)
        plt.axvline(2.269,ls='--',c='k',lw=0.8);
        plt.xlabel('Temperature $T$');
        plt.ylabel('bits');
        plt.title(title);plt.legend();
        plt.tight_layout();
        plt.savefig(OUT_DIR/fname,dpi=300);
        fig_ids.append(fig);
        plt.close(fig)
    plot_df(task_df['A'],'Left–Right Decomposition','fig4_A.png');
    plot_df(task_df['B'],'Vertical–Horizontal Decomposition','fig5_B.png');
    plot_df(task_df['C'],'Four-source Decomposition','fig6_C.png')

    df_phys=pd.DataFrame({'|M|':phys_mag,'χ':chi,'C_v':cv},index=T_arr)
    df_all=pd.concat([df_phys]+[task_df[t] for t in task_df],axis=1)
    with pd.ExcelWriter(OUT_DIR/'ising_pid_results2.xlsx',engine='xlsxwriter') as w:
        df_all.to_excel(w,sheet_name='data');
        corr=np.corrcoef(df_phys.values.T,pd.concat([task_df[t] for t in task_df],axis=1).values.T)[:3,3:]
        corr_df=pd.DataFrame(corr,index=df_phys.columns,columns=pd.concat([task_df[t] for t in task_df],axis=1).columns)
        corr_df.to_excel(w,sheet_name='correlation')

    fig_tbl=plt.figure(figsize=(18,4));ax_tbl=fig_tbl.add_subplot(111);ax_tbl.axis('off');tbl=ax_tbl.table(cellText=np.round(corr,3),rowLabels=corr_df.index,colLabels=corr_df.columns,loc='center');tbl.scale(1,1.5);fig_ids.insert(0,fig_tbl)

    runtime=time.time()-start
    fig_param=plt.figure(figsize=(8.5,4));plt.axis('off');
    txt="Run timestamp: {}\nRuntime: {:.1f} s\n".format(_ts,runtime)
    for k,v in CONFIG.items():
        if k!='OUT_DIR': txt+=f"{k}: {v}\n"
    plt.text(0.01,0.99,txt,va='top',ha='left',family='monospace');fig_ids.insert(0,fig_param)

    with PdfPages(OUT_DIR/'ising_pid_report.pdf') as pdf:
        for fig in fig_ids: pdf.savefig(fig)
    print('PDF saved to',OUT_DIR/'ising_pid_report.pdf')

if __name__=='__main__':
    main()
