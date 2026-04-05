"""
Generate visualization figures for NMMR-Q experimental results.
Produces publication-quality plots for SGD simulation and RHC real-data experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SUMMARY_DIR = os.path.join(PROJECT_ROOT, 'summary')
OUTDIR = os.path.join(SUMMARY_DIR, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────

# --- NMMR standalone (SGD) ---
def load_nmmr_standalone(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['tp_model_name'] = df['tp_model_name'].str.strip()
    df['tp_loss_name'] = df['tp_loss_name'].str.strip()
    return df

nmmr_2000 = load_nmmr_standalone(os.path.join(SUMMARY_DIR, 'nmmr_train_2000.csv'))
nmmr_5000 = load_nmmr_standalone(os.path.join(SUMMARY_DIR, 'nmmr_train_5000.csv'))

# --- DR (SGD) ---
dr_u_2000 = pd.read_csv(os.path.join(SUMMARY_DIR, 'dr_u_sgd_s1_2000.csv'))
dr_v_2000 = pd.read_csv(os.path.join(SUMMARY_DIR, 'dr_v_sgd_s1_2000.csv'))
dr_u_5000 = pd.read_csv(os.path.join(SUMMARY_DIR, 'dr_u_sgd_1_5000.csv'))
dr_v_5000 = pd.read_csv(os.path.join(SUMMARY_DIR, 'dr_v_sgd_s1_5000.csv'))

# --- RHC ---
rhc_u_boot = pd.read_csv(os.path.join(SUMMARY_DIR, 'u_rhc_boostrap_100.csv'))
rhc_v_boot = pd.read_csv(os.path.join(SUMMARY_DIR, 'v_rhc_boostrap_100.csv'))
rhc_u_rep  = pd.read_csv(os.path.join(SUMMARY_DIR, 'u_rhc_repeated_100.csv'))

TRUE_ATE_SGD = 2.0

# ──────────────────────────────────────────────
# Helper: extract NMMR standalone by method
# ──────────────────────────────────────────────
def extract_ate(df, model, loss):
    sub = df[(df['tp_model_name'] == model) & (df['tp_loss_name'] == loss)]
    return sub['avg_ate'].astype(float).values

# ──────────────────────────────────────────────
# Figure 1: SGD Box-plots — NMMR standalone + DR
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, n_label, nmmr_df, dr_u, dr_v in [
    (axes[0], 'n = 2000', nmmr_2000, dr_u_2000, dr_v_2000),
    (axes[1], 'n = 5000', nmmr_5000, dr_u_5000, dr_v_5000),
]:
    data_dict = {
        'H (U)':  extract_ate(nmmr_df, 'nmmr_h', 'U_statistic'),
        'H (V)':  extract_ate(nmmr_df, 'nmmr_h', 'V_statistic'),
        'Q (U)':  extract_ate(nmmr_df, 'nmmr_q', 'U_statistic'),
        'Q (V)':  extract_ate(nmmr_df, 'nmmr_q', 'V_statistic'),
        'PIPW\n(U)': dr_u['PIPW'].values,
        'POR\n(U)':  dr_u['POR'].values,
        'PDR\n(U)':  dr_u['PDR'].values,
        'PIPW\n(V)': dr_v['PIPW'].values,
        'POR\n(V)':  dr_v['POR'].values,
        'PDR\n(V)':  dr_v['PDR'].values,
    }

    positions = list(range(len(data_dict)))
    bp = ax.boxplot(data_dict.values(), positions=positions, widths=0.6,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color: standalone blue, DR-U green, DR-V orange
    colors = ['#4C72B0']*4 + ['#55A868']*3 + ['#C44E52']*3
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(TRUE_ATE_SGD, color='red', ls='--', lw=1.5, label='True ATE = 2.0')
    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys(), fontsize=9)
    ax.set_title(n_label, fontsize=15, fontweight='bold')
    ax.set_ylabel('ATE')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Divider lines
    ax.axvline(3.5, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(6.5, color='gray', ls=':', lw=0.8, alpha=0.5)

fig.suptitle('SGD Simulation: ATE Estimates by Method and Statistic Type', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig1_sgd_boxplot.pdf')
plt.savefig(f'{OUTDIR}/fig1_sgd_boxplot.png')
plt.close()
print("[✓] Figure 1: SGD box-plots saved.")

# ──────────────────────────────────────────────
# Figure 2: SGD Bias bar chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

methods = []
biases = []
stds = []
colors_bar = []
hatches = []

for n_label, nmmr_df, dr_u, dr_v in [
    ('n=2000', nmmr_2000, dr_u_2000, dr_v_2000),
    ('n=5000', nmmr_5000, dr_u_5000, dr_v_5000),
]:
    items = [
        ('H(U)', extract_ate(nmmr_df, 'nmmr_h', 'U_statistic'), '#4C72B0'),
        ('H(V)', extract_ate(nmmr_df, 'nmmr_h', 'V_statistic'), '#4C72B0'),
        ('Q(U)', extract_ate(nmmr_df, 'nmmr_q', 'U_statistic'), '#64B5F6'),
        ('Q(V)', extract_ate(nmmr_df, 'nmmr_q', 'V_statistic'), '#64B5F6'),
        ('PIPW(U)', dr_u['PIPW'].values, '#55A868'),
        ('POR(U)',  dr_u['POR'].values,  '#55A868'),
        ('PDR(U)',  dr_u['PDR'].values,  '#55A868'),
        ('PIPW(V)', dr_v['PIPW'].values, '#C44E52'),
        ('POR(V)',  dr_v['POR'].values,  '#C44E52'),
        ('PDR(V)',  dr_v['PDR'].values,  '#C44E52'),
    ]
    for name, vals, c in items:
        methods.append(f'{name}\n{n_label}')
        biases.append(np.mean(vals) - TRUE_ATE_SGD)
        stds.append(np.std(vals))
        colors_bar.append(c)

x = np.arange(len(methods))
bars = ax.bar(x, biases, yerr=stds, width=0.7, color=colors_bar, alpha=0.75,
              edgecolor='black', linewidth=0.5, capsize=3, error_kw={'lw': 0.8})

ax.axhline(0, color='red', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Bias (= Mean ATE − True ATE)')
ax.set_title('SGD Simulation: Bias ± SD by Method', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# vertical dividers between n=2000 and n=5000
ax.axvline(9.5, color='gray', ls='--', lw=1, alpha=0.5)
ax.text(4.5, ax.get_ylim()[1]*0.9, 'n = 2000', ha='center', fontsize=12, fontstyle='italic')
ax.text(14.5, ax.get_ylim()[1]*0.9, 'n = 5000', ha='center', fontsize=12, fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig2_sgd_bias.pdf')
plt.savefig(f'{OUTDIR}/fig2_sgd_bias.png')
plt.close()
print("[✓] Figure 2: SGD bias chart saved.")

# ──────────────────────────────────────────────
# Figure 3: SGD — DR estimators comparison (U vs V, n=2000 vs 5000)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, n_label, dr_u, dr_v in [
    (axes[0], 'n = 2000', dr_u_2000, dr_v_2000),
    (axes[1], 'n = 5000', dr_u_5000, dr_v_5000),
]:
    data = {
        'PIPW (U)': dr_u['PIPW'].values,
        'POR (U)':  dr_u['POR'].values,
        'PDR (U)':  dr_u['PDR'].values,
        'PIPW (V)': dr_v['PIPW'].values,
        'POR (V)':  dr_v['POR'].values,
        'PDR (V)':  dr_v['PDR'].values,
    }

    bp = ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    colors = ['#55A868']*3 + ['#C44E52']*3
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(TRUE_ATE_SGD, color='red', ls='--', lw=1.5, label='True ATE = 2.0')
    ax.axvline(3.5, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_title(n_label, fontsize=15, fontweight='bold')
    ax.set_ylabel('ATE')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

fig.suptitle('SGD: Proximal DR Estimators (PIPW / POR / PDR)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig3_sgd_dr_boxplot.pdf')
plt.savefig(f'{OUTDIR}/fig3_sgd_dr_boxplot.png')
plt.close()
print("[✓] Figure 3: SGD DR box-plots saved.")

# ──────────────────────────────────────────────
# Figure 4: RHC Bootstrap — point estimates + 95% CI
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Cui et al. reference values
cui_ref = {'PIPW': -1.72, 'POR': -1.80, 'PDR': -1.66}

y_pos = []
y_labels = []
idx = 0

def plot_rhc_ci(ax, df, label_prefix, color, idx_start):
    idx = idx_start
    for col in ['PIPW', 'POR', 'PDR']:
        vals = df[col].values.astype(float)
        mean = np.mean(vals)
        ci_lo = np.percentile(vals, 2.5)
        ci_hi = np.percentile(vals, 97.5)
        sd = np.std(vals)

        ax.errorbar(mean, idx, xerr=[[mean - ci_lo], [ci_hi - mean]],
                     fmt='o', color=color, markersize=8, capsize=5, capthick=1.5,
                     elinewidth=1.5, label=f'{label_prefix}' if col == 'PIPW' else '')
        ax.text(ci_hi + 0.08, idx, f'{mean:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]',
                va='center', fontsize=9, color=color)

        y_pos.append(idx)
        y_labels.append(f'{col} ({label_prefix})')
        idx += 1
    return idx

idx = plot_rhc_ci(ax, rhc_u_boot, 'U-stat Boot', '#4C72B0', 0)
idx += 0.5  # gap
idx = plot_rhc_ci(ax, rhc_v_boot, 'V-stat Boot', '#C44E52', idx)

# Reference lines from Cui et al.
ax.axvline(cui_ref['PDR'], color='green', ls='--', lw=1.2, alpha=0.7, label='Cui et al. PDR = −1.66')
ax.axvline(cui_ref['PIPW'], color='purple', ls=':', lw=1.2, alpha=0.7, label='Cui et al. PIPW = −1.72')
ax.axvline(cui_ref['POR'], color='orange', ls=':', lw=1.2, alpha=0.7, label='Cui et al. POR = −1.80')

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels)
ax.set_xlabel('ATE')
ax.set_title('RHC: Bootstrap 95% CI (100 replicates)', fontsize=15, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig4_rhc_bootstrap_ci.pdf')
plt.savefig(f'{OUTDIR}/fig4_rhc_bootstrap_ci.png')
plt.close()
print("[✓] Figure 4: RHC bootstrap CI saved.")

# ──────────────────────────────────────────────
# Figure 5: RHC Bootstrap — box-plots U vs V
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, df, stat_label, color in [
    (axes[0], rhc_u_boot, 'U-statistic', '#4C72B0'),
    (axes[1], rhc_v_boot, 'V-statistic', '#C44E52'),
]:
    data = {col: df[col].values.astype(float) for col in ['PIPW', 'POR', 'PDR']}
    bp = ax.boxplot(data.values(), labels=data.keys(), patch_artist=True,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Cui reference
    for i, col in enumerate(['PIPW', 'POR', 'PDR']):
        ax.plot(i + 1, cui_ref[col], 'D', color='gold', markersize=10, markeredgecolor='black',
                markeredgewidth=1, zorder=5, label='Cui et al.' if i == 0 else '')

    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_title(f'RHC Bootstrap — {stat_label}', fontsize=14, fontweight='bold')
    ax.set_ylabel('ATE')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('RHC: Bootstrap ATE Estimates vs Cui et al. Reference', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig5_rhc_bootstrap_boxplot.pdf')
plt.savefig(f'{OUTDIR}/fig5_rhc_bootstrap_boxplot.png')
plt.close()
print("[✓] Figure 5: RHC bootstrap box-plots saved.")

# ──────────────────────────────────────────────
# Figure 6: RHC U-stat Repeated — showing H degeneracy
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for col, color, marker in [('PIPW', '#4C72B0', 'o'), ('POR', '#55A868', 's'), ('PDR', '#C44E52', '^')]:
    vals = rhc_u_rep[col].values.astype(float)
    ax.plot(range(len(vals)), vals, marker=marker, color=color, alpha=0.6,
            markersize=4, linewidth=0.8, label=col)

ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_xlabel('Trial')
ax.set_ylabel('ATE')
ax.set_title('RHC U-stat Repeated: POR Degeneracy (POR ≡ 0 due to H U-stat failure)',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Annotation
ax.annotate('POR ≡ 0: H(U-stat) degenerates\non high-dim RHC data',
            xy=(50, 0), xytext=(60, -1.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=11, color='green', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig6_rhc_u_repeated_degeneracy.pdf')
plt.savefig(f'{OUTDIR}/fig6_rhc_u_repeated_degeneracy.png')
plt.close()
print("[✓] Figure 6: RHC U-stat repeated (degeneracy) saved.")

# ──────────────────────────────────────────────
# Figure 7: Summary statistics table (all experiments)
# ──────────────────────────────────────────────
rows = []

def add_stats(name, vals, true_ate=None):
    m = np.mean(vals)
    s = np.std(vals)
    ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
    bias = m - true_ate if true_ate is not None else np.nan
    rows.append({
        'Experiment': name,
        'Mean': round(m, 4),
        'SD': round(s, 4),
        'Bias': round(bias, 4) if true_ate is not None else '',
        '95% CI Low': round(ci_lo, 4),
        '95% CI High': round(ci_hi, 4),
        'N': len(vals),
    })

# SGD standalone
for n_label, df in [('2000', nmmr_2000), ('5000', nmmr_5000)]:
    for model, loss, label in [
        ('nmmr_h', 'U_statistic', 'H(U)'),
        ('nmmr_h', 'V_statistic', 'H(V)'),
        ('nmmr_q', 'U_statistic', 'Q(U)'),
        ('nmmr_q', 'V_statistic', 'Q(V)'),
    ]:
        add_stats(f'SGD n={n_label} {label}', extract_ate(df, model, loss), TRUE_ATE_SGD)

# SGD DR
for n_label, dr_u, dr_v in [
    ('2000', dr_u_2000, dr_v_2000), ('5000', dr_u_5000, dr_v_5000)
]:
    for stat, dr_df in [('U', dr_u), ('V', dr_v)]:
        for col in ['PIPW', 'POR', 'PDR']:
            add_stats(f'SGD n={n_label} {col}({stat})', dr_df[col].values.astype(float), TRUE_ATE_SGD)

# RHC
for stat, df in [('U-Boot', rhc_u_boot), ('V-Boot', rhc_v_boot)]:
    for col in ['PIPW', 'POR', 'PDR']:
        add_stats(f'RHC {stat} {col}', df[col].values.astype(float))

for col in ['PIPW', 'POR', 'PDR']:
    add_stats(f'RHC U-Rep {col}', rhc_u_rep[col].values.astype(float))

stats_df = pd.DataFrame(rows)
stats_df.to_csv(f'{OUTDIR}/summary_statistics.csv', index=False)
print(f"[✓] Summary statistics table saved ({len(rows)} rows).")

# ──────────────────────────────────────────────
# Figure 8: SGD convergence — RMSE across sample sizes
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

methods_rmse = {}
for label, get_vals_fn in [
    ('H(U)', lambda df: extract_ate(df, 'nmmr_h', 'U_statistic')),
    ('H(V)', lambda df: extract_ate(df, 'nmmr_h', 'V_statistic')),
    ('Q(U)', lambda df: extract_ate(df, 'nmmr_q', 'U_statistic')),
    ('Q(V)', lambda df: extract_ate(df, 'nmmr_q', 'V_statistic')),
]:
    rmses = []
    for df in [nmmr_2000, nmmr_5000]:
        vals = get_vals_fn(df)
        rmses.append(np.sqrt(np.mean((vals - TRUE_ATE_SGD)**2)))
    methods_rmse[label] = rmses

for label, get_vals_fn in [
    ('PIPW(U)', lambda dr_u, dr_v: dr_u['PIPW'].values),
    ('POR(U)',  lambda dr_u, dr_v: dr_u['POR'].values),
    ('PDR(U)',  lambda dr_u, dr_v: dr_u['PDR'].values),
    ('PIPW(V)', lambda dr_u, dr_v: dr_v['PIPW'].values),
    ('POR(V)',  lambda dr_u, dr_v: dr_v['POR'].values),
    ('PDR(V)',  lambda dr_u, dr_v: dr_v['PDR'].values),
]:
    rmses = []
    for dr_u, dr_v in [(dr_u_2000, dr_v_2000), (dr_u_5000, dr_v_5000)]:
        vals = get_vals_fn(dr_u, dr_v).astype(float)
        rmses.append(np.sqrt(np.mean((vals - TRUE_ATE_SGD)**2)))
    methods_rmse[label] = rmses

ns = [2000, 5000]
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
colors_line = plt.cm.tab10(np.linspace(0, 1, len(methods_rmse)))

for i, (label, rmses) in enumerate(methods_rmse.items()):
    ax.plot(ns, rmses, marker=markers[i % len(markers)], color=colors_line[i],
            label=label, linewidth=1.5, markersize=7)

ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('RMSE')
ax.set_title('SGD: RMSE vs Sample Size', fontsize=15, fontweight='bold')
ax.legend(ncol=2, fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(ns)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig7_sgd_rmse_convergence.pdf')
plt.savefig(f'{OUTDIR}/fig7_sgd_rmse_convergence.png')
plt.close()
print("[✓] Figure 7: SGD RMSE convergence saved.")

print("\n" + "="*50)
print("All figures saved to figures/ directory.")
print("="*50)
