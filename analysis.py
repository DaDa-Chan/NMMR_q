"""
NMMR-Q 项目数据分析脚本
1. NMMR_Q / NMMR_H 重复实验分析与假设检验
2. Linear model 分析
3. NMMR 单独训练 vs Linear 横向比较
4. 双重稳健性假设检验与 Linear 横向比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'

TRUE_ATE = 2.0
OUTPUT_DIR = '/Users/chen/Study/CI/NMMR_q/figures'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
tune_df = pd.read_csv('/Users/chen/Study/CI/NMMR_q/summary/nmmr_tune_5000.csv', skipinitialspace=True)
tune_df.columns = tune_df.columns.str.strip()

train_df = pd.read_csv('/Users/chen/Study/CI/NMMR_q/summary/nmmr_train_5000.csv', skipinitialspace=True)
train_df.columns = train_df.columns.str.strip()

dr_u_df = pd.read_csv('/Users/chen/Study/CI/NMMR_q/summary/dr_u_s1_5000.csv', skipinitialspace=True)
dr_u_df.columns = dr_u_df.columns.str.strip()

dr_v_df = pd.read_csv('/Users/chen/Study/CI/NMMR_q/summary/df_v_s1_5000.csv', skipinitialspace=True)
dr_v_df.columns = dr_v_df.columns.str.strip()

linear_df = pd.read_csv('/Users/chen/Study/CI/NMMR_q/predicts/linear/linear_predict.csv', skipinitialspace=True)
linear_df.columns = linear_df.columns.str.strip()

fold_cols = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

def get_group(model, loss):
    mask = ((train_df['tp_model_name'].str.strip() == model) &
            (train_df['tp_loss_name'].str.strip() == loss))
    return train_df[mask]

groups = {
    'NMMR_H (U-stat)': get_group('nmmr_h', 'U_statistic'),
    'NMMR_H (V-stat)': get_group('nmmr_h', 'V_statistic'),
    'NMMR_Q (U-stat)': get_group('nmmr_q', 'U_statistic'),
    'NMMR_Q (V-stat)': get_group('nmmr_q', 'V_statistic'),
}
all_ates = {name: df['avg_ate'].values for name, df in groups.items()}

def ttest(values, name):
    t, p = stats.ttest_1samp(values, TRUE_ATE)
    m, s = np.mean(values), np.std(values, ddof=1)
    se = s / np.sqrt(len(values))
    ci = (m - 1.96*se, m + 1.96*se)
    sig = 'Cannot reject H0' if p > 0.05 else 'Reject H0'
    return {'name': name, 'mean': m, 'std': s, 'bias': m - TRUE_ATE,
            'ci_low': ci[0], 'ci_high': ci[1], 't': t, 'p': p, 'sig': sig}

# ============================================================
# Part 1: NMMR_Q / NMMR_H 分析与假设检验
# ============================================================
print("=" * 60)
print("Part 1: NMMR 重复实验分析与假设检验")
print("=" * 60)

# 调优
tune_h = tune_df[tune_df['model type'].str.strip() == 'nmmr_h'].copy()
tune_q = tune_df[tune_df['model type'].str.strip() == 'nmmr_q'].copy()
tune_h['bias'] = abs(tune_h['ATE'] - TRUE_ATE)
tune_q['bias'] = abs(tune_q['ATE'] - TRUE_ATE)

for model, tdf in [('NMMR_H', tune_h), ('NMMR_Q', tune_q)]:
    print(f"\n{model} 调优:")
    for lt in sorted(tdf['loss type'].str.strip().unique()):
        sub = tdf[tdf['loss type'].str.strip() == lt]
        best = sub.loc[sub['bias'].idxmin()]
        print(f"  {lt.upper()}-stat: 最优ATE={best['ATE']:.4f}, bias={best['bias']:.4f}, "
              f"width={int(best['net_width'])}, lr={float(best['lr']):.6f}")

# 重复实验
nmmr_tests = []
for name, ates in all_ates.items():
    r = ttest(ates, name)
    nmmr_tests.append(r)
    print(f"\n{name}: Mean={r['mean']:.4f}±{r['std']:.4f}, Bias={r['bias']:.4f}, "
          f"95%CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}], p={r['p']:.4f} -> {r['sig']}")

# --- 图1: NMMR 箱线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
colors_4 = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
bp = ax.boxplot([all_ates[k] for k in all_ates],
                labels=[k.replace(' ', '\n') for k in all_ates],
                patch_artist=True, widths=0.5, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
for patch, c in zip(bp['boxes'], colors_4):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(y=TRUE_ATE, color='red', linestyle='--', linewidth=2, label='True ATE = 2')
ax.set_ylabel('ATE Estimate', fontsize=12)
ax.set_title('NMMR Repeated Experiments (20 trials, n=5000, 5-fold CF)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_nmmr_boxplot.png')
plt.close()

# --- 图2: NMMR 折叠间方差 ---
fig, ax = plt.subplots(figsize=(10, 6))
intra_stds = {name: np.std(groups[name][fold_cols].values, axis=1, ddof=1) for name in groups}
bp = ax.boxplot(intra_stds.values(),
                labels=[k.replace(' ', '\n') for k in intra_stds],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
for patch, c in zip(bp['boxes'], colors_4):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel('Intra-experiment Fold Std Dev', fontsize=12)
ax.set_title('Cross-Fitting Stability (Lower = More Stable)', fontsize=13)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_nmmr_fold_variance.png')
plt.close()

# --- 图3: NMMR 假设检验森林图 ---
fig, ax = plt.subplots(figsize=(9, 4))
for i, r in enumerate(nmmr_tests):
    pc = 'green' if r['p'] > 0.05 else 'red'
    ax.errorbar(r['mean'], i, xerr=[[r['mean']-r['ci_low']], [r['ci_high']-r['mean']]],
                fmt='o', color=pc, capsize=5, markersize=8, linewidth=2)
    sig_star = '' if r['p'] > 0.05 else ('*' if r['p'] > 0.01 else ('**' if r['p'] > 0.001 else '***'))
    ax.text(r['ci_high'] + 0.003, i, f'p={r["p"]:.4f} {sig_star}', va='center', fontsize=10)
ax.axvline(x=TRUE_ATE, color='red', linestyle='--', linewidth=2, alpha=0.7, label='True ATE=2')
ax.set_yticks(range(len(nmmr_tests)))
ax.set_yticklabels([r['name'] for r in nmmr_tests])
ax.set_xlabel('ATE Estimate (95% CI)')
ax.set_title('NMMR Hypothesis Test: H₀: ATE = 2', fontsize=13)
ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_nmmr_hypothesis_test.png')
plt.close()

print("\n[图1-3] NMMR 分析图表已保存")

# ============================================================
# Part 2: Linear model 分析
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Linear Model 分析")
print("=" * 60)

linear_s1 = {}
for est in ['POR', 'PIPW', 'PDR']:
    v = linear_df[f'{est}_s1'].values
    r = ttest(v, f'Linear {est}')
    linear_s1[est] = r
    print(f"  {est}: Mean={r['mean']:.4f}±{r['std']:.4f}, Bias={r['bias']:.4f}, p={r['p']:.4f}")

# ============================================================
# Part 3: NMMR 单独训练 vs Linear 横向比较
# ============================================================
print("\n" + "=" * 60)
print("Part 3: NMMR vs Linear 横向比较")
print("=" * 60)

# 汇总表
comp_rows = []
for name, ates in all_ates.items():
    comp_rows.append({'Method': name, 'Mean': np.mean(ates),
                      'Std': np.std(ates, ddof=1), 'Bias': np.mean(ates) - TRUE_ATE})
for est in ['POR', 'PIPW', 'PDR']:
    v = linear_df[f'{est}_s1'].values
    comp_rows.append({'Method': f'Linear {est}', 'Mean': np.mean(v),
                      'Std': np.std(v, ddof=1), 'Bias': np.mean(v) - TRUE_ATE})
comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False, float_format='%.4f'))

# --- 图4: NMMR vs Linear 横向比较 ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 左: 箱线图
ax = axes[0]
nmmr_data = [all_ates[k] for k in all_ates]
linear_data = [linear_df[f'{e}_s1'].values for e in ['POR', 'PIPW', 'PDR']]
all_data = nmmr_data + linear_data
all_labels = [k.replace(' ', '\n') for k in all_ates] + ['Linear\nPOR', 'Linear\nPIPW', 'Linear\nPDR']
all_colors = colors_4 + ['#9C27B0'] * 3

bp = ax.boxplot(all_data, labels=all_labels, patch_artist=True, widths=0.5, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markersize=5))
for patch, c in zip(bp['boxes'], all_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(y=TRUE_ATE, color='red', linestyle='--', linewidth=2, label='True ATE=2')
ax.set_ylabel('ATE Estimate')
ax.set_title('NMMR (20 trials) vs Linear (100 trials)')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# 右: |Bias| 柱状图
ax = axes[1]
x = range(len(comp_df))
ax.bar(x, comp_df['Bias'].abs(), color=all_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(comp_df['Method'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('|Bias|')
ax.set_title('Absolute Bias Comparison')
ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('NMMR Standalone Training vs Linear Model (Scenario 1)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_nmmr_vs_linear.png')
plt.close()
print("\n[图4] NMMR vs Linear 横向比较已保存")

# ============================================================
# Part 4: 双重稳健性假设检验 + 与 Linear 横向比较
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 双重稳健性分析与假设检验")
print("=" * 60)

dr_tests = []
for df, stat_name in [(dr_u_df, 'DR U-stat'), (dr_v_df, 'DR V-stat')]:
    for col in ['PIPW', 'POR', 'PDR']:
        r = ttest(df[col].values, f'{stat_name} {col}')
        dr_tests.append(r)
        print(f"  {r['name']}: Mean={r['mean']:.4f}±{r['std']:.4f}, "
              f"Bias={r['bias']:.4f}, p={r['p']:.4f} -> {r['sig']}")

# Linear S1 作为对照
linear_tests = []
for est in ['POR', 'PIPW', 'PDR']:
    r = ttest(linear_df[f'{est}_s1'].values, f'Linear {est}')
    linear_tests.append(r)

all_dr_tests = dr_tests + linear_tests

# --- 图5: DR 假设检验森林图 ---
fig, ax = plt.subplots(figsize=(10, 7))
for i, r in enumerate(all_dr_tests):
    pc = 'green' if r['p'] > 0.05 else 'red'
    ax.errorbar(r['mean'], i, xerr=[[r['mean']-r['ci_low']], [r['ci_high']-r['mean']]],
                fmt='o', color=pc, capsize=5, markersize=8, linewidth=2)
    sig_star = '' if r['p'] > 0.05 else ('*' if r['p'] > 0.01 else ('**' if r['p'] > 0.001 else '***'))
    ax.text(r['ci_high'] + 0.003, i, f'p={r["p"]:.4f} {sig_star}', va='center', fontsize=9)

# 分隔线
ax.axhline(y=5.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=8.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(-0.02, 2.5, 'DR U-stat', transform=ax.get_yaxis_transform(),
        ha='right', fontsize=9, color='gray', style='italic')
ax.text(-0.02, 7, 'DR V-stat', transform=ax.get_yaxis_transform(),
        ha='right', fontsize=9, color='gray', style='italic')
ax.text(-0.02, 10, 'Linear', transform=ax.get_yaxis_transform(),
        ha='right', fontsize=9, color='gray', style='italic')

ax.axvline(x=TRUE_ATE, color='red', linestyle='--', linewidth=2, alpha=0.7, label='True ATE=2')
ax.set_yticks(range(len(all_dr_tests)))
ax.set_yticklabels([r['name'] for r in all_dr_tests])
ax.set_xlabel('ATE Estimate (95% CI)')
ax.set_title('Double Robustness Hypothesis Test: H₀: ATE = 2\n(Green = Cannot reject, Red = Reject)', fontsize=13)
ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_dr_hypothesis_test.png')
plt.close()

# --- 图6: DR vs Linear 箱线图横向比较 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
dr_colors = ['#E91E63', '#3F51B5', '#009688']

titles = ['DR (U-statistic)', 'DR (V-statistic)', 'Linear (Scenario 1)']
datasets = [
    [dr_u_df['PIPW'], dr_u_df['POR'], dr_u_df['PDR']],
    [dr_v_df['PIPW'], dr_v_df['POR'], dr_v_df['PDR']],
    [linear_df['PIPW_s1'], linear_df['POR_s1'], linear_df['PDR_s1']],
]

for ax, data, title in zip(axes, datasets, titles):
    bp = ax.boxplot(data, labels=['PIPW', 'POR', 'PDR'], patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    for patch, c in zip(bp['boxes'], dr_colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(y=TRUE_ATE, color='red', linestyle='--', linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('ATE Estimate')
    ax.grid(True, axis='y', alpha=0.3)
    # 标注 mean 和 bias
    for j, col_data in enumerate(data):
        m = col_data.mean()
        ax.text(j+1, ax.get_ylim()[0] + 0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                f'μ={m:.3f}\nbias={m-TRUE_ATE:.3f}', ha='center', fontsize=8, color='darkblue')

plt.suptitle('Double Robustness: NMMR vs Linear (100 trials, True ATE=2)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_dr_vs_linear.png')
plt.close()

print("\n[图5-6] 双重稳健性分析图表已保存")

# ============================================================
# 汇总统计表
# ============================================================
print("\n" + "=" * 60)
print("汇总统计表")
print("=" * 60)

summary_rows = []
for r in nmmr_tests:
    summary_rows.append({'Method': r['name'], 'Source': 'NMMR', 'n': 20,
        'Mean': r['mean'], 'Std': r['std'], 'Bias': r['bias'],
        't-stat': r['t'], 'p-value': r['p']})
for r in dr_tests:
    summary_rows.append({'Method': r['name'], 'Source': 'DR', 'n': 100,
        'Mean': r['mean'], 'Std': r['std'], 'Bias': r['bias'],
        't-stat': r['t'], 'p-value': r['p']})
for r in linear_tests:
    summary_rows.append({'Method': r['name'], 'Source': 'Linear', 'n': 100,
        'Mean': r['mean'], 'Std': r['std'], 'Bias': r['bias'],
        't-stat': r['t'], 'p-value': r['p']})

summary = pd.DataFrame(summary_rows)
summary.to_csv(f'{OUTPUT_DIR}/summary_statistics.csv', index=False, float_format='%.6f')
print(summary.to_string(index=False, float_format='%.4f'))

print(f"\n{'='*60}")
print("生成图表:")
print("  01_nmmr_boxplot.png        - NMMR 重复实验箱线图")
print("  02_nmmr_fold_variance.png  - NMMR 折叠间方差")
print("  03_nmmr_hypothesis_test.png - NMMR 假设检验")
print("  04_nmmr_vs_linear.png      - NMMR vs Linear 横向比较")
print("  05_dr_hypothesis_test.png  - DR 假设检验")
print("  06_dr_vs_linear.png        - DR vs Linear 横向比较")
print("  summary_statistics.csv     - 汇总统计表")
print(f"{'='*60}")
