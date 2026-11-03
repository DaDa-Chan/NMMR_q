# NMMR-Q: 基于神经最大矩约束学习处理混杂桥函数及代理双稳健因果效应估计

[English](README_EN.md) | 中文

## 概述

本项目提出 **NMMR-Q**，将神经最大矩约束（NMMR）方法应用于处理混杂桥函数 $q$ 的学习，并与已有的 NMMR-H 结合实现代理双稳健（PDR）估计。

**核心贡献：**

1. **NMMR-Q 方法**：推导适配于 $q$ 的残差构造、核空间选择和双网络 softplus 架构，实现代理逆概率加权（PIPW）估计量
2. **代理双稳健估计**：将 NMMR-Q 与 NMMR-H 结合，在神经 KMMR 框架内首次实现完整的 PDR 估计管线，使 PIPW、POR、PDR 三种估计量可在统一框架内计算

## 项目结构

```
NMMR_q/
├── src/                          # 核心源代码
│   ├── experiment.py             #   主实验入口（单模型训练 + 评估）
│   ├── models/
│   │   ├── NMMRq/                #   NMMR-Q：模型、损失、训练器、实验
│   │   ├── NMMRh/                #   NMMR-H：模型、损失、训练器、实验
│   │   ├── hpo_protocol.py        #   SGD pilot HPO 协议与产物管理
│   │   └── LINEAR/               #   线性基线模型
│   ├── data/ate/                 #   数据生成与加载
│   │   ├── sgd_pv.py             #     合成数据生成（真实 ATE = 2.0）
│   │   ├── data_class.py         #     SGDDataset / RHCDataset / BootstrapDataset
│   │   └── rhc_preprocess.py     #     RHC 数据预处理
│   └── utils.py                  #   工具函数
├── scripts/                      # HPO、数据分析与可视化脚本
│   ├── run_pilot_protocol.py     #   SGD HPO 主入口
│   ├── analysis.py               #   统计分析与假设检验
│   ├── generate_figures.py       #   生成报告用图表（7 张图 + 统计汇总表）
│   ├── parse_train.py            #   解析训练结果日志 → CSV
│   └── parse_tune.py             #   解析调参结果日志 → CSV
├── configs/                      # JSON 配置文件
│   ├── sgd/                      #   SGD 数据集配置（nmmr_q / nmmr_h）
│   ├── sgd_frozen/               #   人工确认后供正式实验使用的参数产物
│   └── rhc/                      #   RHC 数据集配置（nmmr_q / nmmr_h）
├── data/                         # 输入数据
│   ├── pilot/                    #   按 DGP/scenario/seed/样本量保存的 HPO pilot 数据
│   ├── SGD/                      #   合成数据
│   └── right_heart_catheterization/  # RHC 观测数据
├── summary/                      # 实验结果汇总
│   ├── *.csv                     #   汇总 CSV 数据文件
│   └── figures/                  #   生成的图表（PNG + PDF）
├── predicts/                     # 模型预测输出
├── train_results/                # 训练日志
├── tune_results/                 # HPO study、trial 表、Round 2 与 confirm 报告
├── report/                       # LaTeX 报告
│   ├── report.tex                #   英文报告
│   ├── report_cn.tex             #   中文报告
│   └── references.bib            #   参考文献
├── figures/                      # 早期分析图表
├── double_rob.py                 # 双稳健估计实验入口
├── main.py                       # 项目主入口
└── requirements.txt              # Python 依赖
```

## 环境准备

**要求**：Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

主要依赖：PyTorch 2.2、scikit-learn 1.5、Optuna 4.6、TensorBoard 2.16。

## 数据准备

### 合成数据 (SGD)

将 `sgd_train.csv` 与 `sgd_cf.csv` 放入 `data/SGD/`。如需重新生成：

```bash
python src/data/ate/sgd_pv.py
```

支持 4 种场景（控制代理变量的非线性程度），真实 ATE = 2.0。

### RHC 真实数据

将以下文件放入 `data/right_heart_catheterization/`：

- `rhc_train.csv`、`rhc_val.csv`、`rhc_test.csv`
- `RHC_X_allfeatures_list.csv`、`RHC_X_significantfeatures_list.csv`

## 实验工作流

项目采用**调参与正式实验分离**的工作流。带搜索范围的 config 只能用于 HPO；`src/experiment.py` 仅接受已解析为固定标量的参数，不会再内联启动 Optuna。

## SGD HPO 调参

### 1. 入口与支持的方法

SGD 的 HPO 统一从以下脚本启动：

```bash
python scripts/run_pilot_protocol.py --help
```

可以分别调整四种方法：

| method ID | 桥函数 | 训练统计量 | 主要效应估计量 |
|---|---|---|---|
| `H-U` | $h(A,W,X)$ | U-statistic | POR |
| `H-V` | $h(A,W,X)$ | V-statistic | POR |
| `Q-U` | $q(Z,A,X)$ | U-statistic | PIPW |
| `Q-V` | $q(Z,A,X)$ | V-statistic | PIPW |

`--methods Q-U` 只运行 Q-U；`--methods Q-U,Q-V` 运行两种 Q 方法；不传 `--methods` 时按顺序运行 H-U、H-V、Q-U、Q-V。

### 2. HPO 使用的搜索 config

脚本按 method ID 读取固定路径：

```text
H-U -> configs/sgd/nmmr_h/nmmr_h_u_sgd.json
H-V -> configs/sgd/nmmr_h/nmmr_h_v_sgd.json
Q-U -> configs/sgd/nmmr_q/nmmr_q_u_sgd.json
Q-V -> configs/sgd/nmmr_q/nmmr_q_v_sgd.json
```

搜索参数以字典形式声明，例如：

```json
"learning_rate": {
  "type": "float",
  "low": 0.0001,
  "high": 0.005,
  "log": true
},
"network_width": {
  "type": "categorical",
  "choices": [32, 64, 128]
}
```

修改搜索范围时直接修改对应的现有 config，但必须为新范围使用新的 `--results_dir`。当前脚本不会根据新建的任意 config 文件名自动切换搜索配置。

### 3. 调参的完整顺序

```text
读取方法的搜索 config
        ↓
生成/复用 pilot-select 与 pilot-confirm
        ↓
Round 1：Optuna TPE 初筛
        ↓
取有限 held-out moment risk 最好的 top-k
        ↓
Round 2：top-k × 多个 init seed × K-fold 重新训练
        ↓
one-standard-error rule 选择一个 provisional candidate
        ↓
在独立 pilot-confirm 上进行一次 cross-fitting 确认
        ↓
通过：写入 --config_dir
失败：只在 --results_dir 保存 provisional config 和失败报告
```

#### 3.1 pilot-select 与 pilot-confirm

- `pilot-select` 用于 Round 1 和 Round 2 的参数选择。
- `pilot-confirm` 与 pilot-select 使用不同的 data seed，只用于最终确认。
- confirm 不应被反复用来比较多个 candidate，否则它会变成新的调参集。

数据按 DGP、scenario、base seed、role 和样本量保存：

```text
data/pilot/
└── sgd_v1/
    └── scenario_1/
        └── seed_20261103/
            ├── select_n5000.csv
            ├── select_n10000.csv
            ├── confirm_n5000.csv
            └── manifest.json
```

请求 `select=10000` 不会覆盖已有的 `select_n5000.csv`。`manifest.json` 记录 role、样本量、data seed、文件名和 SHA-256，用于确认恢复运行时读取的仍是同一份数据。

#### 3.2 Round 1

每个 Optuna trial 对应一组超参数。每组参数在 pilot-select 上做 K-fold 评估，并使用独立 held-out fold 的 V-statistic moment risk 统一评分。

U/V 是训练目标的区别；为了让 U/V 可比，HPO 排名使用相同的 held-out V-statistic score。有 NaN/Inf、无有效 checkpoint、Q 明显数值爆炸或任一 fold 失败的 trial 记为失败，不进入有限 trial 排名。

如果本次新增的 trial（以及旧 study 中已有的 trial）全部失败，代码会在排名/打印"最佳参数"之前就抛出 `NoFiniteRound1TrialError` 并打印各失败原因的计数：不会进入 Round 2，不会跑 pilot-confirm，也不会生成任何 candidate；`run_pilot_protocol.py` 会把该方法报告为 `round1_failed` 并继续处理 `--methods` 里的其他方法。

#### 3.3 Round 2

Round 1 中最好的 `top_k` 组参数会用多个神经网络初始化 seed 重新训练：

$$N_{\text{Round 2 fits}}=\text{top\_k}\times |\text{init seeds}|\times \text{n\_splits}.$$

例如 `top_k=5`、3个 init seed、5-fold 时，每个方法需要 $5\times3\times5=75$ 次 Round 2 训练。

同一候选只要有任意一个 (seed, fold) 触发失败条件，该候选的分数就整体记为 `+inf`，不会用剩下跑成功的 fold 顶上。如果 `top_k` 个候选在 Round 2 后全部是 `+inf`，代码抛出 `NoFiniteRound2CandidateError`（与 Round 1 失败是不同的异常类型），`run_pilot_protocol.py` 报告为 `round2_failed`，同样不生成 candidate、不跑 confirm。

最终使用 one-standard-error rule：先找平均矩风险最小的 candidate，再在其 one-SE 范围内优先选择更简单的网络。

#### 3.4 pilot-confirm

Round 2 选中的唯一 candidate 会在从未参与 HPO 的 pilot-confirm 上运行一次 cross-fitting，输出：

- ATE 与每折 ATE；
- held-out moment risk；
- 每折 best epoch、stop reason 和时间；
- Q 的分位数及极端值比例；
- scenario 1 下的真 H/Q 桥 MSE（仅用于诊断）。

当前代码还将时间、跑满 epoch 比例和 Q>100 比例用作 confirm 硬门槛；这些项后续将精简为诊断输出而非通过条件。在修改前，阅读 confirm 失败原因时需要注意。

### 4. 命令示例

#### 4.1 单方法 smoke test

先用小数据、少量 trial 确认端到端流程可以运行：

```bash
python scripts/run_pilot_protocol.py \
  --methods Q-U \
  --n_pilot_select 300 \
  --n_pilot_confirm 300 \
  --n_trials 3 \
  --top_k 1 \
  --round2_init_seeds 11 \
  --n_epochs 30 \
  --n_splits 3 \
  --patience 5 \
  --data_dir data/pilot \
  --results_dir tune_results/q_u_smoke_v1 \
  --config_dir tune_results/q_u_smoke_v1/confirmed_configs
```

smoke test 只用于检查代码路径、数值稳定性和大致耗时，不用于生成正式参数。

#### 4.2 Q-U 正式 HPO 示例

```bash
python scripts/run_pilot_protocol.py \
  --methods Q-U \
  --n_pilot_select 5000 \
  --n_pilot_confirm 5000 \
  --n_trials 30 \
  --top_k 5 \
  --round2_init_seeds 11,22,33 \
  --n_epochs 300 \
  --n_splits 5 \
  --patience 30 \
  --scenario 1 \
  --base_seed 20261103 \
  --data_dir data/pilot \
  --results_dir tune_results/q_u_formal_v1 \
  --config_dir tune_results/q_u_formal_v1/confirmed_configs
```

这里故意将 `--config_dir` 放在当次 `results_dir` 内，避免尚未人工比较的 candidate 直接覆盖 `configs/sgd_frozen/` 中的正式 config。

#### 4.3 相同 study 追加 trial

如果搜索范围、pilot 数据和训练设置都没有变化，重复使用同一 `results_dir`：

```bash
python scripts/run_pilot_protocol.py \
  --methods Q-U \
  --n_pilot_select 5000 \
  --n_pilot_confirm 5000 \
  --n_trials 20 \
  --top_k 5 \
  --round2_init_seeds 11,22,33 \
  --n_epochs 300 \
  --n_splits 5 \
  --patience 30 \
  --base_seed 20261103 \
  --data_dir data/pilot \
  --results_dir tune_results/q_u_formal_v1 \
  --config_dir tune_results/q_u_formal_v1/confirmed_configs
```

`--n_trials 20` 表示“本次再追加20个”。如果原 study 已有30个 trial，运行后累计50个；原有30个不会被删除。

#### 4.4 修改搜索范围

修改对应的 `configs/sgd/...json` 后，必须使用新目录：

```bash
--results_dir tune_results/q_u_refined_v1 \
--config_dir tune_results/q_u_refined_v1/confirmed_configs
```

不同搜索范围的 candidate 默认不合并。coarse study 用于找合理范围，refined/formal study 用于产生最终 candidate。

### 5. 命令行参数

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `--methods` | 四种方法 | 逗号分隔的 method ID，例如 `Q-U` 或 `Q-U,Q-V` |
| `--n_pilot_select` | `300` | Round 1/2 使用的 pilot-select 样本量 |
| `--n_pilot_confirm` | `300` | 独立 pilot-confirm 样本量 |
| `--n_trials` | `3` | **本次追加**的 Optuna trial 数，不是累计上限 |
| `--top_k` | `2` | Round 1 进入 Round 2 的 candidate 数 |
| `--round2_init_seeds` | `11,22` | Round 2 重复训练的初始化 seed |
| `--n_epochs` | `15` | 单次训练的最大 epoch，覆盖搜索 config 中的同名值 |
| `--n_splits` | `3` | Round 1/2 和 confirm 的折数 |
| `--patience` | `5` | early stopping 容忍的无改善 epoch 数 |
| `--scenario` | `1` | SGD DGP scenario |
| `--base_seed` | `20261103` | 派生 pilot data、split 和 tuning seed 的基础 seed |
| `--data_dir` | `data/pilot` | pilot 数据根目录 |
| `--results_dir` | `tune_results` | Optuna DB、Round 1/2 表和 confirm 报告的输出目录 |
| `--config_dir` | `configs/sgd_frozen` | confirm 通过后 frozen config 的写入目录；建议调参时指向当次 results 目录 |
| `--force-regenerate` | 关闭 | 只重新生成当前请求的 select/confirm 样本量，不影响其他样本量 |

`--force-regenerate` 不应用作普通的恢复运行选项。如果旧数据与 manifest 一致，默认直接复用即可。

### 6. Study 的恢复和隔离规则

一个 study 是某个方法的一本 Optuna 调参记录，其中包含多个 trial。

| 操作 | 当前行为 |
|---|---|
| 同方法、同 pilot-select、同搜索空间、同 `results_dir` | 加载旧 study 并追加 trial |
| 搜索空间改变，仍使用旧 `results_dir` | 拒绝混合并报错 |
| 搜索空间改变，使用新 `results_dir` | 创建新 study |
| H-U/H-V/Q-U/Q-V 使用同一 `results_dir` | 每种方法使用各自的 SQLite DB |

当前 study 身份会检查 method、pilot-select SHA-256、搜索空间 SHA-256、`n_epochs`、`n_splits`、`patience`、`early_stopping_min_delta`、`split_seed` 和 Round 1 评分规则。任一项变化时，程序会拒绝将新 trial 追加到旧 study，并提示使用新 `results_dir`。

### 7. 生成的文件

假设：

```bash
--methods Q-U \
--results_dir tune_results/q_u_formal_v1 \
--config_dir tune_results/q_u_formal_v1/confirmed_configs
```

则主要产物为：

```text
tune_results/q_u_formal_v1/
├── optuna_studies/
│   └── q_u_round1.db
├── q_u_round1_trials.csv
├── q_u_round2_finalists.csv
├── q_u_pilot_confirm_report.json
├── q_u_provisional_config.json       # 仅 confirm 失败时生成
└── confirmed_configs/
    └── q_u_frozen_config.json        # 仅 confirm 通过时生成
```

| 文件 | 内容 |
|---|---|
| `q_u_round1.db` | Optuna SQLite study，保留所有历史 trial，用于追加/恢复 Round 1 |
| `q_u_round1_trials.csv` | Round 1 的 trial/fold 超参数、seed、矩风险、训练结果和失败原因 |
| `q_u_round2_finalists.csv` | top-k candidate 在多 init seed、多 fold 下的重新评估结果 |
| `q_u_pilot_confirm_report.json` | 独立 confirm 的 ATE、moment risk、Q 诊断、真桥 MSE 和通过/失败原因 |
| `q_u_provisional_config.json` | confirm 失败的候选参数，不是正式 config |
| `q_u_frozen_config.json` | confirm 通过的 `FrozenConfig` 产物 |

#### 覆盖注意

当前代码在 confirm 通过后会以普通写入方式保存到 `--config_dir` 中的固定文件名。同一 method 再次通过 confirm 时，会覆盖该 `config_dir` 中的同名文件。

在 candidate/正式 config 完全分离前，建议每次搜索范围使用独立的 `results_dir` 和其内部的 `config_dir`，不要在试验阶段使用默认 `configs/sgd_frozen`。

### 8. 如何挑选最终参数

单个 study 内由程序按下列规则自动选择：

1. 排除失败或非有限 trial。
2. Round 1 取 held-out moment risk 最好的 top-k。
3. Round 2 在相同 folds 和 init seeds 下重新比较 top-k。
4. 使用 one-SE rule 在表现接近时优先选更简单的网络。
5. 只将该唯一 candidate 送入 pilot-confirm。

如果同一方法存在多个正式 study，只在 pilot 数据、folds、epochs、early stopping、Round 2 seeds 和评分规则都相同时比较它们的 Round 2 结果。主要看：

1. Round 2 平均 held-out moment risk（越小越好）；
2. 跨 fold/seed 标准差（越小越稳定）；
3. one-SE 范围内的网络复杂度（越简单越好）；
4. 总训练时间（次要指标）。

pilot-confirm 只是最终的 passed/failed 检查，不用 confirm ATE 误差、真桥 MSE 或 confirm moment risk 在多个 candidate 之间继续挑参数。H-U、H-V、Q-U、Q-V 是后续横向比较的四种方法，每种方法各自保留一份最终 config，不是从四种方法中只选一种。

### 9. HPO FrozenConfig 与正式实验的当前接口

HPO 生成的 `q_u_frozen_config.json` 是 `FrozenConfig` 产物，它包含选定参数、seed、Round 1/2 信息、status 和 `config_hash`。

它与旧的 `configs/sgd/.../*_fixed.json` 不是同一外层 JSON 格式：

- HPO `FrozenConfig` 是扁平的调参产物；
- `experiment.py --config_path` 当前仍要求包含 `data_configs` 和 `train_params` 的实验 config。

因此，当前不要直接将 HPO `q_u_frozen_config.json` 传给 `experiment.py --config_path`。HPO 内部的 pilot-confirm 会通过 `FrozenConfig.to_train_params()` 正确解析它；正式 loader 与手动 promotion 流程仍需统一。

在该接口完成前，现有固定参数实验仍使用旧格式的 `*_fixed.json`：

```bash
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd_fixed.json
```

### 双稳健估计 (PDR)

`double_rob.py` 同时运行 NMMR-Q 和 NMMR-H，计算 PIPW、POR、PDR 三种估计量：

```bash
# SGD：100 次试验，每次重新生成数据
python double_rob.py --dataset_name sgd --trials 100 --n_samples 5000

# RHC Bootstrap：有放回抽样，构造 95% 置信区间
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode bootstrap

# RHC Repeated：消除折划分噪声
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode repeated
```

### 数据分析与可视化

分析脚本位于 `scripts/` 目录，所有路径自动定位到项目根目录，可从任意位置运行：

```bash
# 生成报告用图表（输出到 summary/figures/）
python scripts/generate_figures.py

# 统计分析与假设检验（输出到 figures/）
python scripts/analysis.py

# 解析训练/调参日志为 CSV
python scripts/parse_train.py
python scripts/parse_tune.py
```

## 模型架构

### NMMR-Q（处理混杂桥函数） → PIPW 估计量

| 特征 | 说明 |
|------|------|
| 网络结构 | 双网络：$q_0(Z,X)$ / $q_1(Z,X)$ 分别学习 |
| 激活函数 | 隐层 LeakyReLU，输出层 softplus（保证 $q > 0$） |
| 核空间 | 高斯 RBF 核定义在 $(W, X)$ 空间上 |
| 损失函数 | 核最大矩约束（KMMR），支持 U/V 统计量 |

### NMMR-H（结果混杂桥函数） → POR 估计量

| 特征 | 说明 |
|------|------|
| 网络结构 | 单网络：$h(A, W, X)$ |
| 核空间 | 高斯 RBF 核定义在 $(A, Z, X)$ 空间上 |
| ATE | $\hat{\tau} = n^{-1}\sum_i [h(1,W_i,X_i) - h(0,W_i,X_i)]$ |

### PDR（代理双稳健）

$$\hat{\tau}_{\text{PDR}} = \frac{1}{n}\sum_i \left[(2A_i-1)\,q_{A_i}(Z_i,X_i)(Y_i - h(A_i,W_i,X_i)) + h(1,W_i,X_i) - h(0,W_i,X_i)\right]$$

当 $q$ 或 $h$ 任一正确指定时保持一致性。

## 配置文件

当前项目中有三种相关文件：

- **搜索配置**（如 `nmmr_q_v_sgd.json`）：超参数用 dict 声明搜索范围，用于 HPO
- **旧格式固定实验配置**（如 `nmmr_q_v_sgd_fixed.json`）：包含 `data_configs` 和标量化的 `train_params`，当前可传给 `experiment.py`
- **HPO FrozenConfig 产物**（如 `q_v_frozen_config.json`）：包含 HPO 选定参数、seed、selection provenance 和校验哈希；当前尚不能直接传给 `experiment.py --config_path`

| 配置 | 数据集 | 模型 | 损失 |
|------|--------|------|------|
| `nmmr_q_u_*.json` | SGD / RHC | NMMR-Q | U 统计量 |
| `nmmr_q_v_*.json` | SGD / RHC | NMMR-Q | V 统计量 |
| `nmmr_h_u_*.json` | SGD / RHC | NMMR-H | U 统计量 |
| `nmmr_h_v_*.json` | SGD / RHC | NMMR-H | V 统计量 |

## TensorBoard

```bash
tensorboard --logdir results/sgd/nmmr_q --port 6006
```

在配置中设置 `"log_metrics": "True"` 启用训练日志。

## 主要参考文献

- Kompa, Bellot & Gasquez (2022). *Deep Learning Methods for Proximal Inference via Maximum Moment Restriction.* NeurIPS.
- Cui et al. (2024). *Semiparametric Proximal Causal Inference.* JASA.
- Tchetgen Tchetgen et al. (2020). *An Introduction to Proximal Causal Learning.* arXiv.
