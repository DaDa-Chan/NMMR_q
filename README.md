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
├── scripts/                      # HPO 与专项实验脚本
│   ├── run_pilot_protocol.py     #   SGD HPO 主入口
│   └── analyze_rhc_repeated.py   #   RHC repeated-CF 专项分析
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
├── tune_results/                 # HPO study、trial 表、Round 2、confirm 报告与 candidate
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

### CUDA 服务器检查

代码按 `CUDA > Apple MPS > CPU` 的顺序自动选择设备。在 NVIDIA 服务器上安装依赖后，先确认当前 Python 环境确实识别到了 GPU：

```bash
python -c "import torch; print('cuda_available =', torch.cuda.is_available()); print('device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

4090 服务器应输出 `cuda_available = True`，并显示对应的 NVIDIA GPU 名称。若这里是 `False`，不要直接开始正式 HPO，应先检查服务器驱动、CUDA 版 PyTorch 和当前虚拟环境。

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

项目采用**调参与正式实验分离**的工作流。带搜索范围的训练参数只用于 HPO，任何正式训练入口都不会内联启动 Optuna。HPO 产生的 candidate 经人工确认后复制到 `configs/sgd_frozen/`；`src/experiment.py --mode traditional` 分别读取数据配置和这份 frozen config（见第 9 节）。

## SGD HPO 调参

服务器环境准备、四种方法的完整命令、产物检查、candidate 晋升和 standalone 使用步骤，见本地计划目录中的 [HPO 服务器运行与结果使用指南](plan/hpo/HPO服务器运行与结果使用指南.md)。

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

搜索 config 会被逐字节保存到 `--results_dir`。只要搜索 config 的内容发生变化（包括删除或增加一个固定字段），旧目录就不能继续追加 trial；应保留旧目录用于追溯，并为新配置使用新的 `results_dir`。

### 3. 调参的完整顺序

```text
读取方法的搜索 config，原样复制一份到 --results_dir 作为快照
        ↓
生成/复用 pilot-select 与 pilot-confirm
        ↓
Round 1：Optuna TPE 初筛
        ↓
取有限 held-out moment risk 最好的 top-k
        ↓
Round 2：top-k × 多个 init seed × K-fold 重新训练
        ↓
one-standard-error rule 选择一个 candidate
        ↓
在独立 pilot-confirm 上进行一次 cross-fitting 确认
        ↓
无论通过还是失败：candidate 都保存到 --results_dir/candidates/，
confirm_status=passed|failed 和失败原因记在 candidate 里面；
代码从不写入 configs/sgd_frozen/，晋升为正式 config 是人工操作
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

U/V 是训练目标的区别；为了让 U/V 可比，HPO 排名使用相同的 held-out V-statistic score。有 NaN/Inf、无有效 checkpoint、某折缺少所需的处理组，或任一 fold 无法完成有效评分时，该 trial 记为失败，不进入有限 trial 排名。

Round 1 不使用每折训练时间作为通过/失败门槛。`train_seconds` 仍会写入 trial/fold 记录，用于估算计算成本，但服务器快慢不会改变一个统计方法是否有效。

如果本次新增的 trial（以及旧 study 中已有的 trial）全部失败，代码会在排名/打印"最佳参数"之前就抛出 `NoFiniteRound1TrialError` 并打印各失败原因的计数：不会进入 Round 2，不会跑 pilot-confirm，也不会生成任何 candidate；`run_pilot_protocol.py` 会把该方法报告为 `round1_failed` 并继续处理 `--methods` 里的其他方法。

#### 3.3 Round 2

Round 1 中最好的 `top_k` 组参数会用多个神经网络初始化 seed 重新训练：

$$N_{\text{Round 2 fits}}=\text{top\_k}\times |\text{init seeds}|\times \text{n\_splits}.$$

例如 `top_k=5`、3个 init seed、5-fold 时，每个方法需要 $5\times3\times5=75$ 次 Round 2 训练。

同一候选只要有任意一个 (seed, fold) 触发失败条件，该候选的分数就整体记为 `+inf`，不会用剩下跑成功的 fold 顶上。如果 `top_k` 个候选在 Round 2 后全部是 `+inf`，代码抛出 `NoFiniteRound2CandidateError`（与 Round 1 失败是不同的异常类型），`run_pilot_protocol.py` 报告为 `round2_failed`，同样不生成 candidate、不跑 confirm。

Round 2 同样不以训练耗时作为失败条件；每折耗时只作诊断。当前只有 Round 1 的 Optuna SQLite study 可以追加/恢复，Round 2 尚无断点续跑机制，因此正式运行建议按方法分别启动，并为每种方法保留独立的运行目录。

最终使用 one-standard-error rule：先找平均矩风险最小的 candidate，再在其 one-SE 范围内优先选择更简单的网络。

#### 3.4 pilot-confirm

Round 2 选中的唯一 candidate 会在从未参与 HPO 的 pilot-confirm 上运行一次 cross-fitting，输出：

- ATE 与每折 ATE；
- held-out moment risk；
- 每折 best epoch、stop reason 和时间；
- Q 的分位数及极端值比例；
- scenario 1 下的真 H/Q 桥 MSE（仅用于诊断）。

判断 passed/failed 只看固定的硬性检查清单，不受硬件速度影响：

1. 每一折都正常完成并恢复了最佳 checkpoint；
2. ATE 估计（以及每一折的 ATE）是有限值；
3. held-out moment risk 是有限值；
4. H/Q 的最终预测里没有 NaN/Inf。

没有单独的 Q 数值上限门槛。`q_max_abs`、Q 的分位数、`q_a{level}_prop_gt10/50/100`，连同每折/总训练时间、是否跑满 epoch，都只是诊断信息，出现在 confirm 报告里供人工查看，不参与 passed/failed 的判断。

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
  --results_dir tune_results/q_u_smoke_v1
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
  --results_dir tune_results/q_u_formal_v2
```

没有 `--config_dir`：这条命令永远不会碰 `configs/sgd_frozen/`。跑完之后去看
`tune_results/q_u_formal_v2/candidates/q_u_candidate_config.json` 里的
`confirm_status`，确认是 `passed` 后再自己手动复制到
`configs/sgd_frozen/q_u_frozen_config.json`（见第 9 节）。

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
  --results_dir tune_results/q_u_formal_v2
```

`--n_trials 20` 表示“本次再追加20个”。如果原 study 已有30个 trial，运行后累计50个；原有30个不会被删除。恢复运行时，Optuna sampler 的种子会根据已有 trial 数重新派生，不会因为复用同一个 sampler seed 而整批重放第一批的参数——但不同 seed 之间偶然抽到同一个点（例如某个可选项很少的分类参数）仍然可能发生，这不是本次修复要消除的目标。

脚本追加 Round 1 trial 后会重新执行 Round 2 和 pilot-confirm。因此，追加 trial 应视为同一次 HPO 工作尚未结束，而不是在看过 confirm 结果后反复追加、再用同一份 confirm 数据挑 candidate。正式决策时只审查最终一次运行产生的 candidate；不要根据多次 confirm 的好坏决定是否继续追加。

#### 4.4 修改搜索范围

修改对应的 `configs/sgd/...json` 后，必须使用新目录：

```bash
--results_dir tune_results/q_u_refined_v1
```

不同搜索范围的 candidate 默认不合并。coarse study 用于找合理范围，refined/formal study 用于产生最终 candidate。

#### 4.5 在 GPU 服务器上的推荐顺序

1. 提交并同步最新代码到服务器，确认工作目录没有混入旧版本配置。
2. 使用“CUDA 服务器检查”中的命令确认 PyTorch 识别到 GPU。
3. 先为单个方法运行 smoke test，检查完整流程、显存和大致耗时。
4. smoke test 使用独立目录，例如 `tune_results/q_u_smoke_server`；不要把它作为正式 study 继续追加。
5. 正式 HPO 为每个方法使用独立且全新的目录，例如 `q_u_formal_v2`、`q_v_formal_v2`。
6. 长任务建议放在 `tmux`、`screen` 或服务器作业调度系统中。Round 2 中断后需要从该方法的 Round 2 重新开始。
7. 运行结束后检查 candidate、Round 1/2 表和 confirm 报告，再人工晋升通过的 candidate。

例如服务器首次 smoke test：

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
  --results_dir tune_results/q_u_smoke_server
```

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
| `--results_dir` | `tune_results` | 这次运行的全部产物（搜索 config 快照、Optuna DB、Round 1/2 表、confirm 报告、candidate）的输出目录 |
| `--force-regenerate` | 关闭 | 只重新生成当前请求的 select/confirm 样本量，不影响其他样本量 |

没有 `--config_dir`：HPO 代码不写 `configs/sgd_frozen/`，这个目录只能由人手动维护（第 9 节）。

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
--results_dir tune_results/q_u_formal_v2
```

则主要产物为：

```text
tune_results/q_u_formal_v2/
├── search_config_snapshot.json       # 本次运行使用的搜索 config 原文
├── optuna_studies/
│   └── q_u_round1.db
├── q_u_round1_trials.csv
├── q_u_round2_finalists.csv
├── q_u_pilot_confirm_report.json
└── candidates/
    └── q_u_candidate_config.json     # 无论 confirm 通过还是失败都会生成
```

一次运行多个 `--methods` 时，`search_config_snapshot.json` 换成
`search_configs/{method}.json`（每个方法一份，例如 `search_configs/q_u.json`、
`search_configs/h_u.json`），其余文件按 method 各自独立，不受影响。

| 文件 | 内容 |
|---|---|
| `search_config_snapshot.json`（或 `search_configs/{method}.json`） | 该方法这次运行实际读取的搜索 config 原始内容，逐字节复制，用于追溯 |
| `q_u_round1.db` | Optuna SQLite study，保留所有历史 trial，用于追加/恢复 Round 1 |
| `q_u_round1_trials.csv` | Round 1 的 trial/fold 超参数、seed、矩风险、训练结果、训练时间和失败原因；训练时间不参与 pass/fail |
| `q_u_round2_finalists.csv` | top-k candidate 在多 init seed、多 fold 下的重新评估结果及诊断信息；训练时间不参与 pass/fail |
| `q_u_pilot_confirm_report.json` | 独立 confirm 的 ATE、moment risk、Q 诊断、真桥 MSE 和通过/失败原因 |
| `candidates/q_u_candidate_config.json` | 这次选中的候选参数（`FrozenConfig`），`confirm_status`/`confirm_failure_reasons` 字段记录 confirm 是否通过及原因；**无论通过还是失败都会写**，从来不是正式 config |

#### 不会自动覆盖 configs/sgd_frozen/

HPO 代码（`save_pipeline_artifacts`）只写 `--results_dir`，不接受任何指向
`configs/sgd_frozen/` 的参数，也没有任何函数会自动写入该目录。candidate
通过 confirm 之后，晋升为正式 config 永远是人工步骤：自己确认
`confirm_status=="passed"`、method 正确、config 里没有残留的搜索范围或
`hpo_n_trials`，然后手动把 `candidates/q_u_candidate_config.json` 复制成
`configs/sgd_frozen/q_u_frozen_config.json`。不同搜索范围用不同
`results_dir`，各自的 candidate 互不覆盖。

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

### 9. HPO candidate 与正式实验的接口

HPO 生成的 `candidates/q_u_candidate_config.json` 是 `FrozenConfig` 产物，它包含选定参数、seed、Round 1/2 信息、`status`、`confirm_status`/`confirm_failure_reasons` 和用于追溯的 `config_hash`。只有当你自己确认 `confirm_status=="passed"` 之后，手动把这个文件复制成 `configs/sgd_frozen/q_u_frozen_config.json`，它才算是正式 config——HPO 代码本身不会做这一步。

`src/experiment.py --mode traditional`（standalone：在单个 train/test 划分上训练一个模型，作为非 cross-fitting 的横向对比）直接读取这个扁平的 `FrozenConfig` JSON，两个配置文件各自分工：

- `--config_path`：只提供数据/DGP 配置（样本量、数据 seed、scenario），不再要求其中的 `train_params` 是标量——哪怕该文件仍带着 Optuna 搜索空间（例如复用现成的 `configs/sgd/nmmr_q/nmmr_q_u_sgd.json`）也没关系，因为这个模式根本不读取它的 `train_params`。
- `--frozen_config_path`：指向你手动放进 `configs/sgd_frozen/` 的正式 `FrozenConfig` 文件；`experiment.py` 通过 `hpo_protocol.FrozenConfig.load()` 解析，再用它自带的 `to_train_params()` 得到固定训练参数——不会在 `experiment.py` 内重新实现一套字段转换，也不会自动从 `candidates/` 选、自动跑 Optuna，或覆盖/修改这个 frozen config 文件。

使用前会做三项检查，任一不满足都会直接报错并拒绝运行：

1. `confirm_status == "passed"`；
2. `method_id` 与 `--model_name` 一致（如 `Q-*` 配 `nmmr_q`），且 `method_id` 的 U/V 后缀与文件自身的 `training_objective` 字段一致；
3. 解析出的训练参数里不能再有 Optuna 搜索空间或 `hpo_n_trials`。

```bash
python src/experiment.py \
  --mode traditional \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_u_sgd.json \
  --frozen_config_path configs/sgd_frozen/q_u_frozen_config.json
```

standalone 的职责保持精简（精简与config管理.md section 6.6）：只训练一个模型（单个 train/validation/test 划分，validation 只用于早停，test 用于最终效应估计和 moment-risk），Q 返回 `PIPW`，H 返回 `POR`，连同 `test_error`（与 HPO/confirm 相同的 held-out moment-risk 打分）和 `total_runtime_seconds`；默认不保存模型权重，传入 `--save_model` 才会写入 `--model_dir`。目前只支持 `--dataset_name sgd`——RHC 的 standalone 人口（仅 train split）与 cross-fitting 实际使用的全量 RHC 数据（train+val+test 合并）不一致，传 `--dataset_name rhc` 会直接抛 `NotImplementedError`，而不是跑出一个看似能跑、实际无法横向比较的结果。

### 双稳健估计 (PDR)

`double_rob.py` 是 P0/E1/E2 的共用入口。SGD 走新的共用 protocol
(`src/experiments/sgd_experiment_protocol.py`)：只读取 `configs/sgd_frozen/`
下已通过 confirm 的正式 FrozenConfig，不再支持旧的 `*_fixed.json`，也不在内部
运行 Optuna。目前已接入 `oracle`、`parametric_proximal`、`nmmr_u` 和
`nmmr_v`，且只启用 `S1`；LS、LS-QF 和 2SLS 尚未接入，因此下面命令是当前
已实现方法的联调命令，不是完整方法矩阵的正式 P0。RHC 旧流程未改动，继续沿用
原有命令。

```bash
# SGD 当前四种已实现方法的小规模联调（不是完整 P0）
python double_rob.py \
  --dataset_name sgd --run_stage P0 --scenarios S1 \
  --methods oracle,parametric_proximal,nmmr_u,nmmr_v \
  --sample_sizes 300 --repetitions 1 --test_size 1000 --n_splits 5 \
  --base_seed 20261103 --output_dir results/p0_s1_smoke_v1

# RHC Bootstrap：有放回抽样，构造 95% 置信区间
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode bootstrap

# RHC Repeated：消除折划分噪声
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode repeated
```

#### SGD P0/E1/E2 实验运行规则：不支持断点恢复

每次正式实验必须使用全新、空的 `--output_dir`——protocol 在目录下已有
`run_config.json` 时会直接拒绝运行（`FileExistsError`），不会尝试续跑或合并
结果。**如果某次运行中途中断或失败，请删除那个不完整的结果目录，然后用相同
参数从头重新运行**，不要复用旧目录。

这不等于"一个方法失败就整个实验失败"：单个 `(scenario, n, repetition, method)`
组合内部失败（训练异常、NaN/Inf 等）会被记录到 `common/failures.csv`，该方法
的实验正常继续跑完其余方法与重复；只有当运行本身被中断（进程被杀死、机器
崩溃等）时，才需要删除目录重新运行整个命令。

#### SGD P0/E1/E2 输出文件

```text
results/<output_dir>/
├── run_config.json
├── common/
│   ├── repetitions.csv
│   ├── folds.csv
│   ├── fold_diagnostics.csv
│   ├── failures.csv
│   └── prediction_inputs/
│       └── {scenario}_n{n}_rep{repetition:04d}.npz
├── predictions/{scenario}/n{n}/rep{repetition:04d}/
│   ├── {method}.npz
│   └── {method}__{variant}.npz
├── e1/
│   └── bridge_results.csv
└── e2/
    └── effect_results.csv
```

`common/prediction_inputs/` 每个 `(scenario,n,repetition)` 只保存一次共同的
train/test 输入（`train_y,train_a,train_z,train_w,train_x1,train_x2` 和对应
的 `test_*`），所有方法共用，只在 fold plan 成功建立后保存一次。
`predictions/` 下每个成功的 `(scenario,n,repetition,method,variant)` 保存一
份压缩 OOF/测试预测（`h_observed_oof,h0_oof,h1_oof,q_observed_oof,h0_test,
h1_test,q0_test,q1_test`）；无 variant 的方法用 `{method}.npz`，LS/LS-QF/2SLS
用 `{method}__{variant}.npz`。两类文件都用 `numpy.savez_compressed`，不保存
模型权重、DataFrame object、pickle object、`U`、逐 epoch 日志或 GPU/CPU 信
息；方法失败时不生成它的预测文件，但不影响其他方法。只用这两类 npz 文件就能
重算 POR/PIPW/PDR 和 `bridge_results.csv` 里的 H/Q 有限维测试矩
(`h_moment_l2/linf`、`q_moment_l2/linf`，固定特征见 `run_config.json` 的
`prediction_persistence` 字段)，不需要重新训练。

#### SGD P0/E1/E2 结果汇总

```bash
python scripts/summarize_sgd_results.py --results_dir results/p0_s1_v1
```

只读取已完成结果目录下的 `run_config.json`、`e1/bridge_results.csv`、
`e2/effect_results.csv`、`common/failures.csv`（不存在时按零失败处理），生成
或覆盖：

- `e1/bridge_summary.csv`：按 `run_stage,scenario,n,method,variant` 分组，对
  成功重复的桥误差、测试矩、Q 分布指标取算术平均；
- `e2/effect_summary.csv`：按 `run_stage,scenario,n,method,variant,estimator`
  （`estimator` 为 POR/PIPW/PDR 长表）给出 `mean`、`bias`、`abs_bias`、
  `sd`（`ddof=1`）、`rmse`、`mcse_bias`，以及成功重复 `total_seconds` 的
  `median_time`/`p90_time`；
- `common/time_summary.csv`：按同样的分组给出成功重复的 H/Q 训练与评估耗时
  均值，以及总耗时的均值/中位数/P90。

简要定义：`Bias = mean(估计) - true_ate`（真 ATE 来自 `scenarios.py` 的
scenario registry，不是硬编码的 2）；`abs_bias = abs(Bias)`，不是平均绝对
误差；`SD` 用 `ddof=1` 的样本标准差；`RMSE = sqrt(mean((估计-true_ate)^2))`；
`MCSE_Bias = SD/sqrt(成功重复数)`；成功重复数 `< 2` 时 `sd`/`mcse_bias` 记为
NA；`failure_rate = failures/attempts`，失败重复既不计入均值也不记为 0；某
`(scenario,n,method,variant)` 全部失败时该组仍出现在三张表里，统计量记为
NA。汇总前会检查每组的 `successes + failures` 是否等于 `run_config.json` 记
录的 `repetitions`，不满足就直接报错（`IncompleteResultsError`）而不是猜测
或补算缺失结果——按前面"不支持断点恢复"的规则删除不完整目录重新跑。

汇总程序只读取原始结果，不训练模型、不修改 `run_config.json`/原始 CSV/NPZ
预测文件；重复执行会直接覆盖旧的三个 summary 文件，不会累加重复行。正式 P0
仍必须使用全新的 `--output_dir`，汇总程序本身不提供、也不依赖任何训练恢复
机制。

### 正式 Monte Carlo 与结果汇总

旧的日志解析和硬编码分析脚本已经删除。正式基础正确性实验将直接输出结构化的
逐重复/逐折 CSV，再由新的汇总与画图入口读取，不再从终端文本反向解析结果。
详细实施顺序记录在本地的 `plan/p0_e1_e2/` 中，总计划为 `plan/实验计划.md`。

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

SGD 当前使用两类配置：

- **搜索配置**（如 `nmmr_q_v_sgd.json`）：超参数用 dict 声明搜索范围，用于 HPO；`experiment.py --mode traditional` 的 `--config_path` 也可以复用这类文件，只取其中的 `data_configs`
- **HPO candidate / FrozenConfig 产物**（如 `candidates/q_v_candidate_config.json`，人工确认通过后手动复制为 `configs/sgd_frozen/q_v_frozen_config.json`）：包含 HPO 选定参数、seed、selection provenance、`confirm_status` 和追溯哈希；standalone 已读取该格式，正式 cross-fitting 将由新的共用 protocol 读取。

SGD 的旧 `*_fixed.json` 已删除，不再作为正式配置。RHC 的 legacy fixed 配置暂时
保留，只服务尚未迁移的 RHC 旧流程。

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
