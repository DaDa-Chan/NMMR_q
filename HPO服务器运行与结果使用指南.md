# HPO 服务器运行与结果使用指南

本文给出 SGD 数据上 H-U、H-V、Q-U、Q-V 的服务器运行流程：环境检查、smoke test、正式 HPO（单样本量模式与多样本量共享配置模式）、结果检查、candidate 晋升、standalone 使用，以及追加 trial 和常见错误处理。

## 1. 流程总览

```text
同步代码 → 创建环境 → 确认 CUDA → smoke test
→ Round 1 Optuna → Round 2 多 seed 复评
→ one-SE 选择 → 独立 pilot-confirm
→ 人工检查 candidate → 复制到 configs/sgd_frozen
→ standalone / 正式实验
```

HPO 不会自动写入 `configs/sgd_frozen/`，晋升始终是人工复制。

两种模式共用同一个入口 `scripts/run_pilot_protocol.py`：

| 模式 | 触发方式 | 用途 |
|---|---|---|
| 单样本量模式 | 默认 | 一份 pilot-select / pilot-confirm 数据（单个 n）上选出一份配置 |
| 多样本量共享配置模式 | 同时传 `--round2_sample_sizes` 与 `--pilot_confirm_sample_sizes` | 选出**一份**配置，共同用于 n = 500 / 1000 / 2000 / 5000，使学习曲线只反映样本量变化，不混入超参数变化 |

## 2. 训练核与评分核

- **训练核**（训练损失、early-stopping 验证）：带宽 = 该 fold gradient-train 数据的中位数启发式 × 候选自己的 `kernel_bandwidth_multiplier`。该倍数在搜索空间里，只作为训练超参数。
- **评分核**（Round 1、Round 2、pilot-confirm、P0 的 `held_out_moment_risk`、standalone 的 `test_error`）：带宽 = 同一中位数启发式 × 1，对所有候选相同，因此分数可直接比较。H 的核输入为 $(A,Z,X)$，Q 的为 $(W,X)$。
- 中位数带宽只由 gradient-train 数据计算，不使用 early-stop、score、eval 或 test 数据。
- 评分规则版本为 `unified_v_statistic_moment_risk_fixed_reference_bandwidth_v2`。按旧规则（`..._v1`）创建的 Optuna study 不能继续追加 trial；要重新 HPO 请使用新的 `results_dir`，旧 study 不会被删除或覆盖。
- 已有的 `configs/sgd_frozen/*.json` 是按旧评分规则选出的，不会被自动改写；读取时 `scoring_kernel_rule` 记为 legacy。新的 candidate 会记录 `scoring_kernel_rule = "median heuristic fitted on gradient_train, multiplier fixed at 1"`。

## 3. Conda 环境

```bash
conda create -n nmmr_q python=3.11 pip -y
conda activate nmmr_q
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

确认 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

4090 服务器上应看到 `torch.cuda.is_available() == True`。

## 4. 日志设置

```bash
mkdir -p server_logs
set -o pipefail
```

`pipefail` 防止 Python 失败时被 `tee` 的成功退出状态掩盖。每个新 shell/tmux 会话都要重新设置。

## 5. Smoke test

smoke test 只检查代码路径、CUDA、数值稳定性和大致耗时，不用于确定正式参数。

### 5.1 单样本量模式

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
  --scenario 1 \
  --base_seed 20261103 \
  --data_dir data/pilot \
  --results_dir tune_results/q_u_smoke_server_v2 \
  2>&1 | tee server_logs/q_u_smoke_server_v2.log
```

### 5.2 多样本量共享配置模式

用很小的样本量走通完整流程（Round 1 → 多样本量 Round 2 → 多样本量 confirm）：

```bash
python scripts/run_pilot_protocol.py \
  --methods H-U \
  --n_pilot_select 100 \
  --round2_sample_sizes 100,150,200,250 \
  --pilot_confirm_sample_sizes 100,150,200,250 \
  --n_trials 3 --top_k 2 \
  --round2_init_seeds 11,22 \
  --n_epochs 20 --n_splits 3 --patience 5 \
  --scenario 1 --base_seed 20261107 \
  --data_dir data/pilot_shared_smoke \
  --results_dir tune_results/shared_smoke/h_u \
  2>&1 | tee server_logs/shared_smoke_h_u.log
```

smoke 用的 `--data_dir` 和 `--results_dir` 不要与正式运行共用。

### 5.3 smoke 的输出结构

```text
RESULTS_DIR/
├── search_config_snapshot.json
├── optuna_studies/h_u_round1.db
├── h_u_round1_trials.csv
├── h_u_round2_finalists.csv
├── h_u_pilot_confirm_report.json
└── candidates/h_u_candidate_config.json
```

## 6. 正式 HPO

### 6.1 单样本量模式命令模板

正式规模建议每次只运行一个方法：

```bash
python scripts/run_pilot_protocol.py \
  --methods METHOD \
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
  --results_dir RESULTS_DIR \
  2>&1 | tee LOG_FILE
```

| 方法 | `METHOD` | `RESULTS_DIR` | `LOG_FILE` |
|---|---|---|---|
| Q-U | `Q-U` | `tune_results/q_u_formal_v2` | `server_logs/q_u_formal_v2.log` |
| Q-V | `Q-V` | `tune_results/q_v_formal_v2` | `server_logs/q_v_formal_v2.log` |
| H-U | `H-U` | `tune_results/h_u_formal_v2` | `server_logs/h_u_formal_v2.log` |
| H-V | `H-V` | `tune_results/h_v_formal_v2` | `server_logs/h_v_formal_v2.log` |

### 6.2 多样本量共享配置模式

**设计要点**

- **数据角色**：Round 1 用 `round1_n1000`；Round 2 用 `round2_n{500,1000,2000,5000}`；confirm 用 `confirm_n{500,1000,2000,5000}`。每份数据的 `data_seed` 由 `(base_seed, role, n)` 确定性派生，互相独立。
- **Round 1**：只在 `--n_pilot_select`（建议 1000）大小的一份数据上做 Optuna 初筛，取 top-k。
- **Round 2**：每个入围候选在 4 个样本量 × 5 个初始化种子 × 5 折上复评，同一 `(n, fold)` 上所有候选共享数据与分折。
  - 每个 `(n, 种子)` 内先按 held-out 样本量加权平均各折分数；
  - 再对种子取平均，得到每个候选在各 n 下的平均风险，每个 n 取全程有效候选的中位数作为尺度 `B_n`；
  - 每个种子的综合分数 = 各 n 的「该 n 的分数 / `B_n`」的平均；
  - `value` = 综合分数的种子均值，`SE` = 种子级综合分数的标准差 / √种子数；
  - 任一 `(n, 种子, 折)` 失败的候选，`value` 记为 +inf，且不参与 `B_n`。
  - 然后沿用同一个 one-SE 规则（同时保存纯最小值候选供审计）。SE 只反映初始化波动，不包含数据抽样波动，one-SE 是模型选择启发式，不是泛化风险的置信区间。
- **confirm**：对唯一的候选，在 4 个样本量各自独立的 confirm 数据上各跑一次完整 cross-fitting；任一样本量不通过则 `confirm_status=failed`。
- **只用固定参考核下的 held-out 矩风险选参**；真桥 MSE、POR/PIPW/PDR 与真值的距离只做冻结后诊断，不用于选择或替换 candidate。

**命令模板**（四种方法各自独立的 `RESULTS_DIR`）：

```bash
python scripts/run_pilot_protocol.py \
  --methods METHOD \
  --n_pilot_select 1000 \
  --round2_sample_sizes 500,1000,2000,5000 \
  --pilot_confirm_sample_sizes 500,1000,2000,5000 \
  --n_trials 40 \
  --top_k 5 \
  --round2_init_seeds 11,22,33,44,55 \
  --n_epochs 300 \
  --n_splits 5 \
  --patience 30 \
  --scenario 1 \
  --base_seed 20261107 \
  --data_dir data/pilot_shared_v2 \
  --results_dir RESULTS_DIR \
  2>&1 | tee LOG_FILE
```

| 方法 | `METHOD` | `RESULTS_DIR` | `LOG_FILE` |
|---|---|---|---|
| H-U | `H-U` | `tune_results/shared_v2/h_u` | `server_logs/shared_v2_h_u.log` |
| H-V | `H-V` | `tune_results/shared_v2/h_v` | `server_logs/shared_v2_h_v.log` |
| Q-U | `Q-U` | `tune_results/shared_v2/q_u` | `server_logs/shared_v2_q_u.log` |
| Q-V | `Q-V` | `tune_results/shared_v2/q_v` | `server_logs/shared_v2_q_v.log` |

注意：

- `--round2_sample_sizes` 与 `--pilot_confirm_sample_sizes` 必须同时给出；共享模式不允许 `--force-regenerate`，要换数据请换 `--data_dir` 或 `--base_seed`。
- 未显式给出 `--round2_init_seeds` 时，共享模式默认 5 个种子（`11,22,33,44,55`），单样本量模式默认 `11,22`。
- 新的 HPO 必须使用全新的 `--data_dir` 与 `--results_dir`，不要覆盖或混用旧的 `data/pilot/`、旧 study 和旧 candidate。
- 耗时：每个方法的 Round 2 约 `top_k × 4 个 n × 5 个种子 × 5 折` = 500 个评分单元，大 n 与小 batch 下单个方法可能需要数小时；四个方法请分别放在不同的 tmux 窗口或不同 GPU 上，同一张 GPU 不要同时跑多个正式方法。
- `top_k` 在运行前确定。看到结果后临时改大 `top_k` 会让选择过程依赖已看过的结果，应作为新一轮 HPO 重新运行。

**搜索空间**（`configs/sgd/nmmr_*/`）：

| 参数 | H-U / H-V | Q-U / Q-V |
|---|---|---|
| `batch_size` | `{32, 64, 128}` | `{32, 64, 128, 256, 500}` |
| `kernel_bandwidth_multiplier`（仅训练核） | `{0.25, 0.5, 1, 2}` | `{0.5, 1, 2, 4}` |
| `network_depth` | `{2, 3, 4, 5}` | 同左 |
| `network_width` | `{32, 64, 128}` | 同左 |
| `learning_rate` | `[1e-4, 5e-3]`，log | 同左 |
| `l2_penalty` | `[1e-7, 1e-3]`，log | 同左 |

固定：`max_epochs=300`、`patience=30`、`scheduler_patience=10`、`scheduler_factor=0.5`、`scheduler_min_lr=1e-6`。H 的 batch 范围收窄是因为大 batch 在小样本下每个 epoch 梯度步数过少、频繁跑满 `max_epochs`；Q 在原有 256/500 附近已能正常收敛，因此保留大 batch，并加入小 batch 让 Round 2 自行比较。

### 6.3 长任务与 tmux

```bash
tmux new -s h_u_hpo
```

退出但保持运行：`Ctrl-b`，再按 `d`。重新进入：

```bash
tmux attach -t h_u_hpo
```

## 7. Pilot 数据位置

单样本量模式（上述 5000 设置）：

```text
data/pilot/sgd_v1/scenario_1/seed_20261103/
├── select_n5000.csv
├── confirm_n5000.csv
└── manifest.json
```

多样本量共享配置模式：

```text
data/pilot_shared_v2/sgd_shared_v2/scenario_1/seed_20261107/
├── round1_n1000.csv
├── round2_n500.csv   round2_n1000.csv   round2_n2000.csv   round2_n5000.csv
├── confirm_n500.csv  confirm_n1000.csv  confirm_n2000.csv  confirm_n5000.csv
└── manifest.json
```

`manifest.json` 记录每份数据的 role、n、`data_seed`、`seed_scheme` 和 SHA-256。相同设置下再次运行会复用这些文件，并校验文件与 manifest 一致；不一致时直接报错，不会静默重生成。四种方法使用相同设置时共用这些数据。单样本量模式的普通恢复运行不要使用 `--force-regenerate`。

## 8. 如何查看结果

以 H-U 为例（单样本量模式文件名相同，位置在对应的 `RESULTS_DIR` 下）：

- `h_u_round1_trials.csv`：trial 参数、每折矩风险、均值/波动、训练诊断和时间；
- `h_u_round2_finalists.csv`：top-k 的复评记录。
  - 单样本量模式：每行一个 `(候选, 种子, 折)`。
  - 共享模式：每行一个 `(候选, n, 种子, 折)`，另含 `sample_size`、`data_seed`、`data_sha256`、`raw_score`、`n_score`、`scale_B_n`、`normalized_score`、`candidate_round2_value`、`candidate_round2_se`；
- `h_u_pilot_confirm_report.json`：ATE、每折 ATE、moment risk、checkpoint、预测有限性和 Q 分布诊断。共享模式下四个样本量的报告放在 `per_sample_size` 下，顶层的四项检查要求在每个 n 上都成立；
- `candidates/h_u_candidate_config.json`：最终 candidate 及 confirm 状态；
- `optuna_studies/h_u_round1.db`：可追加的 Round 1 study。

共享模式的 candidate 还记录 `round1_sample_size`、`round2_sample_sizes`、`round2_score_normalization`、`pilot_confirm_sample_sizes`。

训练时间、跑满 epoch 的比例和 Q 的分布诊断只作参考，不参与 Round 1、Round 2 或 confirm 的 passed/failed。

快速查看 candidate：

```bash
python -c "import json; p='tune_results/shared_v2/h_u/candidates/h_u_candidate_config.json'; d=json.load(open(p)); print(d['method_id']); print(d['confirm_status']); print(d['confirm_failure_reasons']); print(d['round2_sample_sizes'], d['pilot_confirm_sample_sizes'])"
```

只有 `confirm_status == "passed"` 且失败原因为空的 candidate 才能进入人工晋升。confirm 通过后也不要据此换成纯最小值候选、第二名候选或手工指定的参数。

## 9. Candidate 晋升

以多样本量共享配置模式为例：

```bash
mkdir -p configs/sgd_frozen
cp tune_results/shared_v2/q_u/candidates/q_u_candidate_config.json configs/sgd_frozen/q_u_frozen_config.json
cp tune_results/shared_v2/q_v/candidates/q_v_candidate_config.json configs/sgd_frozen/q_v_frozen_config.json
cp tune_results/shared_v2/h_u/candidates/h_u_candidate_config.json configs/sgd_frozen/h_u_frozen_config.json
cp tune_results/shared_v2/h_v/candidates/h_v_candidate_config.json configs/sgd_frozen/h_v_frozen_config.json
```

复制前检查目标文件，避免误覆盖；不要编辑 candidate 后再复制。`config_hash` 用于追溯，loader 目前不重新验证 hash。

## 10. Standalone 命令

Q-U：

```bash
python src/experiment.py \
  --mode traditional \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_u_sgd.json \
  --frozen_config_path configs/sgd_frozen/q_u_frozen_config.json \
  --n_seeds 1 \
  --log_folder train_results/standalone \
  2>&1 | tee server_logs/q_u_standalone.log
```

其他方法替换：

| 方法 | model | 数据 config | frozen config |
|---|---|---|---|
| Q-V | `nmmr_q` | `configs/sgd/nmmr_q/nmmr_q_v_sgd.json` | `configs/sgd_frozen/q_v_frozen_config.json` |
| H-U | `nmmr_h` | `configs/sgd/nmmr_h/nmmr_h_u_sgd.json` | `configs/sgd_frozen/h_u_frozen_config.json` |
| H-V | `nmmr_h` | `configs/sgd/nmmr_h/nmmr_h_v_sgd.json` | `configs/sgd_frozen/h_v_frozen_config.json` |

Q 返回 PIPW，H 返回 POR；共同返回 `test_error`、`n_test`、`total_runtime_seconds` 和训练诊断。默认不保存模型，确需保存时增加：

```bash
--save_model --model_dir saved_models
```

当前 standalone 仅支持 SGD；frozen config 也被 `double_rob.py` 的 SGD 共用 protocol 读取，但尚未直接接入 `experiment.py --mode cross_fitting`。

## 11. 追加 trial 与恢复

同一搜索配置、pilot 数据和训练设置下，可以用同一 `results_dir` 追加 trial。例如原来 30 个，再追加 20 个时传：

```text
--n_trials 20
--results_dir tune_results/q_u_formal_v2
```

`--n_trials` 表示本次新增数量，不是累计上限。追加后会重新运行 Round 2 和 confirm，因此不要根据已经看过的 confirm 结果决定是否继续追加。

以下情况必须使用新的 `results_dir`（共享模式同时要换 `--data_dir`）：

- 搜索 config 有任何变化；
- Round 1 的 pilot-select 数据改变；
- folds、epochs、patience、split 设置改变；
- 评分规则版本不同（例如要在旧规则下创建的 study 上继续）；
- coarse、refined、formal search 需要隔离。

Round 1 使用 SQLite 保存 trial，可以恢复；Round 2 目前没有断点续跑，中断后需要重新开始 Round 2。

## 12. 常见错误

### CUDA 不可用

确认当前 Conda 环境、NVIDIA 驱动和 CUDA PyTorch。不要用 CPU 启动正式规模。

### CUDA OOM

先停止其他 GPU 任务；一张 GPU 不要同时运行多个正式方法。若必须改变 batch size 或网络范围，使用新的 `results_dir`。

### `NoFiniteRound1TrialError`

Round 1 没有有效 trial，不会进入 Round 2，也不会生成 candidate。检查日志中的 NaN/Inf、checkpoint、处理组和超参数失败原因。

### `NoFiniteRound2CandidateError`

top-k candidate 在 Round 2 全部失败，不会运行 confirm。共享模式下只要某个候选在任一 `(n, 种子, 折)` 上失败，该候选就记为失败。检查 `*_round2_finalists.csv` 和日志。

### confirm failed

candidate 仍会保存，但不得晋升。检查 `*_pilot_confirm_report.json` 和 `confirm_failure_reasons`（共享模式看 `per_sample_size` 下哪个 n 出了问题），不要临时放宽 confirm 标准。

### 搜索 config snapshot 不一致

不要覆盖旧目录；改用新的版本目录，例如：

```text
tune_results/shared_v3/h_u
```

### study 的评分规则不匹配

报错信息包含 `round1_scoring_rule` 时，说明该 `results_dir` 里的 study 是用另一版评分规则创建的。不要继续追加；换新的 `results_dir` 重新运行。

### 数据文件与 manifest 不一致

报错 `inconsistent with manifest.json` 或 `exists but ... has no record` 时，说明数据目录里的文件被改动过或残缺。不要强行覆盖；换新的 `--data_dir`，或手工核对后清理该目录。

## 13. 检查清单

运行前：

- [ ] 已同步最新代码；
- [ ] 已激活正确的 Conda 环境；
- [ ] CUDA 检测为 True；
- [ ] smoke test 已成功；
- [ ] 正式搜索设置已经冻结；
- [ ] 每个方法使用独立的新 `results_dir`（共享模式同时使用新的 `--data_dir`）；
- [ ] 共享模式下 `--round2_init_seeds` 为 5 个种子；
- [ ] 长任务运行在 tmux/screen/调度系统中。

运行后：

- [ ] Round 1 有有限 trial；
- [ ] Round 2 有有限 candidate；
- [ ] method ID 正确；
- [ ] confirm status 为 passed，failure reasons 为空（共享模式下每个样本量都通过）；
- [ ] checkpoint、ATE、moment risk 和预测有限性正常；
- [ ] 通过的 candidate 已复制到正确的 frozen 文件；
- [ ] smoke/failed candidate 没有被晋升。
