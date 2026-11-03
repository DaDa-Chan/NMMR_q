# HPO 服务器运行与结果使用指南

本文给出 SGD 数据上 H-U、H-V、Q-U、Q-V 的服务器运行流程，包括环境检查、smoke test、正式 HPO、结果检查、candidate 晋升和 standalone 使用。

## 1. 流程总览

```text
同步代码 → 创建环境 → 确认 CUDA → smoke test
→ Round 1 Optuna → Round 2 多 seed/K-fold 复评
→ one-SE 选择 → 独立 pilot-confirm
→ 人工检查 candidate → 复制到 configs/sgd_frozen
→ standalone 实验
```

HPO 不会自动写入 `configs/sgd_frozen/`。

## 2. Conda 环境

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

## 3. 日志设置

```bash
mkdir -p server_logs
set -o pipefail
```

`pipefail` 防止 Python 失败时被 `tee` 的成功退出状态掩盖。每个新 shell/tmux 会话都要重新设置。

## 4. Smoke test

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

smoke test 只检查代码路径、CUDA、数值稳定性和大致耗时，不用于确定正式参数。

主要结果：

```text
tune_results/q_u_smoke_server_v2/
├── search_config_snapshot.json
├── optuna_studies/q_u_round1.db
├── q_u_round1_trials.csv
├── q_u_round2_finalists.csv
├── q_u_pilot_confirm_report.json
└── candidates/q_u_candidate_config.json
```

## 5. 正式 HPO 命令模板

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

替换关系：

| 方法 | `METHOD` | `RESULTS_DIR` | `LOG_FILE` |
|---|---|---|---|
| Q-U | `Q-U` | `tune_results/q_u_formal_v2` | `server_logs/q_u_formal_v2.log` |
| Q-V | `Q-V` | `tune_results/q_v_formal_v2` | `server_logs/q_v_formal_v2.log` |
| H-U | `H-U` | `tune_results/h_u_formal_v2` | `server_logs/h_u_formal_v2.log` |
| H-V | `H-V` | `tune_results/h_v_formal_v2` | `server_logs/h_v_formal_v2.log` |

例如 Q-U：

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
  --results_dir tune_results/q_u_formal_v2 \
  2>&1 | tee server_logs/q_u_formal_v2.log
```

长任务建议运行在 tmux：

```bash
tmux new -s q_u_hpo
```

退出但保持运行：`Ctrl-b`，再按 `d`。重新进入：

```bash
tmux attach -t q_u_hpo
```

## 6. Pilot 数据位置

上述正式设置对应：

```text
data/pilot/sgd_v1/scenario_1/seed_20261103/
├── select_n5000.csv
├── confirm_n5000.csv
└── manifest.json
```

四种方法使用相同设置时会复用这些数据。普通恢复运行不要使用 `--force-regenerate`。

## 7. 如何查看结果

以 Q-U 为例：

- `q_u_round1_trials.csv`：trial 参数、每折矩风险、均值/波动、训练诊断和时间；
- `q_u_round2_finalists.csv`：top-k 在多个 init seed 和 folds 下的复评；
- `q_u_pilot_confirm_report.json`：ATE、每折 ATE、moment risk、checkpoint、预测有限性和 Q 分布诊断；
- `candidates/q_u_candidate_config.json`：最终 candidate 及 confirm 状态；
- `optuna_studies/q_u_round1.db`：可追加的 Round 1 study。

训练时间只作诊断，不参与 Round 1、Round 2 或 confirm 的 passed/failed。

快速查看 candidate：

```bash
python -c "import json; p='tune_results/q_u_formal_v2/candidates/q_u_candidate_config.json'; d=json.load(open(p)); print(d['method_id']); print(d['confirm_status']); print(d['confirm_failure_reasons'])"
```

只有 `confirm_status == "passed"` 且失败原因为空的 candidate 才能进入人工晋升。

## 8. Candidate 晋升

```bash
mkdir -p configs/sgd_frozen
cp tune_results/q_u_formal_v2/candidates/q_u_candidate_config.json configs/sgd_frozen/q_u_frozen_config.json
cp tune_results/q_v_formal_v2/candidates/q_v_candidate_config.json configs/sgd_frozen/q_v_frozen_config.json
cp tune_results/h_u_formal_v2/candidates/h_u_candidate_config.json configs/sgd_frozen/h_u_frozen_config.json
cp tune_results/h_v_formal_v2/candidates/h_v_candidate_config.json configs/sgd_frozen/h_v_frozen_config.json
```

复制前检查目标文件，避免误覆盖。不要编辑 candidate 后再复制。当前 `config_hash` 用于追溯，loader 尚不重新验证 hash。

## 9. Standalone 命令

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

当前 standalone 仅支持 SGD。当前 frozen config 已接入 standalone，但尚未直接接入 `experiment.py --mode cross_fitting`。

## 10. 追加 trial 与恢复

同一搜索配置、pilot 数据和训练设置下，可以用同一 `results_dir` 追加 trial。例如原来30个，再追加20个时传：

```text
--n_trials 20
--results_dir tune_results/q_u_formal_v2
```

`--n_trials` 表示本次新增数量，不是累计上限。追加后会重新运行 Round 2 和 confirm，因此不要根据已经看过的 confirm 结果决定是否继续追加。

以下情况必须使用新的 `results_dir`：

- 搜索 config 有任何变化；
- pilot-select 数据改变；
- folds、epochs、patience、split 设置改变；
- coarse、refined、formal search 需要隔离。

Round 1 使用 SQLite 保存 trial，可以恢复；Round 2 当前没有断点续跑，中断后需要重新开始 Round 2。

## 11. 常见错误

### CUDA 不可用

确认当前 Conda 环境、NVIDIA 驱动和 CUDA PyTorch。不要用 CPU 启动正式规模。

### CUDA OOM

先停止其他 GPU 任务；一张 GPU 不要同时运行多个正式方法。若必须改变 batch size 或网络范围，使用新的 `results_dir`。

### `NoFiniteRound1TrialError`

Round 1 没有有效 trial，不会进入 Round 2，也不会生成 candidate。检查日志中的 NaN/Inf、checkpoint、处理组和超参数失败原因。

### `NoFiniteRound2CandidateError`

top-k candidate 在 Round 2 全部失败，不会运行 confirm。检查 `*_round2_finalists.csv` 和日志。

### confirm failed

candidate 仍会保存，但不得晋升。检查 `*_pilot_confirm_report.json` 和 `confirm_failure_reasons`，不要临时放宽 confirm 标准。

### 搜索 config snapshot 不一致

不要覆盖旧目录；改用新的版本目录，例如：

```text
tune_results/q_u_formal_v3
```

## 12. 检查清单

运行前：

- [ ] 已同步最新代码；
- [ ] 已激活正确的 Conda 环境；
- [ ] CUDA 检测为 True；
- [ ] smoke test 已成功；
- [ ] 正式搜索设置已经冻结；
- [ ] 每个方法使用独立的新 `results_dir`；
- [ ] 长任务运行在 tmux/screen/调度系统中。

运行后：

- [ ] Round 1 有有限 trial；
- [ ] Round 2 有有限 candidate；
- [ ] method ID 正确；
- [ ] confirm status 为 passed；
- [ ] confirm failure reasons 为空；
- [ ] checkpoint、ATE、moment risk 和预测有限性正常；
- [ ] 通过的 candidate 已复制到正确的 frozen 文件；
- [ ] smoke/failed candidate 没有被晋升。
