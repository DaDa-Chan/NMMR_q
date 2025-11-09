## NMMR-q Experiments

该项目实现了论文中的 NMMR-q 神经网络，用于在合成 (SGD) 与 RHC 实验数据上估计干预效应。仓库主要包含：

- `src/models/NMMRq/`：模型、损失、训练与实验脚本
- `src/data/ate/`：数据生成/预处理脚本
- `configs/*.json`：数据与训练参数

### 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 依赖包含 PyTorch、scikit-learn、tensorboard、tqdm 等，确保使用 Python 3.10+。

### 2. 数据放置

- **SGD**：`data/SGD/` 下需要 `sgd_train.csv` 与 `sgd_cf.csv`。若需要重新生成，可运行 `src/data/ate/sgd_pv.py`（默认写入 `data/SGD/sgd_pv.csv`，按需拆分）。
- **RHC**：将 `rhc_train.csv / rhc_val.csv / rhc_test.csv` 及特征列表 `RHC_X_*features_list.csv` 放在 `data/right_heart_catheterization/`。

确保 `configs/nmmr_q_v_sgd.json` 或其他配置中的 `data_configs` 路径与实际一致。

### 3. 运行实验

核心入口为 `src/experiment.py`；需要指定数据集、模型与配置文件。示例：

```bash
# 在 SGD 数据上进行 K-fold + cross fitting（结果输出到 results/sgd/nmmr_q/<config>/）
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/nmmr_q_v_sgd.json \
  --dump_folder results

# 在 RHC 数据上进行 cross fitting
python src/experiment.py \
  --dataset_name rhc \
  --model_name nmmr_q \
  --config_path configs/nmmr_q_v_sgd.json \
  --dump_folder results
```

运行时脚本会：
1. 解析配置并设置随机种子。
2. 根据数据集选择流程：SGD 先在 `sgd_train.csv` 做 K-fold 交叉验证，再在 `sgd_cf.csv` 上 cross fitting；RHC 直接 cross fitting（可通过配置的 `cf_split`/`use_all_X` 控制）。
3. 在 `results/<dataset>/<model>/<config>/<seed>/` 下保存日志与 `*.pred.txt`（包含 `E[Y|do(A=0/1)]` 与最终 ATE）。

如需调整折数、学习率等，可在配置文件 `train_params` 中修改相应字段 (`n_splits`, `learning_rate`, `batch_size` 等)。更多实验或数据脚本可参考 `src/data/ate` 与 `src/models/NMMRq`。
