# NMMR-q: Neural Maximum Moment Restriction for Causal Inference

基于神经网络最大矩限制方法 (NMMR) 的因果推断框架，用于从观测数据中估计平均处理效应 (ATE)。项目实现了 NMMR-q（Q 函数）和 NMMR-h（H 函数）两种估计器，并支持近端双重稳健 (PDR) 估计。

## 项目结构

```
NMMR_q/
├── src/
│   ├── experiment.py              # 主实验入口（单模型训练 + 评估）
│   ├── models/
│   │   ├── NMMRq/                 # Q 函数：模型、损失、训练器、实验
│   │   ├── NMMRh/                 # H 函数：模型、损失、训练器、实验
│   │   └── LINEAR/                # 线性基线模型
│   └── data/ate/                  # 数据生成与加载
│       ├── sgd_pv.py              # 合成数据生成（真实 ATE=2.0）
│       └── data_class.py          # SGDDataset / RHCDataset / BootstrapDataset
├── configs/                       # JSON 配置文件
│   ├── sgd/                       # SGD 数据集配置
│   └── rhc/                       # RHC 数据集配置
├── double_rob.py                  # 双重稳健估计实验（PDR）
├── analysis.py                    # 统计分析脚本
├── data/                          # 输入数据
│   ├── SGD/                       # 合成数据
│   └── right_heart_catheterization/  # RHC 观测数据
├── requirements.txt               # Python 依赖
└── README.md
```

## 环境准备

**要求**：Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

主要依赖：PyTorch 2.2、scikit-learn 1.5、Optuna 4.6、TensorBoard 2.16 等。

## 数据准备

### 合成数据 (SGD)

将 `sgd_train.csv` 与 `sgd_cf.csv` 放入 `data/SGD/`。如需重新生成：

```bash
python src/data/ate/sgd_pv.py
```

支持 4 种场景（scenario 1-4），控制代理变量的非线性程度，真实 ATE = 2.0。

### RHC 观测数据

将以下文件放入 `data/right_heart_catheterization/`：
- `rhc_train.csv`、`rhc_val.csv`、`rhc_test.csv`
- `RHC_X_allfeatures_list.csv`、`RHC_X_significantfeatures_list.csv`

确保配置文件中 `data_configs.data_path` 与实际数据位置一致。

## 实验工作流

项目采用**两阶段工作流**：先用 Optuna 搜索最优超参数，再用固定参数进行正式实验。

### 阶段一：超参数优化 (HPO)

使用带搜索空间的配置文件（如 `nmmr_q_v_sgd.json`），通过 K 折交叉验证寻找最优超参数：

```bash
# SGD
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd.json \
  --log_folder results

# RHC
python src/experiment.py \
  --dataset_name rhc \
  --model_name nmmr_q \
  --config_path configs/rhc/nmmr_q/nmmr_q_v_rhc.json \
  --log_folder results
```

搜索空间配置示例（超参数用 dict 格式声明搜索范围）：

```json
{
  "train_params": {
    "n_epochs": 150,
    "batch_size": 500,
    "learning_rate": {"type": "float", "low": 9e-4, "high": 5e-3, "log": true},
    "network_width": {"type": "categorical", "choices": [45, 50, 55]},
    "l2_penalty": {"type": "float", "low": 1e-6, "high": 3e-6, "log": true},
    "hpo_n_trials": 15
  }
}
```

HPO 完成后，Optuna 会输出最优参数组合。

### 阶段二：将最优参数写入 `_fixed.json`

将 HPO 找到的最优超参数回填到对应的 `_fixed.json` 配置中。Fixed 配置的超参数是标量值（不是搜索空间），跳过 Optuna，直接用于训练：

```json
{
  "train_params": {
    "n_epochs": 150,
    "batch_size": 500,
    "learning_rate": 0.008923,
    "network_width": 55,
    "l2_penalty": 2.137e-06
  }
}
```

### 阶段三：正式实验

#### 单模型重复实验

使用 fixed 配置通过 `experiment.py` 运行（在配置中设置 `n_trials` 字段控制重复次数）：

```bash
# SGD: 每次 trial 重新生成数据 + cross-fitting
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd_fixed.json \
  --dump_folder results

# RHC: 在全量数据上 cross-fitting
python src/experiment.py \
  --dataset_name rhc \
  --model_name nmmr_q \
  --config_path configs/rhc/nmmr_q/nmmr_q_v_rhc_fixed.json \
  --dump_folder results
```

#### 双重稳健估计 (PDR)

`double_rob.py` 自动读取 `_fixed.json` 配置，同时运行 NMMR-q 和 NMMR-h，计算 PDR 估计量：

```bash
# SGD: 每次 trial 生成新数据，遍历 4 个 scenario x 2 种统计量
python double_rob.py \
  --dataset_name sgd \
  --trials 100 \
  --n_samples 2000

# RHC (repeated cross-fitting): 同一数据，不同种子控制折划分
python double_rob.py \
  --dataset_name rhc \
  --trials 20 \
  --rhc_mode repeated

# RHC (bootstrap): 每次有放回抽样，构造统计置信区间
python double_rob.py \
  --dataset_name rhc \
  --trials 200 \
  --rhc_mode bootstrap
```

PDR 估计公式：`PDR = (-1)^(1-A) * q * (Y - h_fact) + (h_1 - h_0)`

### SGD vs RHC 实验差异

| | SGD | RHC |
|---|---|---|
| 数据来源 | 每次 trial 重新生成 | 固定数据集（5735 样本） |
| 多次 trial 变异来源 | 数据采样 + 折划分 | 仅折划分（repeated）或重抽样（bootstrap） |
| repeated 模式含义 | - | 消除拆分噪声，得到稳定点估计 |
| bootstrap 模式含义 | - | 逼近采样分布，构造 95% CI |
| 有 scenario | 1-4（代理变量非线性） | 无 |

## 配置文件

配置文件位于 `configs/<dataset>/<model>/` 下，分为两类：

**搜索配置**（HPO 阶段使用）：
| 配置 | 数据集 | 模型 | 损失类型 |
|------|--------|------|----------|
| `nmmr_q_u_sgd.json` | SGD | NMMR-q | U 统计量 |
| `nmmr_q_v_sgd.json` | SGD | NMMR-q | V 统计量 |
| `nmmr_h_u_sgd.json` | SGD | NMMR-h | U 统计量 |
| `nmmr_h_v_sgd.json` | SGD | NMMR-h | V 统计量 |
| `nmmr_q_u_rhc.json` | RHC | NMMR-q | U 统计量 |
| `nmmr_q_v_rhc.json` | RHC | NMMR-q | V 统计量 |
| `nmmr_h_u_rhc.json` | RHC | NMMR-h | U 统计量 |
| `nmmr_h_v_rhc.json` | RHC | NMMR-h | V 统计量 |

**固定配置**（正式实验使用，`_fixed.json` 后缀）：

HPO 完成后将最优参数回填至此。`double_rob.py` 默认读取 fixed 配置。所有搜索配置都有对应的 fixed 版本（如 `nmmr_q_v_sgd.json` → `nmmr_q_v_sgd_fixed.json`）。

## 模型架构

### NMMR-q（Q 函数）→ PIPW 估计量

- **双网络结构**：分别为 A=0 和 A=1 构建独立网络 `q0(Z,X)` / `q1(Z,X)`
- **网络结构**：全连接网络，LeakyReLU 激活，softplus 输出层
- **损失函数**：基于高斯 RBF 核的矩限制损失，核定义在 `(W, X)` 空间上
- **核带宽**：通过中位数启发式自动估计
- **支持 U 统计量和 V 统计量两种损失变体**

### NMMR-h（H 函数）→ POR 估计量

- **单网络结构**：`h(A, W, X)` 预测结果
- **损失函数**：核定义在 `(A, Z, X)` 空间上
- **ATE 预测**：`E[h(1,W,X) - h(0,W,X)]`

### PDR（近端双重稳健）

结合 PIPW 和 POR，即使其中一个模型存在偏差，只要另一个正确指定，PDR 仍然一致。

## TensorBoard 可视化

在配置中设置 `"log_metrics": "True"` 后，训练日志会自动保存：

```bash
# 查看单次实验
tensorboard --logdir results/sgd/nmmr_q/nmmr_q_v_sgd/<timestamp>/<seed>/tensorboard_log_<seed> --port 6006

# 对比多个实验
tensorboard --logdir results/sgd/nmmr_q --port 6006
```

浏览器打开 `http://localhost:6006` 查看损失曲线和训练指标。

## 输出结构

```
results/<dataset>/<model>/<config>/<timestamp>/
├── <seed>/
│   ├── tensorboard_log_<seed>/    # TensorBoard 事件文件
│   ├── fold_1/, fold_2/, ...      # 各折结果
│   └── *.pred.txt                 # E[Y|do(A=0)], E[Y|do(A=1)], ATE
predicts/<dataset>/dr_results/     # 双重稳健实验结果 CSV
predicts/<dataset>/<model>/        # 单模型汇总预测
train_results/                     # 训练结果汇总
```
