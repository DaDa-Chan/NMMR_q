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
│   │   └── LINEAR/               #   线性基线模型
│   ├── data/ate/                 #   数据生成与加载
│   │   ├── sgd_pv.py             #     合成数据生成（真实 ATE = 2.0）
│   │   ├── data_class.py         #     SGDDataset / RHCDataset / BootstrapDataset
│   │   └── rhc_preprocess.py     #     RHC 数据预处理
│   └── utils.py                  #   工具函数
├── scripts/                      # 数据分析与可视化脚本
│   ├── analysis.py               #   统计分析与假设检验
│   ├── generate_figures.py       #   生成报告用图表（7 张图 + 统计汇总表）
│   ├── parse_train.py            #   解析训练结果日志 → CSV
│   └── parse_tune.py             #   解析调参结果日志 → CSV
├── configs/                      # JSON 配置文件
│   ├── sgd/                      #   SGD 数据集配置（nmmr_q / nmmr_h）
│   └── rhc/                      #   RHC 数据集配置（nmmr_q / nmmr_h）
├── data/                         # 输入数据
│   ├── SGD/                      #   合成数据
│   └── right_heart_catheterization/  # RHC 观测数据
├── summary/                      # 实验结果汇总
│   ├── *.csv                     #   汇总 CSV 数据文件
│   └── figures/                  #   生成的图表（PNG + PDF）
├── predicts/                     # 模型预测输出
├── train_results/                # 训练日志
├── tune_results/                 # 调参日志
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

项目采用**两阶段工作流**：先用 Optuna 搜索最优超参数，再用固定参数进行正式实验。

### 阶段一：超参数优化 (HPO)

使用带搜索空间的配置文件，通过 K 折交叉验证寻找最优超参数：

```bash
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd.json \
  --dump_folder results
```

### 阶段二：固定参数正式实验

将 HPO 找到的最优参数回填到 `_fixed.json` 配置中：

```bash
# 单模型实验
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd_fixed.json \
  --dump_folder results
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

配置文件位于 `configs/<dataset>/<model>/`，分为两类：

- **搜索配置**（如 `nmmr_q_v_sgd.json`）：超参数用 dict 声明搜索范围，用于 HPO
- **固定配置**（如 `nmmr_q_v_sgd_fixed.json`）：超参数为标量值，用于正式实验

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
