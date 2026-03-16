"""
训练结果解析脚本
功能: 读取 train_result 目录下所有 txt 文件 + 对应 JSON 配置，输出为 CSV
      每个 txt 可能包含多个 config 段，每个 config 段包含多次 trial（由 n_trials 决定）
"""
import os
import re
import csv
import json
import glob

# ============================================================
# 在这里修改输入/输出路径 、
# ============================================================
INPUT_DIR = "./train_results"          # txt 文件所在目录
OUTPUT_CSV = "./results_summary.csv"  # 输出 CSV 文件路径
CONFIG_BASE_DIR = "."                 # config 路径的根目录（configs/ 所在的父目录）
# ============================================================

# JSON 中需要跳过的 train_params 字段
SKIP_TRAIN_PARAMS = {"log_metrics", "hpo_n_trials"}


def parse_config_json(config_rel_path: str) -> dict:
    """
    读取 JSON 配置文件，提取:
      - data_configs: data_name(key), n_samples, scenario
      - train_params: 除 SKIP_TRAIN_PARAMS 外的所有字段
    """
    full_path = os.path.join(CONFIG_BASE_DIR, config_rel_path)
    if not os.path.isfile(full_path):
        print(f"配置文件不存在: {full_path}")
        return {}

    with open(full_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    params = {}

    # ---------- data_configs ----------
    data_configs = cfg.get("data_configs", {})
    if data_configs:
        data_name = list(data_configs.keys())[0]
        params["data_name"] = data_name
        data_info = data_configs[data_name]
        params["n_samples"] = data_info.get("n_samples")
        params["scenario"] = data_info.get("scenario")

    # ---------- train_params ----------
    train_params = cfg.get("train_params", {})
    for key, value in train_params.items():
        if key not in SKIP_TRAIN_PARAMS:
            params[f"tp_{key}"] = value

    return params


def parse_txt_file(filepath: str) -> list[dict]:
    """
    解析单个 txt 文件:
      1. 按 config 行分割出多个 config 段
      2. 每个 config 段内，按 "平均 ATE" 行分割出多个 trial
      3. 每个 trial 提取 fold1~5 + avg_ate，加上 trial 编号
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 按 config 行分割成段 ---
    # 找到所有 config 行的位置
    config_starts = [(m.start(), m.group().strip().rstrip("\\").strip())
                     for m in re.finditer(r"^configs/[^\n]+", content, re.MULTILINE)]

    if not config_starts:
        print(f"    ⚠️  未找到 config 行: {filepath}")
        return []

    # 把文件切成 (config_path, text_block) 对
    config_segments = []
    for i, (start, config_path) in enumerate(config_starts):
        end = config_starts[i + 1][0] if i + 1 < len(config_starts) else len(content)
        block = content[start:end]
        config_segments.append((config_path, block))

    # --- 逐段处理 ---
    all_records = []
    for config_path, block in config_segments:
        # 读取 JSON 配置
        json_params = parse_config_json(config_path)

        # 按 "平均 ATE" 行切割出每个 trial
        # 找到每个 "平均 ATE" 行的末尾位置作为 trial 的分界点
        avg_positions = [m.end() for m in re.finditer(r"平均\s*ATE\(.*\):\s*[\d.]+", block)]
        trials = []
        prev = 0
        for pos in avg_positions:
            trials.append(block[prev:pos])
            prev = pos
        trials = [t.strip() for t in trials if re.search(r"平均\s*ATE", t)]

        for trial_idx, trial_text in enumerate(trials, start=1):
            record = {"trial": trial_idx}
            record.update(json_params)

            # 提取每个 Fold 的 ATE
            for m in re.finditer(r"Fold\s+(\d+)\s+ATE:\s*([\d.]+)", trial_text):
                record[f"fold{int(m.group(1))}"] = float(m.group(2))

            # 提取平均 ATE
            avg_m = re.search(r"平均\s*ATE\(.*\):\s*([\d.]+)", trial_text)
            if avg_m:
                record["avg_ate"] = float(avg_m.group(1))

            # CF 标签
            cf_m = re.search(r"\[CF-([^\]]+)\]", trial_text)
            if cf_m:
                record["cf_label"] = cf_m.group(1)

            all_records.append(record)

    return all_records


def main():
    txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))

    if not txt_files:
        print(f"在 '{INPUT_DIR}' 目录下未找到任何 .txt 文件")
        return

    print(f"找到 {len(txt_files)} 个 txt 文件")

    all_records = []
    for filepath in txt_files:
        print(f"解析: {os.path.basename(filepath)}")
        records = parse_txt_file(filepath)
        all_records.extend(records)
        print(f"      -> {len(records)} 条 trial")

    if not all_records:
        print("未解析到任何实验记录")
        return

    # ---------- 确定列顺序 ----------
    front_cols = [
        "data_name", "n_samples", "scenario", "cf_label", "trial",
    ]
    fold_cols = ["fold1", "fold2", "fold3", "fold4", "fold5", "avg_ate"]
    tp_order = [
        "tp_model_name", "tp_loss_name",
        "tp_n_epochs", "tp_batch_size",
        "tp_patience", "tp_scheduler_patience", "tp_scheduler_factor",
        "tp_n_splits", "tp_network_depth", "tp_network_width",
        "tp_use_u_statistic", "tp_kernel_gamma",
        "tp_learning_rate", "tp_l2_penalty",
    ]

    all_keys = set()
    for r in all_records:
        all_keys.update(r.keys())

    ordered = []
    for c in front_cols + fold_cols + tp_order:
        if c in all_keys:
            ordered.append(c)
            all_keys.discard(c)
    ordered += sorted(all_keys)

    # ---------- 写入 CSV ----------
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n成功！共 {len(all_records)} 条记录 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()