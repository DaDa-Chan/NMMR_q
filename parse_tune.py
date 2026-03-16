"""
Tune 实验结果解析脚本
功能: 读取 tune_result 目录下所有 txt 文件 + 对应 JSON 配置，输出为 CSV
      每个 trial 包含 Optuna 调参结果 + cross fitting ATE
"""
import os
import re
import csv
import json
import glob

# ============================================================
# 在这里修改输入/输出路径 
# ============================================================
INPUT_DIR = "./tune_results"           # txt 文件所在目录
OUTPUT_CSV = "./tune_summary.csv"     # 输出 CSV 文件路径
CONFIG_BASE_DIR = "."                 # config 路径的根目录（configs/ 所在的父目录）
# ============================================================

HEADER = [
    "model type", "mode", "seed", "dataset", "n_samples",
    "loss type", "epochs", "batch size", "patience", "s_patinece", "s_factor",
    "n_splits", "net_width", "net_depth", "kernel_gamma", "hpo_n_trials",
    "lr", "l2_penalty", "loss_best",
    "fold1", "fold2", "fold3", "fold4", "fold5", "ATE",
]

# loss_name 映射: JSON 里的全名 -> CSV 里的缩写
LOSS_MAP = {
    "V_statistic": "v",
    "U_statistic": "u",
    "v_statistic": "v",
    "u_statistic": "u",
}


def parse_config_json(config_rel_path: str) -> dict:
    """
    读取 JSON 配置，提取固定参数（不含 Optuna 搜索空间的参数）
    """
    full_path = os.path.join(CONFIG_BASE_DIR, config_rel_path)
    if not os.path.isfile(full_path):
        print(f"配置文件不存在: {full_path}")
        return {}

    with open(full_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    params = {}

    # --- data_configs ---
    data_configs = cfg.get("data_configs", {})
    if data_configs:
        data_name = list(data_configs.keys())[0]
        params["dataset"] = data_name
        params["n_samples"] = data_configs[data_name].get("n_samples", "")

    # --- train_params (只取固定值，跳过 dict 类型的搜索空间) ---
    tp = cfg.get("train_params", {})
    params["model type"] = tp.get("model_name", "")
    params["loss type"] = LOSS_MAP.get(tp.get("loss_name", ""), tp.get("loss_name", ""))
    params["epochs"] = tp.get("n_epochs", "")
    params["batch size"] = tp.get("batch_size", "")
    params["patience"] = tp.get("patience", "")
    params["s_patinece"] = tp.get("scheduler_patience", "")
    params["s_factor"] = tp.get("scheduler_factor", "")
    params["n_splits"] = tp.get("n_splits", "")
    params["net_depth"] = tp.get("network_depth", "")
    params["kernel_gamma"] = tp.get("kernel_gamma", "")
    params["hpo_n_trials"] = tp.get("hpo_n_trials", "")

    return params


def parse_txt_file(filepath: str) -> list[dict]:
    """
    解析单个 tune txt 文件:
      1. 按 config 行分割
      2. 每个 config 段内，按 "平均 ATE" 分割出多个 trial
      3. 每个 trial 提取: Optuna 最佳参数 + fold ATE
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 按 config 行分割 ---
    config_starts = [(m.start(), m.group().strip().rstrip("\\").strip())
                     for m in re.finditer(r"^configs/[^\n]+", content, re.MULTILINE)]

    if not config_starts:
        print(f"未找到 config 行: {filepath}")
        return []

    config_segments = []
    for i, (start, config_path) in enumerate(config_starts):
        end = config_starts[i + 1][0] if i + 1 < len(config_starts) else len(content)
        config_segments.append((config_path, content[start:end]))

    # --- 逐 config 段处理 ---
    all_records = []
    for config_path, block in config_segments:
        # 读取 JSON 固定参数
        json_params = parse_config_json(config_path)

        # 按 "平均 ATE" 切割出每个 trial
        avg_positions = [m.end() for m in re.finditer(r"平均\s*ATE\([^)]+\):\s*[\d.]+", block)]
        trials_text = []
        prev = 0
        for pos in avg_positions:
            trials_text.append(block[prev:pos])
            prev = pos
        trials_text = [t for t in trials_text if re.search(r"平均\s*ATE", t)]

        for trial_idx, trial_text in enumerate(trials_text, start=1):
            record = dict(json_params)  # 复制 JSON 固定参数

            # --- Seed ---
            seed_m = re.search(r"Seed\s+(\d+)", trial_text)
            record["seed"] = int(seed_m.group(1)) if seed_m else 42

            # --- mode (从 CF 标签提取) ---
            cf_m = re.search(r"\[CF-([^\]]+)\]", trial_text)
            record["mode"] = "cf" if cf_m else ""

            # --- Optuna 最佳参数 ---
            # 最佳 loss (如 "最佳 h-loss: 0.006594" 或 "最佳 q-loss: 0.003382")
            loss_m = re.search(r"最佳\s+\w+-loss:\s*([\d.]+)", trial_text)
            record["loss_best"] = float(loss_m.group(1)) if loss_m else ""

            # 最佳参数行: {'learning_rate': 0.001, 'network_width': 45, 'l2_penalty': 1e-06}
            params_m = re.search(r"最佳参数:\s*\{([^}]+)\}", trial_text)
            if params_m:
                params_str = params_m.group(1)
                # 提取 learning_rate
                lr_m = re.search(r"'learning_rate':\s*([\d.eE\-+]+)", params_str)
                record["lr"] = lr_m.group(1) if lr_m else ""
                # 提取 network_width
                nw_m = re.search(r"'network_width':\s*(\d+)", params_str)
                record["net_width"] = int(nw_m.group(1)) if nw_m else ""
                # 提取 l2_penalty
                l2_m = re.search(r"'l2_penalty':\s*([\d.eE\-+]+)", params_str)
                record["l2_penalty"] = l2_m.group(1) if l2_m else ""

            # --- Fold ATE ---
            for m in re.finditer(r"Fold\s+(\d+)\s+ATE:\s*([\d.]+)", trial_text):
                record[f"fold{int(m.group(1))}"] = float(m.group(2))

            # --- 平均 ATE ---
            avg_m = re.search(r"平均\s*ATE\([^)]+\):\s*([\d.]+)", trial_text)
            record["ATE"] = float(avg_m.group(1)) if avg_m else ""

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

    # --- 写入 CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n成功！共 {len(all_records)} 条记录 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()