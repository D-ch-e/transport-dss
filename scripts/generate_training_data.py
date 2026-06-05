"""
合成訓練資料產生腳本

用途：
以 taiwan_transport_dataset.csv（720 筆原始資料）為基底，
為每筆資料生成 N 組隨機使用者情境（人數、行李、預算、偏好權重），
產出合成資料供決策樹模型訓練使用。

產出檔案：
- data/static/synthetic/{rows}_{N}_model_data_read.csv  — 可讀版
- data/static/synthetic/{rows}_{N}_model_data_m1.csv     — One-Hot 編碼版

執行方式：
    python scripts/generate_training_data.py [--num-per-row 500]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 確保專案根目錄在 Python 路徑中
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

STATIC_DIR = os.path.join(PROJECT_ROOT, "data", "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "synthetic")


# ── 使用者情境生成 ──
def generate_context(rng: np.random.Generator) -> tuple:
    """
    隨機生成使用者情境。

    Returns
    -------
    tuple
        (people, luggage, budget, time_pref, cost_pref, conv_pref)
    """
    people = rng.integers(1, 9)  # 1~8 人

    # 人數與行李的關聯性
    if people >= 5:
        luggage = rng.integers(2, 5)  # 多人較多行李
    else:
        luggage = rng.integers(0, 3)

    budget = rng.integers(10, 1200)  # 隨機預算
    time_pref = rng.integers(1, 11)  # 時間偏好 1~10
    cost_pref = rng.integers(1, 11)  # 成本偏好 1~10
    conv_pref = rng.integers(1, 11)  # 便利性偏好 1~10

    return people, luggage, budget, time_pref, cost_pref, conv_pref


# ── 便利性調整（Colab 版本） ──
def adjust_convenience(row: pd.Series, people: int, luggage: int) -> float:
    """
    根據行李與人數調整便利性分數。

    使用連續線性懲罰，不同交通方式有不同權重。
    """
    base = row["convenience"]
    mode = row["mode"]

    # 行李懲罰
    if mode == "Bus":
        penalty = luggage * 0.5
    elif mode == "TRA":
        penalty = luggage * 0.3
    elif mode == "HSR":
        penalty = luggage * 0.25
    else:  # Car
        penalty = luggage * 0.1

    # 人數懲罰
    if mode == "Bus":
        penalty += people * 0.25
    elif mode == "TRA":
        penalty += people * 0.15
    elif mode == "HSR":
        penalty += people * 0.1
    else:  # Car（多人租車更方便）
        penalty -= people * 0.05

    return max(1, base - penalty)


# ── 成本調整 ──
def adjust_cost(row: pd.Series, people: int) -> float:
    """租車成本人均分攤，大眾運輸為票價。"""
    if row["mode"] == "Car":
        return row["cost"] / people
    else:
        return row["cost"]


# ── 正規化 ──
def normalize(series: pd.Series) -> pd.Series:
    """Min-Max 正規化。"""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(1.0, index=series.index)
    return (series - s_min) / (s_max - s_min)


# ── 權重計算 ──
def get_weights(time_pref: int, cost_pref: int, conv_pref: int) -> tuple:
    """歸一化偏好權重。"""
    total = time_pref + cost_pref + conv_pref
    return time_pref / total, cost_pref / total, conv_pref / total


# ── 分數計算 ──
def compute_score(row: pd.Series) -> float:
    """加權評分：時間與成本越低越好 → (1-x)，便利性越高越好。"""
    w_time, w_cost, w_conv = get_weights(
        row["time_pref"], row["cost_pref"], row["conv_pref"]
    )
    return (
        w_time * (1 - row["time_n"])
        + w_cost * (1 - row["cost_n"])
        + w_conv * row["conv_n"]
    )


def main(num_per_row: int = 500, seed: int = 42):
    """
    主流程：讀取原始資料 → 擴展情境 → 計算特徵 → 標註最佳 → 輸出。
    """
    # 讀取原始資料
    data_path = os.path.join(STATIC_DIR, "taiwan_transport_dataset.csv")
    if not os.path.exists(data_path):
        print(f"❌ 找不到 {data_path}")
        print("   請將 taiwan_transport_dataset.csv 放到 data/static/ 資料夾。")
        sys.exit(1)

    data = pd.read_csv(data_path)
    print(f"✅ 讀取原始資料：{len(data)} 筆")

    rng = np.random.default_rng(seed)

    # ── 擴展資料集 ──
    expanded_rows = []
    for _, row in data.iterrows():
        for _ in range(num_per_row):
            people, luggage, budget, time_pref, cost_pref, conv_pref = generate_context(rng)

            new_row = row.copy()
            new_row["people"] = people
            new_row["luggage"] = luggage
            new_row["budget"] = budget
            new_row["time_pref"] = time_pref
            new_row["cost_pref"] = cost_pref
            new_row["conv_pref"] = conv_pref

            # 調整便利性與成本
            new_row["conv_adj"] = adjust_convenience(new_row, people, luggage)
            new_row["cost"] = adjust_cost(new_row, people)

            expanded_rows.append(new_row)

    model_data = pd.DataFrame(expanded_rows)
    print(f"✅ 擴展完成：{len(model_data)} 筆合成資料")

    # ── 分組正規化 ──
    group_cols = ["origin", "destination"]
    model_data["time_n"] = model_data.groupby(group_cols)["time_min"].transform(normalize)
    model_data["cost_n"] = model_data.groupby(group_cols)["cost"].transform(normalize)
    model_data["conv_n"] = model_data.groupby(group_cols)["conv_adj"].transform(normalize)

    # ── 計算分數 ──
    model_data["score"] = model_data.apply(compute_score, axis=1)

    # ── 標註最佳選擇 ──
    model_data["label"] = "not_best"
    label_group_cols = [
        "origin", "destination", "people", "luggage",
        "budget", "time_pref", "cost_pref", "conv_pref",
    ]
    best_idx = model_data.groupby(label_group_cols)["score"].idxmax()
    model_data.loc[best_idx, "label"] = "best"

    best_count = (model_data["label"] == "best").sum()
    print(f"✅ 標註完成：{best_count} 筆 best 標籤")

    # ── 輸出 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = len(data)

    # 可讀版
    read_path = os.path.join(OUTPUT_DIR, f"{rows}_{num_per_row}_model_data_read.csv")
    model_data.to_csv(read_path, index=False)
    print(f"📄 可讀版：{read_path}")

    # One-Hot 版
    model_data_ml = pd.get_dummies(
        model_data, columns=["origin", "destination", "mode", "service"]
    )
    ml_path = os.path.join(OUTPUT_DIR, f"{rows}_{num_per_row}_model_data_m1.csv")
    model_data_ml.to_csv(ml_path, index=False)
    print(f"📄 ML 版：{ml_path}")

    print(f"\n✅ 完成！共產生 {len(model_data)} 筆合成資料。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="產生合成訓練資料")
    parser.add_argument(
        "--num-per-row", type=int, default=500,
        help="每筆原始資料的擴展數量（預設 500）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子（預設 42）",
    )
    args = parser.parse_args()
    main(num_per_row=args.num_per_row, seed=args.seed)
