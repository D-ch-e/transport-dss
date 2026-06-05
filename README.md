# Transport DSS - Explainable AI Transportation Decision Support System

可解釋交通決策支援系統。

## 系統簡介

Transport DSS（Explainable AI Decision Support System）是一套針對北北基桃景點交通情境設計的決策支援工具。使用者從 10 處景點輸入起訖地、人數、行李量、預算與時間需求後，系統會推薦最適合的交通方式，並透過可解釋 AI 技術說明推薦依據。

## 技術架構

```text
使用者輸入（Streamlit UI）
        ↓
輸入驗證（input_validator.py）
        ↓
資料整合（api_client.py ← Google Maps API / 靜態 CSV fallback）
        ↓
特徵計算（feature_engineer.py → cost / time / convenience）
        ↓
決策模型（scoring_model.py + decision_tree.py）
        ↓
XAI 解釋（xai_report.py → SHAP + 決策路徑 + 反事實）
        ↓
結果輸出（Streamlit UI）
```

---

## 專案結構

```text
xai-dss/
├── data/
│   ├── api_client.py          # 外部 API 串接 + 快取
│   ├── input_validator.py     # 使用者輸入驗證
│   └── static/
│       ├── geocoding_attractions.csv # 北北基桃 10 處景點座標
│       ├── fare_table.csv            # legacy 靜態票價與時間參考表
│       └── city_hub_map.json         # legacy 城市名 → 交通樞紐座標對照表
│
├── features/
│   └── feature_engineer.py   # cost / time / convenience 特徵計算
│
├── models/
│   ├── scoring_model.py       # 加權評分模型
│   └── decision_tree.py       # 決策樹模型
│
├── xai/
│   └── xai_report.py          # 解釋報告
│
├── ui/
│   └── app.py                 # 主介面
│
├── tests/
│   └── tests_placeholder.py
│
├── .gitignore
├── .gitattributes
├── README.md
└── requirements.txt
```

## 安裝方式

```text
bash
pip install -r requirements.txt
```

---

## 執行方式

```text
bash
streamlit run ui/app.py
```
