# Transport DSS — 可解釋交通決策支援系統

針對北北基桃地區景點旅遊情境設計的交通推薦決策支援系統，整合即時 API 資料、機器學習模型與可解釋 AI 技術，協助使用者在多種交通方式中做出最適選擇。

---

## 系統功能

- **交通方式推薦**：根據起訖地、人數、行李量、預算與時間偏好，從高鐵、台鐵、客運、捷運、計程車等多種交通方式中推薦最佳選項
- **即時資料整合**：串接 Google Maps API、TDX 運輸資料、YouBike API；無 API 金鑰時自動切換為靜態資料 fallback
- **可解釋 AI（XAI）**：提供加權貢獻分解、決策樹 IF-THEN 決策路徑、SHAP 特徵推力、反事實提示
- **互動式 UI**：Streamlit 介面，支援成本/時間/便利性偏好權重調整與即時重新計算

---

## 技術架構

```text
使用者輸入（Streamlit UI）
        ↓
輸入驗證（input_validator.py）
        ↓
資料整合（api_client.py ← Google Maps / TDX / YouBike API，靜態 CSV fallback）
        ↓
特徵計算（feature_engineer.py → cost / time / convenience）
        ↓
決策模型（scoring_model.py 加權評分 + decision_tree.py 決策樹分類）
        ↓
XAI 解釋（xai/report.py → 貢獻分解 + 決策路徑 + SHAP + 反事實）
        ↓
結果輸出（Streamlit UI）
```

---

## 專案結構

```text
transport-dss/
├── config/
│   └── settings.py            # 環境變數與全域設定（API 金鑰、快取 TTL）
│
├── data/
│   ├── api_client.py          # 外部 API 串接 + 分層快取
│   ├── input_validator.py     # 使用者輸入驗證
│   ├── clients/
│   │   ├── google_client.py   # Google Maps Directions API
│   │   ├── tdx_client.py      # TDX 運輸資料平台 API
│   │   └── youbike_client.py  # YouBike 站點即時資料
│   └── static/                # 靜態參考資料（票價表、座標、路由表）
│
├── features/
│   └── feature_engineer.py    # cost / time / convenience 特徵計算與正規化
│
├── models/
│   ├── scoring_model.py       # 加權評分模型
│   └── decision_tree.py       # 決策樹分類模型（92% 準確率，29 特徵）
│
├── services/
│   ├── route_planner.py       # 多模態路線規劃主邏輯
│   ├── multimodal.py          # 多段接駁組合
│   └── transfer_planner.py    # 轉乘計算
│
├── xai/
│   └── report.py              # XAI 解釋報告（加權分解 + 決策路徑 + SHAP + 反事實）
│
├── ui/
│   └── app.py                 # Streamlit 主介面
│
├── tests/                     # 單元測試與整合測試
├── scripts/                   # 資料前處理工具腳本
├── docs/                      # 架構文件
├── .env.example               # API 金鑰設定範本
├── requirements.txt
└── pytest.ini
```

---

## 安裝與設定

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定 API 金鑰（選用）

複製範本並填入金鑰：

```bash
cp .env.example .env
```

```env
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
TDX_CLIENT_ID=your_tdx_client_id
TDX_CLIENT_SECRET=your_tdx_client_secret
```

> 無 API 金鑰時系統仍可執行，自動使用靜態 CSV 資料作為 fallback。

---

## 執行方式

```bash
streamlit run ui/app.py
```

### 執行測試

```bash
pytest
```

---

## 支援景點

北北基桃地區共 10 處景點：台北車站、台北 101、小人國、三峽老街、碧潭、陽明山竹子湖、桃園機場、基隆廟口夜市、木柵動物園、龍山寺。

---

## 使用的 API

| API                        | 用途                     | 無金鑰 fallback   |
| -------------------------- | ------------------------ | ----------------- |
| Google Maps Directions API | 即時行車時間、路線       | 靜態距離矩陣 CSV  |
| TDX 運輸資料平台           | 高鐵/台鐵/客運班表與票價 | 靜態票價 CSV      |
| YouBike API                | 站點即時可借車數         | 停用 YouBike 選項 |
