# 生成模型訓練資料與模型訓練、測試

## 簡介

本專案資料夾(ML_project)為利用 Google Maps API 蒐集真實交通路線資料，並生成模擬乘客情境資料，建立交通工具推薦模型，的測試Colab執行檔。

系統主要流程如下：

```text
Google Maps API
       ↓
路線資料蒐集
       ↓
資料前處理與特徵整理
       ↓
乘客情境資料生成
       ↓
交通評分系統
       ↓
Decision Tree 模型訓練
       ↓
交通方式推薦
```

---

# 專案目錄結構

請先確認 Google Drive 中具有以下資料夾結構：

```text
ML_project/
│
├── code/
│
├── content/
│   ├── data_frommap/
│   └── ...
│
└── model/
```
[Google Drive 連結](https://drive.google.com/drive/folders/1OE42tzgXiJbVzUeXDQRQWMZpZRJRT7AC?usp=sharing)

---

# 執行環境

建議使用 Google Colab 執行。

# Step 1：蒐集交通路線資料

## Notebook 位置

```text
ML_project/content/data_frommap/
└── 期末專題_路線蒐集(訓練資料).ipynb
```

## 設定 Google Maps API Key

在 Google Colab 左側工具列中：

```text
Secrets（鑰匙圖示）
↓
新增 Secret
↓
名稱：
GOOGLE_MAPS_API_KEY

值：
你的 Google Maps API Key
```

## 執行方式

完成 API 設定後，依序執行 Notebook 中所有程式區塊。

執行完成後將產生：

```text
taiwan_route_dataset.csv
```

此資料集包含：

* 起點與終點資訊
* 路線距離
* 旅行時間
* 步行時間
* 轉乘次數
* 交通工具類型
* 交通費用資訊

作為後續模型訓練的基礎資料來源。

---

# Step 2：生成乘客情境資料

## Notebook 位置

```text
ML_project/code/
└── traffic system (real static data version).ipynb
```

---

## Section 1：Prepare the Train Data

首先執行：

```python
# Prepare_the_Train_Data.py
```

此步驟將：

* 讀取路線資料集
* 建立模型訓練所需格式
* 整理交通路線特徵

---

## Section 2：Generate Synthetic Passenger Scenario Data

接著執行：

```python
# Generate_Synthetic_Scenario_Data_real.py
```

此步驟將隨機產生大量乘客情境資料，例如：

* 出行人數
* 行李數量
* 時間偏好
* 預算偏好
* 便利性偏好

並透過評分系統標記最佳交通方案。

---

## 輸出資料

執行完成後將產生：

```text
synthetic_real_model_data_read.csv
```

供人工閱讀與檢查資料內容。

以及：

```text
synthetic_real_model_data_ml.csv
```

供機器學習模型訓練使用。

---

# Step 3：交通評分系統測試

位置：

```text
traffic system (real static data version).ipynb
```

執行區塊：

```python
# Section 3 : Score Ranking System
```

## 功能說明

此部分為交通方案評分系統。

系統會根據：

* 交通時間
* 交通成本
* 步行時間
* 轉乘次數
* 使用者偏好

計算每條候選路線的綜合評分。

分數越高代表越符合該乘客情境需求。

此區塊主要用於：

* 驗證評分邏輯是否合理
* 觀察最佳方案是否符合預期
* 作為後續模型標籤生成的參考

---

# Step 4：Decision Tree 模型訓練

位置：

```text
traffic system (real static data version).ipynb
```

執行區塊：

```python
# Section 4 : Decision Tree System
```

執行：

```python
# Decision_Tree_System.py
```

## 功能說明

利用：

```text
synthetic_real_model_data_ml.csv
```

訓練交通推薦模型。

模型將學習：

* 使用者特徵
* 路線特徵
* 最佳交通方式

三者之間的關聯性。

---

## 輸出模型

訓練完成後會產生：

```text
model/
├── decision_tree_model.pkl
└── feature_columns.pkl
```

### decision_tree_model.pkl

訓練完成的 Decision Tree 分類模型。

### feature_columns.pkl

記錄模型訓練所使用的特徵欄位，供推論階段保持資料格式一致。

---

# Step 5：使用者推薦測試

位置：

```text
traffic system (real static data version).ipynb
```

執行區塊：

```python
# Section 5 : User Input
```

執行：

```python
# user_inference.py
```

## 功能說明

模擬真實使用者輸入：

* 起點
* 終點
* 出行人數
* 行李數量
* 預算偏好
* 時間偏好
* 舒適度偏好

系統將：

1. 載入訓練完成模型
2. 分析候選交通方案
3. 預測最佳交通方式
4. 輸出推薦結果

---

# 完整執行流程

請依照以下順序執行：

```text
Step 1
蒐集交通路線資料
↓
產生 taiwan_route_dataset.csv

Step 2
生成乘客情境資料
↓
產生 synthetic_real_model_data_read.csv
產生 synthetic_real_model_data_ml.csv

Step 3
交通評分系統測試

Step 4
Decision Tree 模型訓練
↓
產生 decision_tree_model.pkl
產生 feature_columns.pkl

Step 5
使用者推薦測試
↓
輸出推薦交通方式
```

---

# 注意事項

1. 使用路線蒐集功能前，需自行申請 Google Maps API Key。

2. 若重新生成交通路線資料或乘客情境資料，建議重新訓練模型，以維持模型與資料一致性。

3. 執行使用者推薦測試前，請確認以下模型檔案已成功產生：

```text
model/
├── decision_tree_model.pkl
└── feature_columns.pkl
```

4. 建議使用 Google Colab 執行，以避免本機環境套件版本不一致造成錯誤。
