"""
M1 · API Client — 資料整合模組

功能：
- 讀取靜態 fare_table.csv 與 city_hub_map.json
- 讀取 taiwan_transport_dataset.csv（完整交通資料集）
- 讀取 Google Maps API 靜態資料：
    - geocoding_attractions.csv（北北基桃景點座標）
    - distance_matrix_driving.csv（景點間開車距離與時間）
    - directions_transit.csv（大眾運輸路線與轉乘）
- 根據起訖地查詢各交通方式的成本、時間與距離
- 統一輸出資料格式供後續模組使用
- 作為 MVP 階段的主要資料來源
"""

import json
import os

import pandas as pd

# 取得 static 資料夾路徑
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 英中交通方式對照表
MODE_EN_TO_ZH = {
    "HSR": "高鐵",
    "TRA": "台鐵",
    "Bus": "客運",
    "Car": "租車",
}

MODE_ZH_TO_EN = {v: k for k, v in MODE_EN_TO_ZH.items()}


def load_city_hub_map() -> dict:
    """讀取城市與交通樞紐對照表。"""
    path = os.path.join(_STATIC_DIR, "city_hub_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vehicle_params() -> dict:
    """讀取交通方式參數設定。"""
    path = os.path.join(_STATIC_DIR, "vehicle_params.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fare_table() -> pd.DataFrame:
    """讀取靜態票價與時間資料表。"""
    path = os.path.join(_STATIC_DIR, "fare_table.csv")
    return pd.read_csv(path)


def load_transport_dataset() -> pd.DataFrame | None:
    """
    讀取完整交通資料集（taiwan_transport_dataset.csv）。

    若檔案不存在則回傳 None。
    此資料集包含 720 筆資料，欄位：
    origin, destination, mode, service, distance_km, time_min, cost, convenience
    """
    path = os.path.join(_STATIC_DIR, "taiwan_transport_dataset.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_supported_cities() -> list[str]:
    """取得支援的城市列表。"""
    city_map = load_city_hub_map()
    return list(city_map.keys())


_OFFLINE_TABLE_CACHE: pd.DataFrame | None = None


def load_offline_route_table() -> pd.DataFrame | None:
    """
    讀取景點層級離線備援表（offline_route_table.csv）。

    由 scripts/fill_offline_route_costs.py 產生，供 ENABLE_REALTIME=false 或
    geocoding 失敗時的景點查詢使用（城市層級 fare_table.csv 不含 10 個景點）。
    檔案不存在則回傳 None。
    """
    global _OFFLINE_TABLE_CACHE
    if _OFFLINE_TABLE_CACHE is None:
        path = os.path.join(_STATIC_DIR, "offline_route_table.csv")
        if not os.path.exists(path):
            return None
        _OFFLINE_TABLE_CACHE = pd.read_csv(path)
    return _OFFLINE_TABLE_CACHE


def _offline_route_data(origin: str, destination: str) -> list[dict]:
    """從景點層級離線備援表查詢（雙向）。查無回 []。"""
    df = load_offline_route_table()
    if df is None or df.empty:
        return []

    mask = (df["origin"] == origin) & (df["destination"] == destination)
    route_df = df[mask]
    if route_df.empty:
        mask = (df["origin"] == destination) & (df["destination"] == origin)
        route_df = df[mask]
    if route_df.empty:
        return []

    results = []
    for _, row in route_df.iterrows():
        results.append({
            "mode": row["mode"],
            "cost": int(row["cost"]),
            "duration_min": int(row["duration_min"]),
            "distance_km": float(row["distance_km"]),
            "transfers": int(row["transfers"]),
            "frequency_per_day": int(row["frequency_per_day"]),
            "convenience": float(row["convenience"]),
            "source": "offline",
        })
    return results


def get_route_data(origin: str, destination: str) -> list[dict]:
    """
    根據起訖地查詢各交通方式的資料。

    先查景點層級離線備援表（offline_route_table.csv，涵蓋 UI 的 10 個景點）；
    查無再退回城市層級 fare_table.csv（台北/台中/台南/高雄）。

    回傳格式：
    [
        {
            "mode": "高鐵",
            "cost": 1490,
            "duration_min": 96,
            "distance_km": 345,
            "transfers": 0,
            "frequency_per_day": 120,
            "convenience": 8,
            "source": "static"
        },
        ...
    ]
    """
    offline = _offline_route_data(origin, destination)
    if offline:
        return offline

    df = load_fare_table()

    # 支援雙向查詢（若查不到 A→B，則嘗試 B→A）
    mask = (df["origin"] == origin) & (df["destination"] == destination)
    route_df = df[mask]

    if route_df.empty:
        mask = (df["origin"] == destination) & (df["destination"] == origin)
        route_df = df[mask]

    if route_df.empty:
        return []

    results = []
    for _, row in route_df.iterrows():
        results.append({
            "mode": row["mode"],
            "cost": int(row["cost"]),
            "duration_min": int(row["duration_min"]),
            "distance_km": float(row["distance_km"]),
            "transfers": int(row["transfers"]),
            "frequency_per_day": int(row["frequency_per_day"]),
            "convenience": float(row.get("convenience", 5)),
            "source": "static",
        })

    return results


# ──────────────────────────────────────────────
# Google Maps 靜態資料讀取
# ──────────────────────────────────────────────

def load_geocoding_attractions() -> pd.DataFrame:
    """
    讀取北北基桃景點座標資料。

    欄位：地點簡稱, 查詢名稱, Google回傳地址, 緯度_lat, 經度_lng,
          place_id, location_type, 查詢狀態
    """
    path = os.path.join(_STATIC_DIR, "geocoding_attractions.csv")
    return pd.read_csv(path)


def load_distance_matrix_driving() -> pd.DataFrame:
    """
    讀取景點間開車距離與時間矩陣。

    欄位：起點, 終點, 查詢狀態, 距離, 距離_公尺, 時間, 時間_秒
    共 100 筆（10 景點 × 10 景點）。
    """
    path = os.path.join(_STATIC_DIR, "distance_matrix_driving.csv")
    return pd.read_csv(path)


# ──────────────────────────────────────────────
# 臺北捷運票價表（big5 編碼）
# ──────────────────────────────────────────────

_MRT_FARE_CACHE: pd.DataFrame | None = None


def load_mrt_fare_matrix() -> pd.DataFrame:
    """
    讀取臺北捷運系統票價資料表（站對站 OD 矩陣）。

    欄位（big5 → 重新命名）：
      origin_station, dest_station, full_fare, concession_fare, distance_km
    """
    global _MRT_FARE_CACHE
    if _MRT_FARE_CACHE is not None:
        return _MRT_FARE_CACHE
    path = os.path.join(_STATIC_DIR, "臺北捷運系統票價資料(1141016).csv")
    df = pd.read_csv(path, encoding="big5")
    df = df.rename(
        columns={
            "起站": "origin_station",
            "訖站": "dest_station",
            "全票票價[金額]": "full_fare",
            "敬老卡愛心卡愛心陪伴卡及臺北市與新北市兒童優惠票價[金額]": "concession_fare",
            "距離": "distance_km",
        }
    )
    _MRT_FARE_CACHE = df
    return df


# 站名別名 → 票價表中的正式站名
_MRT_STATION_ALIASES = {
    "北車": "台北車站",
    "台北": "台北車站",
    "台北車站": "台北車站",
    "101": "台北101/世貿",
    "台北101": "台北101/世貿",
    "世貿": "台北101/世貿",
    "南港": "南港",
    "板橋": "板橋",
    "松山": "松山",
    "西門": "西門",
    "龍山寺": "龍山寺",
    "動物園": "動物園",
    "木柵動物園": "動物園",
    "新店": "新店",
    "碧潭": "新店",
    "淡水": "淡水",
    "北投": "北投",
    "象山": "象山",
}


def normalize_mrt_station(name: str) -> str | None:
    """把口語名/景點名/樞紐簡稱對齊到票價表中的站名。"""
    if not name:
        return None
    if name in _MRT_STATION_ALIASES:
        return _MRT_STATION_ALIASES[name]
    df = load_mrt_fare_matrix()
    stations = set(df["origin_station"].unique())
    if name in stations:
        return name
    return None


def lookup_mrt_fare(origin: str, destination: str) -> dict | None:
    """
    查兩個捷運站間的票價與距離；雙向查詢。

    Returns
    -------
    dict | None
        {
          "origin_station": "...",
          "dest_station": "...",
          "fare": 25,
          "distance_km": 5.4,
          "duration_min": 16,   # 由距離估算：2.0 min/km + 2 min 上下車緩衝
        }
    """
    o = normalize_mrt_station(origin)
    d = normalize_mrt_station(destination)
    if not (o and d):
        return None
    if o == d:
        return {
            "origin_station": o, "dest_station": d,
            "fare": 0, "distance_km": 0.0, "duration_min": 0,
        }
    df = load_mrt_fare_matrix()
    hit = df[(df["origin_station"] == o) & (df["dest_station"] == d)]
    if hit.empty:
        hit = df[(df["origin_station"] == d) & (df["dest_station"] == o)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    dist_km = float(r["distance_km"])
    # 估算搭乘時間：~30 km/h 平均（含停站），加 2 分鐘上下車緩衝
    duration_min = max(2, int(round(dist_km * 2.0)) + 2)
    return {
        "origin_station": o,
        "dest_station": d,
        "fare": int(r["full_fare"]),
        "distance_km": dist_km,
        "duration_min": duration_min,
    }


# ──────────────────────────────────────────────
# 北北基桃站間靜態票價表（高鐵 / 台鐵 / 國道客運）
# ──────────────────────────────────────────────

_HSR_FARE_CACHE: pd.DataFrame | None = None
_TRA_FARE_CACHE: pd.DataFrame | None = None
_BUS_FARE_CACHE: pd.DataFrame | None = None


def _load_station_fare_csv(filename: str) -> pd.DataFrame:
    """讀取站對站票價 CSV（欄位 origin_station,dest_station,fare,duration_min,distance_km）。"""
    path = os.path.join(_STATIC_DIR, filename)
    return pd.read_csv(path, encoding="utf-8", comment="#")


def load_hsr_fare_matrix() -> pd.DataFrame:
    """讀取高鐵北北基桃站間票價表。"""
    global _HSR_FARE_CACHE
    if _HSR_FARE_CACHE is None:
        _HSR_FARE_CACHE = _load_station_fare_csv("hsr_fare.csv")
    return _HSR_FARE_CACHE


def load_tra_fare_matrix() -> pd.DataFrame:
    """讀取台鐵北北基桃站間票價表。"""
    global _TRA_FARE_CACHE
    if _TRA_FARE_CACHE is None:
        _TRA_FARE_CACHE = _load_station_fare_csv("tra_fare.csv")
    return _TRA_FARE_CACHE


def load_bus_intercity_fare() -> pd.DataFrame:
    """讀取國道客運站間票價表。"""
    global _BUS_FARE_CACHE
    if _BUS_FARE_CACHE is None:
        _BUS_FARE_CACHE = _load_station_fare_csv("bus_intercity_fare.csv")
    return _BUS_FARE_CACHE


def _lookup_station_fare(
    df: pd.DataFrame, origin: str, destination: str,
    train_type: str | None = None,
) -> dict | None:
    """
    在站對站票價表中雙向查詢，回傳 {fare, duration_min, distance_km}（含 train_type 欄時一併帶出）。
    查無回 None；同站回 0。
    train_type 僅在表含 train_type 欄時生效；指定車種查無則退回該 OD 任一車種。
    """
    if not origin or not destination:
        return None
    if origin == destination:
        result = {"fare": 0, "duration_min": 0, "distance_km": 0.0}
        if "train_type" in df.columns and train_type:
            result["train_type"] = train_type
        return result

    hit = df[(df["origin_station"] == origin) & (df["dest_station"] == destination)]
    if hit.empty:
        hit = df[(df["origin_station"] == destination) & (df["dest_station"] == origin)]
    if hit.empty:
        return None

    if train_type and "train_type" in df.columns:
        typed = hit[hit["train_type"] == train_type]
        hit = typed if not typed.empty else hit

    r = hit.iloc[0]
    # 容忍缺值：fare/duration_min/distance_km 任一為 NaN（尚未填入真實票價）即視為查無，
    # 避免 int(NaN) 例外。已填的 OD 仍可正常查詢。
    if pd.isna(r["fare"]) or pd.isna(r["duration_min"]) or pd.isna(r["distance_km"]):
        return None
    result = {
        "fare": int(r["fare"]),
        "duration_min": int(r["duration_min"]),
        "distance_km": float(r["distance_km"]),
    }
    if "train_type" in df.columns:
        result["train_type"] = str(r["train_type"])
    if "transfers" in df.columns and not pd.isna(r["transfers"]):
        result["transfers"] = int(r["transfers"])
    if "route_nbr" in df.columns and not pd.isna(r["route_nbr"]):
        result["route_nbr"] = str(r["route_nbr"])
    return result


def lookup_hsr_fare(origin: str, destination: str) -> dict | None:
    """查高鐵兩站間票價（雙向）。"""
    return _lookup_station_fare(load_hsr_fare_matrix(), origin, destination)


def lookup_tra_fare(origin: str, destination: str, train_type: str = "區間") -> dict | None:
    """查台鐵兩站間票價（雙向）。預設區間車，可指定 '自強' 取備選快車票價。"""
    return _lookup_station_fare(load_tra_fare_matrix(), origin, destination, train_type)


def lookup_bus_fare(origin: str, destination: str) -> dict | None:
    """查國道客運兩站間票價（雙向）。"""
    return _lookup_station_fare(load_bus_intercity_fare(), origin, destination)


def load_directions_transit() -> pd.DataFrame:
    """
    讀取大眾運輸路線資料（101 → 小人國，轉乘次數較少偏好）。

    欄位：路線編號, 起點, 終點, 交通方式, 偏好, 總距離, 總距離_公尺,
          總時間, 總時間_秒, 步行時間_分鐘, 大眾運輸搭乘段數, 估計轉乘次數, 搭乘路線
    """
    path = os.path.join(_STATIC_DIR, "directions_transit.csv")
    return pd.read_csv(path)


def get_driving_distance(origin: str, destination: str) -> dict | None:
    """
    查詢兩景點間的開車距離與時間。

    Parameters
    ----------
    origin : str
        起點簡稱（如「北車」「101」）
    destination : str
        終點簡稱

    Returns
    -------
    dict | None
        {"origin", "destination", "distance_m", "duration_sec", "distance_text", "duration_text"}
        若查無資料則回傳 None
    """
    df = load_distance_matrix_driving()
    mask = (df["起點"] == origin) & (df["終點"] == destination)
    row = df[mask]

    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "origin": r["起點"],
        "destination": r["終點"],
        "distance_m": int(r["距離_公尺"]),
        "duration_sec": int(r["時間_秒"]),
        "distance_text": r["距離"],
        "duration_text": r["時間"],
    }


def get_transit_routes(origin: str, destination: str) -> list[dict]:
    """
    查詢兩地間的大眾運輸路線方案。

    Returns
    -------
    list[dict]
        每個 dict 包含：route_id, distance_m, duration_sec, walking_min,
        transit_segments, transfers, route_names
    """
    df = load_directions_transit()
    mask = (df["起點"] == origin) & (df["終點"] == destination)
    rows = df[mask]

    if rows.empty:
        return []

    results = []
    for _, r in rows.iterrows():
        results.append({
            "route_id": int(r["路線編號"]),
            "origin": r["起點"],
            "destination": r["終點"],
            "distance_m": int(r["總距離_公尺"]),
            "duration_sec": int(r["總時間_秒"]),
            "walking_min": float(r["步行時間_分鐘"]),
            "transit_segments": int(r["大眾運輸搭乘段數"]),
            "transfers": int(r["估計轉乘次數"]),
            "route_names": r["搭乘路線"],
        })

    return results