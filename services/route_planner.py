"""
services.route_planner — 多來源路線編排（D4：單模式聚合）

對外提供 plan_routes(origin, destination, ...) → list[RouteOption]，
每個 RouteOption 對齊 feature_engineer 期望的 schema：
    mode, cost, duration_min, distance_km, transfers,
    frequency_per_day, convenience, source, fetched_at, segments

支援的交通方式（北北基桃景點層級）：
    - 高鐵：兩端皆鄰近 THSR 站時才生成（TDX 班次 + 靜態票價）
    - 台鐵：兩端皆鄰近 TRA 站時才生成（TDX 班次 + 靜態票價）
    - 客運/捷運：Google Directions transit
    - 租車：Google Directions driving（含 duration_in_traffic 與高速里程）

D5 會再加：YouBike 末端接駁、多式聯運組合。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd

from config.settings import SETTINGS
from data.api_client import (
    get_route_data as static_get_route_data,
    load_geocoding_attractions,
    load_vehicle_params,
    lookup_bus_fare,
    lookup_hsr_fare,
    lookup_mrt_fare,
    lookup_tra_fare,
)
from data.clients.base import APIError
from data.clients.google_client import GoogleMapsClient
from data.clients.tdx_client import TDXClient
from data.clients.youbike_client import YouBikeClient, _haversine_m

logger = logging.getLogger(__name__)


# ── 景點 ↔ 樞紐就近對應表（人工編寫，覆蓋 geocoding_attractions.csv 的 10 個景點） ──
# 用途：當景點為起/迄時，找到「最近的 THSR/TRA 站」做為大眾運輸銜接點。
# None 代表此景點不適合該系統（距離太遠或步行接駁不合理）。
ATTRACTION_NEAR_HUBS: dict[str, dict[str, str | None]] = {
    "北車":       {"THSR": "台北", "TRA": "台北"},
    "101":        {"THSR": "台北", "TRA": "台北"},      # 走捷運轉乘
    "陽明山竹子湖": {"THSR": None,   "TRA": None},
    "碧潭":       {"THSR": None,   "TRA": None},        # 走捷運新店線
    "小人國":     {"THSR": "桃園", "TRA": "中壢"},
    "廟口夜市":   {"THSR": None,   "TRA": "基隆"},
    "三峽老街":   {"THSR": None,   "TRA": None},
    "桃機":       {"THSR": "桃園", "TRA": "桃園"},      # 桃捷接駁
    "龍山寺":     {"THSR": "台北", "TRA": "萬華"},
    "木柵動物園":  {"THSR": None,   "TRA": None},
}

# 從樞紐到景點的步行/接駁估算（分鐘）
HUB_TO_ATTRACTION_TRANSFER_MIN = 25


# ── Geocoding ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _attraction_df() -> pd.DataFrame:
    return load_geocoding_attractions()


def resolve_location(name: str) -> dict[str, Any] | None:
    """
    景點/地名 → {lat, lng, address, source}。
    先查 geocoding_attractions.csv，未命中再打 Google Geocoding。
    """
    df = _attraction_df()
    hit = df[df["地點簡稱"] == name]
    if hit.empty:
        hit = df[df["查詢名稱"] == name]
    if not hit.empty:
        r = hit.iloc[0]
        return {
            "lat": float(r["緯度_lat"]),
            "lng": float(r["經度_lng"]),
            "address": r["Google回傳地址"],
            "source": "static",
        }

    if SETTINGS.has_google and SETTINGS.enable_realtime:
        try:
            g = GoogleMapsClient().geocode(name)
            if g:
                return {**g, "address": g.get("formatted_address", ""), "source": "google"}
        except APIError as e:
            logger.warning("geocode fallback failed for %s: %s", name, e)
    return None


# ── 計價常數 ─────────────────────────────────────────────────

# 市區公車：每段（每次上車）固定票價
BUS_FLAT_FARE = 15


# ── 各模式 builder ───────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _build_hsr(
    origin_name: str, dest_name: str,
    o_loc: dict, d_loc: dict,
) -> dict | None:
    """高鐵：兩端鄰近 THSR 站 + TDX 班次 + 靜態票價。"""
    o_hub = ATTRACTION_NEAR_HUBS.get(origin_name, {}).get("THSR")
    d_hub = ATTRACTION_NEAR_HUBS.get(dest_name, {}).get("THSR")
    if not (o_hub and d_hub) or o_hub == d_hub:
        return None

    tdx = TDXClient()
    o_id = tdx.lookup_station("THSR", o_hub)
    d_id = tdx.lookup_station("THSR", d_hub)
    if not (o_id and d_id):
        return None

    duration_min = None
    frequency_per_day = 0
    try:
        if SETTINGS.has_tdx and SETTINGS.enable_realtime:
            sched = tdx.thsr_od_timetable(o_id, d_id)
            frequency_per_day = len(sched)
            if sched:
                # 取最短班次當代表
                duration_min = min(s["duration_min"] for s in sched if s["duration_min"] > 0)
    except APIError as e:
        logger.warning("THSR live failed: %s", e)

    # 票價走北北基桃站間靜態票價表；查無時用距離 × 4 元估算
    fare_info = lookup_hsr_fare(o_hub, d_hub)
    distance_km = round(fare_info["distance_km"] if fare_info else _haversine_km(o_loc, d_loc), 2)
    cost = fare_info["fare"] if fare_info else max(165, int(distance_km * 4))
    if duration_min is None:
        duration_min = fare_info["duration_min"] if fare_info else 0
    if duration_min == 0:
        return None

    # 加上景點 ↔ 樞紐接駁時間（雙端）
    total_dur = duration_min + 2 * HUB_TO_ATTRACTION_TRANSFER_MIN
    main_fare = cost if cost is not None else 0

    return {
        "mode": "高鐵",
        "cost": main_fare,
        "duration_min": int(total_dur),
        "distance_km": float(distance_km),
        "transfers": 1,    # 至少一次（景點→車站）
        "frequency_per_day": frequency_per_day or 60,
        "convenience": 7,
        "source": "tdx+static" if frequency_per_day else "static",
        "fetched_at": _now_iso(),
        "segments": [
            {"mode": "步行/接駁", "from": origin_name, "to": f"{o_hub}高鐵站",
             "duration_min": HUB_TO_ATTRACTION_TRANSFER_MIN, "cost": 0, "distance_km": 0.0},
            {"mode": "高鐵", "from": f"{o_hub}高鐵站", "to": f"{d_hub}高鐵站",
             "duration_min": int(duration_min), "cost": main_fare,
             "distance_km": float(distance_km)},
            {"mode": "步行/接駁", "from": f"{d_hub}高鐵站", "to": dest_name,
             "duration_min": HUB_TO_ATTRACTION_TRANSFER_MIN, "cost": 0, "distance_km": 0.0},
        ],
    }


def _build_tra(
    origin_name: str, dest_name: str,
    o_loc: dict, d_loc: dict,
) -> dict | None:
    """台鐵：類似 HSR。"""
    o_hub = ATTRACTION_NEAR_HUBS.get(origin_name, {}).get("TRA")
    d_hub = ATTRACTION_NEAR_HUBS.get(dest_name, {}).get("TRA")
    if not (o_hub and d_hub) or o_hub == d_hub:
        return None

    tdx = TDXClient()
    o_id = tdx.lookup_station("TRA", o_hub)
    d_id = tdx.lookup_station("TRA", d_hub)
    if not (o_id and d_id):
        return None

    duration_min = None
    frequency_per_day = 0
    try:
        if SETTINGS.has_tdx and SETTINGS.enable_realtime:
            sched = tdx.tra_od_timetable(o_id, d_id)
            # 優先取自強/區間快這類較快車種
            fast = [s for s in sched if s.get("train_type") in {"自強", "自強(3000)", "區間快", "普悠瑪", "太魯閣"}]
            pool = fast or sched
            frequency_per_day = len(sched)
            if pool:
                duration_min = min(s["duration_min"] for s in pool if s["duration_min"] > 0)
    except APIError as e:
        logger.warning("TRA live failed: %s", e)

    fare_info = lookup_tra_fare(o_hub, d_hub)
    distance_km = round(fare_info["distance_km"] if fare_info else _haversine_km(o_loc, d_loc), 2)
    cost = fare_info["fare"] if fare_info else max(41, int(distance_km * 1.46))  # 自強約 1.46/km
    if duration_min is None:
        duration_min = fare_info["duration_min"] if fare_info else 0
    if duration_min == 0:
        return None

    total_dur = duration_min + 2 * HUB_TO_ATTRACTION_TRANSFER_MIN
    main_fare = int(cost)
    return {
        "mode": "台鐵",
        "cost": main_fare,
        "duration_min": int(total_dur),
        "distance_km": float(distance_km),
        "transfers": 1,
        "frequency_per_day": frequency_per_day or 40,
        "convenience": 6,
        "source": "tdx+static" if frequency_per_day else "static",
        "fetched_at": _now_iso(),
        "segments": [
            {"mode": "步行/接駁", "from": origin_name, "to": f"{o_hub}車站",
             "duration_min": HUB_TO_ATTRACTION_TRANSFER_MIN, "cost": 0, "distance_km": 0.0},
            {"mode": "台鐵", "from": f"{o_hub}車站", "to": f"{d_hub}車站",
             "duration_min": int(duration_min), "cost": main_fare,
             "distance_km": float(distance_km)},
            {"mode": "步行/接駁", "from": f"{d_hub}車站", "to": dest_name,
             "duration_min": HUB_TO_ATTRACTION_TRANSFER_MIN, "cost": 0, "distance_km": 0.0},
        ],
    }


def _build_transit(
    origin_name: str, dest_name: str,
    o_loc: dict, d_loc: dict,
) -> dict | None:
    """客運（含捷運/公車）：Google Directions transit。"""
    if not SETTINGS.has_google or not SETTINGS.enable_realtime:
        return None
    try:
        gc = GoogleMapsClient()
        routes = gc.directions(
            f"{o_loc['lat']},{o_loc['lng']}",
            f"{d_loc['lat']},{d_loc['lng']}",
            mode="transit",
        )
    except APIError as e:
        logger.warning("transit failed: %s", e)
        return None
    if not routes:
        return None

    # 取總時間最短的 alternative
    r = min(routes, key=lambda x: x["duration_sec"])
    duration_min = max(1, round(r["duration_sec"] / 60))
    distance_km = round(r["distance_m"] / 1000.0, 2)

    raw_segs = r["transit_segments"] or []
    # 逐段計價：公車（每次上車）固定 15 元；捷運段查臺北捷運票價，查無則退回 15 元
    segments = []
    for s in raw_segs:
        vtype = (s.get("vehicle_type") or "").upper()
        is_bus = "BUS" in vtype
        if is_bus:
            seg_cost = BUS_FLAT_FARE
            seg_mode = "公車"
        else:
            mrt = lookup_mrt_fare(s["from"], s["to"])
            seg_cost = mrt["fare"] if mrt else BUS_FLAT_FARE
            seg_mode = "捷運"
        segments.append({
            "mode": seg_mode,
            "from": s["from"], "to": s["to"],
            "line": s["line"],
            "duration_min": round(s["duration_sec"] / 60),
            "cost": int(seg_cost),
            "distance_km": round(s.get("distance_m", 0) / 1000.0, 2),
        })
    if not segments:
        segments = [{
            "mode": "公車", "from": origin_name, "to": dest_name,
            "duration_min": duration_min, "cost": BUS_FLAT_FARE,
            "distance_km": float(distance_km),
        }]

    cost = sum(int(s["cost"]) for s in segments)

    return {
        "mode": "大眾運輸",
        "cost": cost,
        "duration_min": duration_min,
        "distance_km": distance_km,
        "transfers": r["transfers"],
        "frequency_per_day": 80,
        "convenience": 6.0 - min(r["transfers"], 3),  # 每多一次轉乘 -1
        "source": "google",
        "fetched_at": _now_iso(),
        "segments": segments,
    }


def _build_intercity_bus(
    origin_name: str, dest_name: str,
    o_loc: dict, d_loc: dict,
) -> dict | None:
    """
    國道/公路客運：景點層級查詢 bus_intercity_fare.csv（資料萃取自 taiwan_route_dataset.csv
    含四位數路線編號的客運路線）。票價/時間/距離/轉乘/路線皆來自該表；fare 未填則查無、不生成。
    """
    fare_info = lookup_bus_fare(origin_name, dest_name)
    if not fare_info:
        return None

    distance_km = round(fare_info["distance_km"], 2)
    cost = int(fare_info["fare"])
    ride_min = int(fare_info["duration_min"])
    transfers = int(fare_info.get("transfers", 1))
    route_nbr = fare_info.get("route_nbr", "")
    bus_label = f"客運{route_nbr}" if route_nbr else "客運"

    return {
        "mode": "客運",
        "cost": cost,
        "duration_min": ride_min,
        "distance_km": float(distance_km),
        "transfers": transfers,
        "frequency_per_day": 40,
        "convenience": round(6.0 - min(transfers, 3), 1),
        "source": "static",
        "fetched_at": _now_iso(),
        "segments": [
            {"mode": bus_label, "from": origin_name, "to": dest_name,
             "duration_min": ride_min, "cost": cost, "distance_km": float(distance_km)},
        ],
    }


def _build_driving(
    origin_name: str, dest_name: str,
    o_loc: dict, d_loc: dict,
    num_people: int, luggage: int = 0,
) -> dict | None:
    """租車：Google driving + 自訂車種規則（D4 暫用 vehicle_params 預設）。"""
    if not SETTINGS.has_google or not SETTINGS.enable_realtime:
        return None
    try:
        gc = GoogleMapsClient()
        routes = gc.directions(
            f"{o_loc['lat']},{o_loc['lng']}",
            f"{d_loc['lat']},{d_loc['lng']}",
            mode="driving",
        )
    except APIError as e:
        logger.warning("driving failed: %s", e)
        return None
    if not routes:
        return None

    # 取含路況最短時間
    r = min(routes, key=lambda x: x.get("duration_in_traffic_sec") or x["duration_sec"])
    duration_sec = r.get("duration_in_traffic_sec") or r["duration_sec"]
    duration_min = max(1, round(duration_sec / 60))
    distance_km = round(r["distance_m"] / 1000.0, 2)
    highway_km = r["highway_distance_m"] / 1000.0

    _, fits_luggage = _select_car_type(num_people, luggage)
    cost, car_type = _estimate_car_cost(
        distance_km, highway_km, num_people, duration_min, luggage
    )
    # 車型裝得下人數+行李 → 便利性高；裝不下（需多趟/擠）→ 扣分但仍顯示
    convenience = 8.0 if fits_luggage else 5.5

    return {
        "mode": "租車",
        "cost": cost,
        "duration_min": duration_min,
        "distance_km": distance_km,
        "transfers": 0,
        "frequency_per_day": 999,
        "convenience": convenience,
        "luggage_fits": fits_luggage,
        "source": "google",
        "fetched_at": _now_iso(),
        "car_type": car_type,
        "segments": [{
            "mode": "租車", "from": origin_name, "to": dest_name,
            "duration_min": duration_min,
            "distance_km": float(distance_km),
            "cost": int(cost),
            "highway_km": round(highway_km, 2),
            "car_type": car_type,
            "luggage_fits": fits_luggage,
            "summary": r.get("summary", ""),
        }],
    }


# ── 租車費用估算（iRent 時租制） ──────────────────────────────

# 車型/費率不在靜態檔時的後備值（iRent 平日定價概估）
_DEFAULT_CAR_TYPES = [
    {"name": "小型車", "hourly_price": 250, "mileage_fee_per_km": 3.4, "max_passengers": 5, "max_luggage": 2},
    {"name": "七人座", "hourly_price": 320, "mileage_fee_per_km": 3.5, "max_passengers": 7, "max_luggage": 4},
]


def _car_params() -> dict:
    try:
        return load_vehicle_params().get("租車", {}) or {}
    except Exception:
        return {}


def _select_car_type(num_people: int, luggage: int) -> tuple[dict, bool]:
    """
    依人數與行李挑車型，回傳 (車型 dict, 行李是否裝得下)。
    - 先挑「可載人數 ≥ 人數」的車型；
    - 其中優先挑「行李容量 ≥ 行李數」者，取最便宜；
    - 若無車型同時裝得下，退而求其次取行李容量最大者並標記裝不下。
    """
    car = _car_params()
    car_types = car.get("car_types") or _DEFAULT_CAR_TYPES

    fit_people = [c for c in car_types if c.get("max_passengers", 0) >= num_people] or car_types
    fit_both = [c for c in fit_people if c.get("max_luggage", 0) >= luggage]
    if fit_both:
        chosen = min(fit_both, key=lambda c: c.get("hourly_price", float("inf")))
        return chosen, True
    # 行李裝不下：在可載人數的車型中取行李容量最大者（同容量取較便宜）
    chosen = max(
        fit_people,
        key=lambda c: (c.get("max_luggage", 0), -c.get("hourly_price", 0)),
    )
    return chosen, False


def _estimate_car_cost(
    distance_km: float, highway_km: float, num_people: int, duration_min: int,
    luggage: int = 0,
) -> tuple[int, str]:
    """
    iRent 時租制計價：
        費用 = 時租價 × 租用時數 + 里程費 × 距離 + 過路費 × 高速里程
    - 車型：依 _select_car_type（可載人數 + 行李容量）挑選；里程費依車型而定。
    - 租用時數：依單趟行車時間無條件進位（ceil(分鐘/60)，至少 1 小時）
    回傳 (費用, 車型名稱)。
    """
    car = _car_params()
    toll_rate = car.get("toll_per_km", 1.2)
    default_mileage = car.get("mileage_fee_per_km", 3.5)

    chosen, _ = _select_car_type(num_people, luggage)
    hourly = chosen.get("hourly_price", 300)
    mileage_rate = chosen.get("mileage_fee_per_km", default_mileage)
    car_name = chosen.get("name", "租車")

    rental_hours = max(1, math.ceil(duration_min / 60))
    cost = hourly * rental_hours + distance_km * mileage_rate + highway_km * toll_rate
    return int(round(cost)), car_name


# ── 工具 ─────────────────────────────────────────────────────


def _haversine_km(a: dict, b: dict) -> float:
    return _haversine_m(a["lat"], a["lng"], b["lat"], b["lng"]) / 1000.0


# ── 對外主 entry ─────────────────────────────────────────────


def plan_routes(
    origin: str,
    destination: str,
    *,
    num_people: int = 1,
    luggage: int = 0,
    departure_time: datetime | None = None,
    enable_realtime: bool | None = None,
    with_multimodal: bool = True,
) -> list[dict]:
    """
    回傳已 normalize 的 RouteOption 列表（尚未評分）。
    若 enable_realtime=None 則跟隨 SETTINGS.enable_realtime。

    若全部 builder 都拿不到資料，回退到 static_get_route_data() 的城市層級結果。
    """
    realtime_enabled = SETTINGS.enable_realtime if enable_realtime is None else enable_realtime
    if not realtime_enabled:
        return static_get_route_data(origin, destination)

    o_loc = resolve_location(origin)
    d_loc = resolve_location(destination)
    if not (o_loc and d_loc):
        # geocode 失敗 → 嘗試靜態 fare table
        return static_get_route_data(origin, destination)

    options: list[dict] = []
    for builder in (
        lambda: _build_hsr(origin, destination, o_loc, d_loc),
        lambda: _build_tra(origin, destination, o_loc, d_loc),
        lambda: _build_transit(origin, destination, o_loc, d_loc),
        lambda: _build_intercity_bus(origin, destination, o_loc, d_loc),
        lambda: _build_driving(origin, destination, o_loc, d_loc, num_people, luggage),
    ):
        try:
            opt = builder()
            if opt:
                options.append(opt)
        except Exception as e:
            logger.exception("builder failed: %s", e)

    if not options:
        # 完全沒拿到 → 靜態 fallback
        return static_get_route_data(origin, destination)

    if with_multimodal:
        # 延遲 import 避免循環依賴（multimodal 也會 import route_planner）
        from services.multimodal import augment_with_youbike
        from services.transfer_planner import augment_with_mrt
        options = augment_with_mrt(origin, destination, options)
        options = augment_with_youbike(origin, destination, options)

    options = [_finalize_segments(opt) for opt in options]
    return options


def _finalize_segments(opt: dict) -> dict:
    """
    收尾：確保每個 segment 都有 cost/distance_km/duration_min 欄位；
    用 segments 加總重算頂層 cost/duration_min/transfers，
    保證「分段加總 == 整段顯示值」這個不變式。
    """
    segs = opt.get("segments") or []
    if not segs:
        return opt
    for s in segs:
        s.setdefault("cost", 0)
        s.setdefault("distance_km", 0.0)
        s.setdefault("duration_min", 0)
    total_cost = sum(int(s["cost"]) for s in segs)
    total_dur = sum(int(s["duration_min"]) for s in segs)
    # 若原本頂層 cost/duration 與 segments 加總對不上，以 segments 為準（單一資料源）
    opt["cost"] = total_cost
    opt["duration_min"] = total_dur
    opt["transfers"] = max(0, len(segs) - 1)
    return opt


def youbike_context(
    origin: str, destination: str, *, k: int = 2,
) -> dict[str, list[dict]]:
    """
    回傳起訖兩端附近的 YouBike 站點資訊（給 UI 顯示 / D5 末端接駁使用）。
    """
    o_loc = resolve_location(origin)
    d_loc = resolve_location(destination)
    if not (o_loc and d_loc):
        return {"origin": [], "destination": []}
    yb = YouBikeClient()
    return {
        "origin": yb.nearest_stations(o_loc["lat"], o_loc["lng"], k=k, require_bikes=True),
        "destination": yb.nearest_stations(d_loc["lat"], d_loc["lng"], k=k, require_bikes=False),
    }
