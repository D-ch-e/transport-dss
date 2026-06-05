"""
services.transfer_planner — 多式聯運：以臺北捷運替換 HSR/TRA 主幹線兩端的步行接駁。

策略：
- 對於既有的 高鐵 / 台鐵 RouteOption（segments 為 [步行/接駁, 主幹線, 步行/接駁]）
- 若景點與樞紐站在臺北捷運網內可查到票價（lookup_mrt_fare），
  則把該端的 25 分鐘步行/接駁 segment 換成「捷運」segment（含實際票價與時間）。
- 雙端皆可替換時兩段都換；若皆無 MRT 對應則不產生新方案。
- 同時新增一個「純捷運」方案（市內景點互通），與 Google transit 結果並存。

對外：augment_with_mrt(origin, destination, base_options) → list[dict]
回傳：原方案 + 多式聯運（含捷運接駁）方案。
"""

from __future__ import annotations

import logging
from copy import deepcopy

from data.api_client import lookup_mrt_fare, normalize_mrt_station

logger = logging.getLogger(__name__)


# 景點 → 最近捷運站（None 表示該景點不在 MRT 服務範圍內）
ATTRACTION_NEAR_MRT: dict[str, str | None] = {
    "北車": "台北車站",
    "101": "台北101/世貿",
    "龍山寺": "龍山寺",
    "碧潭": "新店",
    "木柵動物園": "動物園",
    "陽明山竹子湖": None,
    "三峽老街": None,
    "小人國": None,
    "廟口夜市": None,
    "桃機": None,
}

# HSR/TRA 樞紐站 → 對應的捷運站名（None 表示該樞紐不通捷運）
HUB_TO_MRT: dict[str, str | None] = {
    "台北": "台北車站",
    "板橋": "板橋",
    "南港": "南港",
    "松山": "松山",
    # 以下樞紐目前票價表內無對應 → None
    "桃園": None, "中壢": None, "新竹": None,
    "嘉義": None, "台南": None, "左營": None,
    "基隆": None, "萬華": None,
}


def _mrt_segment(from_name: str, to_name: str) -> dict | None:
    """以景點/樞紐名稱查 MRT，產生一個 segment dict。"""
    fare = lookup_mrt_fare(from_name, to_name)
    if not fare:
        return None
    return {
        "mode": "捷運",
        "from": normalize_mrt_station(from_name) or from_name,
        "to": normalize_mrt_station(to_name) or to_name,
        "duration_min": fare["duration_min"],
        "cost": fare["fare"],
        "distance_km": fare["distance_km"],
        "line": "臺北捷運",
    }


def _augment_one(opt: dict, origin: str, destination: str) -> dict | None:
    """
    對單一 HSR/TRA option 嘗試把兩端 25 分鐘的步行接駁換成 MRT。
    需至少替換一端才會回傳新方案。
    """
    mode = opt.get("mode")
    if mode not in {"高鐵", "台鐵"}:
        return None
    segs = opt.get("segments") or []
    if len(segs) < 3:
        return None

    # 從 segments 抓樞紐站名（位於主幹線 segment 的 from/to）
    main_seg = segs[1]
    o_hub_full = main_seg.get("from", "")  # 例 "台北高鐵站" / "桃園車站"
    d_hub_full = main_seg.get("to", "")

    # 樞紐站名去掉 "高鐵站"/"車站" 後綴
    def _strip(name: str) -> str:
        for suffix in ("高鐵站", "車站"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    o_hub = _strip(o_hub_full)
    d_hub = _strip(d_hub_full)

    o_mrt_hub = HUB_TO_MRT.get(o_hub)
    d_mrt_hub = HUB_TO_MRT.get(d_hub)
    o_mrt_attr = ATTRACTION_NEAR_MRT.get(origin)
    d_mrt_attr = ATTRACTION_NEAR_MRT.get(destination)

    new_segs = list(segs)
    replaced = 0

    # 起點端：景點 MRT → 樞紐 MRT
    if o_mrt_attr and o_mrt_hub and o_mrt_attr != o_mrt_hub:
        mrt_seg = _mrt_segment(origin, o_hub)
        if mrt_seg:
            new_segs[0] = mrt_seg
            replaced += 1

    # 終點端：樞紐 MRT → 景點 MRT
    if d_mrt_attr and d_mrt_hub and d_mrt_attr != d_mrt_hub:
        mrt_seg = _mrt_segment(d_hub, destination)
        if mrt_seg:
            new_segs[-1] = mrt_seg
            replaced += 1

    if replaced == 0:
        return None

    new_opt = deepcopy(opt)
    new_opt["segments"] = new_segs
    new_opt["mode"] = f"{mode}+捷運"
    new_opt["source"] = opt.get("source", "static") + "+mrt"
    # 便利性：MRT 比走路省力，但多一次轉乘 → 微幅淨增
    new_opt["convenience"] = round(opt.get("convenience", 6) + 0.3, 2)
    return new_opt


def _pure_mrt_option(origin: str, destination: str) -> dict | None:
    """若兩端都在臺北捷運網內，產生一個純捷運的 RouteOption。"""
    o_mrt = ATTRACTION_NEAR_MRT.get(origin)
    d_mrt = ATTRACTION_NEAR_MRT.get(destination)
    if not (o_mrt and d_mrt) or o_mrt == d_mrt:
        return None
    seg = _mrt_segment(origin, destination)
    if not seg:
        return None
    return {
        "mode": "捷運",
        "cost": seg["cost"],
        "duration_min": seg["duration_min"],
        "distance_km": seg["distance_km"],
        "transfers": 0,
        "frequency_per_day": 200,
        "convenience": 7.5,
        "source": "static-mrt",
        "fetched_at": opt_now_iso(),
        "segments": [seg],
    }


def opt_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def augment_with_mrt(origin: str, destination: str, base_options: list[dict]) -> list[dict]:
    """
    對 base_options 中的 高鐵/台鐵 方案，嘗試把兩端步行/接駁換成捷運；
    另外若兩端皆在捷運網內，加一個純捷運方案。

    回傳：原方案 + MRT 變體 + 純捷運方案（如可）。
    """
    added: list[dict] = []
    for opt in base_options:
        new_opt = _augment_one(opt, origin, destination)
        if new_opt:
            added.append(new_opt)

    # 純捷運（避免重複加：如果 base 已含 mode=="捷運" 就跳過）
    if not any(o.get("mode") == "捷運" for o in base_options):
        pm = _pure_mrt_option(origin, destination)
        if pm:
            added.append(pm)

    return base_options + added
