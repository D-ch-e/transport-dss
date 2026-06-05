"""
data.clients.google_client — Google Maps Platform 即時 client

支援：
- Geocoding（地點 → 經緯度）
- Distance Matrix（多對多開車距離 / 時間，含路況）
- Directions（單對單路線；driving 含 duration_in_traffic、transit 含轉乘資訊）

快取策略：
- Geocoding：永久（TTL = static_metadata）— 景點座標基本不變
- Distance Matrix / Directions driving：5 分鐘（含路況需即時）
- Directions transit：5 分鐘（班次會變）

所有對外方法回傳已 normalize 的 dict，欄位名與 data/api_client.py 的靜態版本對齊，
方便 planner 直接拼接。
"""

from __future__ import annotations

import logging
from typing import Literal

from config.settings import SETTINGS
from data.clients.base import APIError, BaseAPIClient

logger = logging.getLogger(__name__)

GOOGLE_BASE = "https://maps.googleapis.com/maps/api"

TravelMode = Literal["driving", "transit", "walking", "bicycling"]


class GoogleMapsClient(BaseAPIClient):
    source_name = "google"
    default_timeout = 4.0
    max_retries = 2

    def __init__(self) -> None:
        super().__init__()
        self.api_key = SETTINGS.google_maps_api_key
        if not self.api_key:
            logger.warning("GOOGLE_MAPS_API_KEY missing — GoogleMapsClient will refuse to call live API")

    def _require_key(self) -> None:
        if not self.api_key:
            raise APIError("GOOGLE_MAPS_API_KEY not configured")

    def _get_json(self, path: str, params: dict) -> dict:
        self._require_key()
        params = {**params, "key": self.api_key, "language": "zh-TW", "region": "tw"}
        url = f"{GOOGLE_BASE}{path}"
        resp = self._request_with_retry("GET", url, params=params)
        try:
            body = resp.json()
        except ValueError as e:
            raise APIError(f"Google API returned non-JSON: {e}") from e
        status = body.get("status")
        # Google 的 status：OK / ZERO_RESULTS / OVER_QUERY_LIMIT / REQUEST_DENIED / INVALID_REQUEST / UNKNOWN_ERROR
        if status not in {"OK", "ZERO_RESULTS"}:
            raise APIError(f"Google API status={status}: {body.get('error_message', '')}")
        return body

    # ── Geocoding ─────────────────────────────────────────────

    def geocode(self, address: str) -> dict | None:
        """
        地址 / 景點名稱 → {lat, lng, formatted_address, place_id}
        回傳 None 表示 ZERO_RESULTS。
        """
        endpoint = "/geocode/json"
        params = {"address": address}

        def fetcher() -> dict | None:
            body = self._get_json(endpoint, params)
            results = body.get("results") or []
            if not results:
                return None
            r = results[0]
            loc = r["geometry"]["location"]
            return {
                "lat": loc["lat"],
                "lng": loc["lng"],
                "formatted_address": r.get("formatted_address", ""),
                "place_id": r.get("place_id", ""),
            }

        data, _ = self.cached_call(
            endpoint, fetcher, params=params, ttl=SETTINGS.ttl.static_metadata,
        )
        return data

    # ── Distance Matrix ───────────────────────────────────────

    def distance_matrix(
        self,
        origins: list[str],
        destinations: list[str],
        mode: TravelMode = "driving",
        departure_time: str | int = "now",
    ) -> list[list[dict | None]]:
        """
        多對多距離矩陣。回傳 [len(origins)][len(destinations)] 的 dict 陣列：
            {"distance_m", "duration_sec", "duration_in_traffic_sec"?}
        無結果的格 = None。
        """
        endpoint = "/distancematrix/json"
        params = {
            "origins": "|".join(origins),
            "destinations": "|".join(destinations),
            "mode": mode,
        }
        if mode == "driving":
            params["departure_time"] = str(departure_time)
            params["traffic_model"] = "best_guess"

        def fetcher() -> list[list[dict | None]]:
            body = self._get_json(endpoint, params)
            rows = body.get("rows", [])
            out: list[list[dict | None]] = []
            for row in rows:
                row_out: list[dict | None] = []
                for el in row.get("elements", []):
                    if el.get("status") != "OK":
                        row_out.append(None)
                        continue
                    cell = {
                        "distance_m": el["distance"]["value"],
                        "duration_sec": el["duration"]["value"],
                    }
                    if "duration_in_traffic" in el:
                        cell["duration_in_traffic_sec"] = el["duration_in_traffic"]["value"]
                    row_out.append(cell)
                out.append(row_out)
            return out

        ttl = SETTINGS.ttl.directions if mode == "driving" else SETTINGS.ttl.static_metadata
        data, _ = self.cached_call(endpoint, fetcher, params=params, ttl=ttl)
        return data

    # ── Directions ────────────────────────────────────────────

    def directions(
        self,
        origin: str,
        destination: str,
        mode: TravelMode = "driving",
        departure_time: str | int = "now",
        alternatives: bool = True,
    ) -> list[dict]:
        """
        單對單路線。回傳已 normalize 的路線列表（可能多條 alternatives）：
            {
                "mode": "driving" | "transit" | ...,
                "distance_m": int,
                "duration_sec": int,
                "duration_in_traffic_sec": int | None,
                "highway_distance_m": int,    # 估算高速里程（mode=driving 才有意義）
                "transit_segments": [         # mode=transit 才有
                    {"line": "高鐵", "from": "台北", "to": "桃園",
                     "duration_sec": 1320, "num_stops": 1}
                ],
                "transfers": int,
                "summary": str,
            }
        """
        endpoint = "/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "alternatives": "true" if alternatives else "false",
        }
        if mode == "driving":
            params["departure_time"] = str(departure_time)
            params["traffic_model"] = "best_guess"
        elif mode == "transit":
            params["departure_time"] = str(departure_time)

        def fetcher() -> list[dict]:
            body = self._get_json(endpoint, params)
            routes = body.get("routes", [])
            return [_normalize_route(r, mode) for r in routes]

        data, _ = self.cached_call(endpoint, fetcher, params=params, ttl=SETTINGS.ttl.directions)
        return data


# ── normalize helpers ────────────────────────────────────────

def _normalize_route(route: dict, mode: str) -> dict:
    """把 Google Directions 的單一 route 攤平成模型可用的 dict。"""
    legs = route.get("legs", [])
    total_dist = sum(l["distance"]["value"] for l in legs)
    total_dur = sum(l["duration"]["value"] for l in legs)
    dur_traffic = 0
    has_traffic = False
    for l in legs:
        if "duration_in_traffic" in l:
            dur_traffic += l["duration_in_traffic"]["value"]
            has_traffic = True

    transit_segments: list[dict] = []
    highway_dist = 0
    transfers = 0

    for l in legs:
        for step in l.get("steps", []):
            travel_mode = step.get("travel_mode")
            if travel_mode == "TRANSIT":
                td = step.get("transit_details", {})
                line = td.get("line", {})
                transit_segments.append({
                    "line": line.get("short_name") or line.get("name", ""),
                    "vehicle_type": line.get("vehicle", {}).get("type", ""),
                    "from": td.get("departure_stop", {}).get("name", ""),
                    "to": td.get("arrival_stop", {}).get("name", ""),
                    "duration_sec": step.get("duration", {}).get("value", 0),
                    "num_stops": td.get("num_stops", 0),
                })
            elif travel_mode == "DRIVING":
                # html_instructions 內含 "高速公路" 或道路代號（國道/Hwy）→ 視為高速
                html = (step.get("html_instructions") or "").lower()
                if any(k in html for k in ("國道", "高速公路", "freeway", "hwy", "highway")):
                    highway_dist += step.get("distance", {}).get("value", 0)

    if transit_segments:
        transfers = max(0, len(transit_segments) - 1)

    return {
        "mode": mode,
        "distance_m": total_dist,
        "duration_sec": total_dur,
        "duration_in_traffic_sec": dur_traffic if has_traffic else None,
        "highway_distance_m": highway_dist,
        "transit_segments": transit_segments,
        "transfers": transfers,
        "summary": route.get("summary", ""),
    }
