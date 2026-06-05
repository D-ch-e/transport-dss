"""
data.clients.youbike_client — YouBike 即時站點狀態

支援來源（皆免金鑰）：
- 台北市：https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json
- 新北市：https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json

兩來源欄位不同，normalize 後統一 schema：
    {"id", "name", "area", "city", "lat", "lng",
     "available_rent_bikes", "available_return_bikes", "total", "active",
     "update_time"}
"""

from __future__ import annotations

import logging
import math

from config.settings import SETTINGS
from data.clients.base import APIError, BaseAPIClient

logger = logging.getLogger(__name__)

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NEWTAIPEI_URL = (
    "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json"
    "?size=5000"
)
# 桃園市政府開放資料平台的 YouBike API 只回「區域層級彙總」（20 筆，無 lat/lng），
# 無法做最近站查詢。改用官方 YouBike 全站清單，依 area_code 過濾出桃園 670 站。
YOUBIKE_OFFICIAL_URL = "https://apis.youbike.com.tw/json/station-yb2.json"
TAOYUAN_AREA_CODE = "07"


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """兩經緯度距離（公尺）。"""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class YouBikeClient(BaseAPIClient):
    source_name = "youbike"

    # 官方 apis.youbike.com.tw 會用 UA 擋預設 python-requests，補一個瀏覽器 UA
    _BROWSER_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    def _fetch(self, url: str) -> list[dict]:
        resp = self._request_with_retry(
            "GET", url, headers={"User-Agent": self._BROWSER_UA, "Accept": "application/json"},
        )
        if resp.status_code >= 400:
            raise APIError(f"YouBike {url} returned {resp.status_code}")
        try:
            return resp.json()
        except ValueError as e:
            raise APIError(f"YouBike non-JSON: {e}") from e

    def _stations_taipei(self) -> list[dict]:
        def fetcher() -> list[dict]:
            raw = self._fetch(TAIPEI_URL)
            return [
                {
                    "id": r.get("sno"),
                    "name": r.get("sna"),
                    "area": r.get("sarea"),
                    "city": "台北市",
                    "lat": _to_float(r.get("latitude")),
                    "lng": _to_float(r.get("longitude")),
                    "available_rent_bikes": _to_int(r.get("available_rent_bikes")),
                    "available_return_bikes": _to_int(r.get("available_return_bikes")),
                    "total": _to_int(r.get("Quantity")),
                    "active": r.get("act") == "1",
                    "update_time": r.get("updateTime"),
                }
                for r in raw
            ]

        data, _ = self.cached_call(
            "youbike_taipei", fetcher, params=None, ttl=SETTINGS.ttl.youbike,
        )
        return data

    def _stations_newtaipei(self) -> list[dict]:
        def fetcher() -> list[dict]:
            raw = self._fetch(NEWTAIPEI_URL)
            return [
                {
                    "id": r.get("sno"),
                    "name": r.get("sna"),
                    "area": r.get("sarea"),
                    "city": "新北市",
                    "lat": _to_float(r.get("lat")),
                    "lng": _to_float(r.get("lng")),
                    "available_rent_bikes": _to_int(r.get("sbi_quantity")),
                    "available_return_bikes": _to_int(r.get("bemp")),
                    "total": _to_int(r.get("tot_quantity")),
                    "active": r.get("act") == "1",
                    "update_time": r.get("mday"),
                }
                for r in raw
            ]

        data, _ = self.cached_call(
            "youbike_newtaipei", fetcher, params=None, ttl=SETTINGS.ttl.youbike,
        )
        return data

    def _stations_taoyuan(self) -> list[dict]:
        """桃園市站點：用官方 YouBike 全站 JSON 過濾 area_code='07'。"""
        def fetcher() -> list[dict]:
            raw = self._fetch(YOUBIKE_OFFICIAL_URL)
            out = []
            for r in raw:
                if r.get("area_code") != TAOYUAN_AREA_CODE:
                    continue
                out.append({
                    "id": r.get("station_no"),
                    "name": r.get("name_tw"),
                    "area": r.get("district_tw"),
                    "city": "桃園市",
                    "lat": _to_float(r.get("lat")),
                    "lng": _to_float(r.get("lng")),
                    "available_rent_bikes": _to_int(r.get("available_spaces")),
                    "available_return_bikes": _to_int(r.get("empty_spaces")),
                    "total": _to_int(r.get("parking_spaces")),
                    "active": r.get("status") == 1,
                    "update_time": r.get("updated_at"),
                })
            return out

        data, _ = self.cached_call(
            "youbike_taoyuan", fetcher, params=None, ttl=SETTINGS.ttl.youbike,
        )
        return data

    def get_all_stations(self) -> list[dict]:
        """合併台北市 + 新北市 + 桃園市站點。任一來源失敗仍回傳其他。"""
        out: list[dict] = []
        for fn, label in [
            (self._stations_taipei, "Taipei"),
            (self._stations_newtaipei, "NewTaipei"),
            (self._stations_taoyuan, "Taoyuan"),
        ]:
            try:
                out.extend(fn())
            except APIError as e:
                logger.warning("YouBike %s fetch failed: %s", label, e)
        return out

    def nearest_stations(
        self,
        lat: float,
        lng: float,
        *,
        k: int = 3,
        max_radius_m: float = 800.0,
        require_bikes: bool = True,
    ) -> list[dict]:
        """
        找最近的 k 個站點。
        - require_bikes=True：rent（取車）端要至少 1 台
        - 結果欄位額外含 `distance_m`
        """
        stations = self.get_all_stations()
        scored: list[tuple[float, dict]] = []
        for s in stations:
            if not s["active"]:
                continue
            if require_bikes and s["available_rent_bikes"] < 1:
                continue
            if s["lat"] == 0 and s["lng"] == 0:
                continue
            d = _haversine_m(lat, lng, s["lat"], s["lng"])
            if d > max_radius_m:
                continue
            scored.append((d, {**s, "distance_m": int(d)}))
        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored[:k]]
