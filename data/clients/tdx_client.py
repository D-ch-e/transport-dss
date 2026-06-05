"""
data.clients.tdx_client — TDX 運輸資料流通服務 v3

支援端點：
- OAuth2 client_credentials token（自動 cache 22 小時，TDX token TTL = 24h）
- 高鐵 THSR：DailyTimetable / OD（指定起訖站 + 日期 → 班次清單）
- 台鐵 TRA：DailyTimetable / OD
- 公車：EstimatedTimeOfArrival（City + RouteName）
- 捷運：StationOfRoute（路線站序）+ LiveBoard（即時到站，限 TRTC / KRTC 等）

注意：
- TDX v3 base：https://tdx.transportdata.tw/api/basic/v3
- 票價不打此 client（架構文件 §1 已決定走靜態 CSV）
- 站名 ↔ StationID 對照表收在 _STATIONS，給最常用的北部站。未涵蓋者可直接傳 ID。
"""

from __future__ import annotations

import logging
import time
from datetime import date

from config.settings import SETTINGS
from data.clients.base import APIError, BaseAPIClient

logger = logging.getLogger(__name__)

TDX_AUTH_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
TDX_BASE = "https://tdx.transportdata.tw/api/basic"

# 北部地區常用車站速查（站名 → StationID）
# THSR / TRA 皆為 4 碼字串。台 / 臺 互通：lookup_station 會自動轉換。
_STATIONS: dict[str, dict[str, str]] = {
    "THSR": {
        "南港": "0990", "台北": "1000", "板橋": "1010",
        "桃園": "1020", "新竹": "1030", "苗栗": "1035",
        "台中": "1040", "彰化": "1043", "雲林": "1047",
        "嘉義": "1050", "台南": "1060", "左營": "1070",
    },
    "TRA": {
        # 北部 / 桃園線
        "基隆": "0900", "七堵": "0930",
        "南港": "0980", "松山": "0990", "台北": "1000",
        "萬華": "1010", "板橋": "1020", "樹林": "1040",
        "桃園": "1080", "中壢": "1100", "新竹": "1210",
    },
}


class TDXClient(BaseAPIClient):
    source_name = "tdx"

    def __init__(self) -> None:
        super().__init__()
        self.client_id = SETTINGS.tdx_client_id
        self.client_secret = SETTINGS.tdx_client_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        if not (self.client_id and self.client_secret):
            logger.warning("TDX credentials missing — TDXClient will refuse live calls")

    # ── OAuth ─────────────────────────────────────────────────

    def _require_creds(self) -> None:
        if not (self.client_id and self.client_secret):
            raise APIError("TDX credentials not configured")

    def _get_token(self) -> str:
        # in-memory cache（避免每次 request 都打 auth）
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        self._require_creds()
        # token endpoint 用 application/x-www-form-urlencoded，直接呼叫 session.post
        # （base 的 _request_with_retry 只走 JSON / query，故此處單獨處理）
        resp = self.session.post(
            TDX_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.default_timeout,
        )
        if resp.status_code != 200:
            raise APIError(f"TDX auth failed: {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        self._token = body["access_token"]
        # TDX token 預設 86400 秒（24h），保守抓 22h
        self._token_expires_at = time.time() + min(int(body.get("expires_in", 86400)), 79200)
        return self._token

    def _get_json(self, path: str, params: dict | None = None) -> list | dict:
        token = self._get_token()
        url = f"{TDX_BASE}{path}"
        full_params = {"$format": "JSON", **(params or {})}
        resp = self._request_with_retry(
            "GET", url,
            params=full_params,
            headers={"authorization": f"Bearer {token}", "accept-encoding": "gzip"},
        )
        if resp.status_code == 401:
            # token 過期就強制重抓一次
            self._token = None
            token = self._get_token()
            resp = self._request_with_retry(
                "GET", url,
                params=full_params,
                headers={"authorization": f"Bearer {token}"},
            )
        if resp.status_code >= 400:
            raise APIError(f"TDX {path} returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ── 站名工具 ──────────────────────────────────────────────

    @staticmethod
    def lookup_station(system: str, name: str) -> str | None:
        """system: 'THSR' or 'TRA'。回傳 StationID 或 None。台↔臺自動互通。"""
        table = _STATIONS.get(system, {})
        if name in table:
            return table[name]
        # 繁簡互通：臺北 ↔ 台北 / 臺中 ↔ 台中
        alt = name.replace("臺", "台") if "臺" in name else name.replace("台", "臺")
        return table.get(alt)

    # ── 高鐵 THSR ─────────────────────────────────────────────

    def thsr_od_timetable(
        self,
        origin_station_id: str,
        dest_station_id: str,
        train_date: str | None = None,
    ) -> list[dict]:
        """
        指定起訖站 + 日期，回傳當日所有班次。
        train_date：YYYY-MM-DD，預設今天。

        normalize 後欄位：
            {"train_no", "depart_time", "arrive_time", "duration_min"}
        """
        if train_date is None:
            train_date = date.today().isoformat()
        path = f"/v2/Rail/THSR/DailyTimetable/OD/{origin_station_id}/to/{dest_station_id}/{train_date}"
        params = {"$top": 200, "$orderby": "OriginStopTime/DepartureTime"}

        def fetcher() -> list[dict]:
            items = self._get_json(path, params)
            if not isinstance(items, list):
                return []
            out = []
            for it in items:
                ti = it.get("DailyTrainInfo", {})
                dep = it.get("OriginStopTime", {}).get("DepartureTime", "")
                arr = it.get("DestinationStopTime", {}).get("ArrivalTime", "")
                out.append({
                    "train_no": ti.get("TrainNo", ""),
                    "depart_time": dep,
                    "arrive_time": arr,
                    "duration_min": _hhmm_diff_min(dep, arr),
                })
            return out

        data, _ = self.cached_call(path, fetcher, params=params, ttl=SETTINGS.ttl.timetable)
        return data

    # ── 台鐵 TRA ──────────────────────────────────────────────

    def tra_od_timetable(
        self,
        origin_station_id: str,
        dest_station_id: str,
        train_date: str | None = None,
    ) -> list[dict]:
        """
        台鐵 OD 時刻表。normalize 後欄位同 THSR，並多回 train_type（自強/區間/普悠瑪…）。
        """
        if train_date is None:
            train_date = date.today().isoformat()
        path = f"/v2/Rail/TRA/DailyTimetable/OD/{origin_station_id}/to/{dest_station_id}/{train_date}"
        params = {"$top": 300, "$orderby": "OriginStopTime/DepartureTime"}

        def fetcher() -> list[dict]:
            items = self._get_json(path, params)
            if not isinstance(items, list):
                return []
            out = []
            for it in items:
                ti = it.get("DailyTrainInfo", {})
                dep = it.get("OriginStopTime", {}).get("DepartureTime", "")
                arr = it.get("DestinationStopTime", {}).get("ArrivalTime", "")
                tt = ti.get("TrainTypeName", {})
                train_type = tt.get("Zh_tw") if isinstance(tt, dict) else str(tt)
                out.append({
                    "train_no": ti.get("TrainNo", ""),
                    "train_type": train_type or "",
                    "depart_time": dep,
                    "arrive_time": arr,
                    "duration_min": _hhmm_diff_min(dep, arr),
                })
            return out

        data, _ = self.cached_call(path, fetcher, params=params, ttl=SETTINGS.ttl.timetable)
        return data

    # ── 公車 ETA ──────────────────────────────────────────────

    def bus_eta(self, city: str, route_name: str | None = None) -> list[dict]:
        """
        公車即時到站。city 為英文（e.g. 'Taipei', 'NewTaipei', 'Taoyuan'）。
        route_name 可省略 → 全市；建議指定減少回傳量。

        normalize 欄位：
            {"route_name", "stop_name", "direction", "estimate_sec"}
        """
        if route_name:
            path = f"/v2/Bus/EstimatedTimeOfArrival/City/{city}/{route_name}"
        else:
            path = f"/v2/Bus/EstimatedTimeOfArrival/City/{city}"
        params = {"$top": 500}

        def fetcher() -> list[dict]:
            items = self._get_json(path, params)
            if not isinstance(items, list):
                return []
            out = []
            for it in items:
                sn = it.get("StopName", {})
                stop_name = sn.get("Zh_tw") if isinstance(sn, dict) else str(sn)
                rn = it.get("RouteName", {})
                route = rn.get("Zh_tw") if isinstance(rn, dict) else rn
                out.append({
                    "route_name": route,
                    "stop_name": stop_name,
                    "direction": it.get("Direction"),
                    "estimate_sec": it.get("EstimateTime"),   # None = 尚未發車 / 末班已過
                })
            return out

        data, _ = self.cached_call(path, fetcher, params=params, ttl=SETTINGS.ttl.bus_eta)
        return data

    # ── 捷運 Metro ────────────────────────────────────────────

    def metro_liveboard(self, operator: str = "TRTC") -> list[dict]:
        """
        operator: TRTC（台北捷運）/ KRTC（高雄）/ KLRT / TYMC（桃捷）
        回傳各站下班列車預估到站時間。
        """
        path = f"/v2/Rail/Metro/LiveBoard/{operator}"
        params = {"$top": 1000}

        def fetcher() -> list[dict]:
            items = self._get_json(path, params)
            if not isinstance(items, list):
                return []
            out = []
            for it in items:
                sn = it.get("StationName", {})
                station_name = sn.get("Zh_tw") if isinstance(sn, dict) else str(sn)
                out.append({
                    "station_id": it.get("StationID"),
                    "station_name": station_name,
                    "line_no": it.get("LineNo"),
                    "trip_head_sign": it.get("TripHeadSign"),
                    "estimate_sec": it.get("EstimateTime"),
                })
            return out

        data, _ = self.cached_call(path, fetcher, params=params, ttl=SETTINGS.ttl.bus_eta)
        return data


# ── helpers ───────────────────────────────────────────────────

def _hhmm_diff_min(start: str, end: str) -> int:
    """'08:15' / '09:42' → 87；跨夜會回正值。"""
    try:
        sh, sm = map(int, start.split(":")[:2])
        eh, em = map(int, end.split(":")[:2])
        diff = (eh * 60 + em) - (sh * 60 + sm)
        if diff < 0:
            diff += 24 * 60
        return diff
    except (ValueError, AttributeError):
        return 0
