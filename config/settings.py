"""
config.settings — 環境變數與全域設定

讀取 .env，提供：
- API 金鑰（Google / TDX）
- 快取目錄與分層 TTL
- realtime 旗標（缺金鑰時自動 fallback）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CacheTTL:
    """分層 TTL（秒）— 對應架構文件 §5。"""
    static_metadata: int = 86400      # 站點 / 路線清單
    timetable: int = 21600            # OD 時刻表
    directions: int = 300             # Google Directions（含路況）
    bus_eta: int = 30                 # 公車到站
    youbike: int = 60                 # YouBike 站點狀態


@dataclass(frozen=True)
class Settings:
    google_maps_api_key: str = ""
    tdx_client_id: str = ""
    tdx_client_secret: str = ""
    enable_realtime: bool = True
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    ttl: CacheTTL = field(default_factory=CacheTTL)

    @property
    def has_google(self) -> bool:
        return bool(self.google_maps_api_key)

    @property
    def has_tdx(self) -> bool:
        return bool(self.tdx_client_id and self.tdx_client_secret)

    def missing_keys(self) -> list[str]:
        miss = []
        if not self.has_google:
            miss.append("GOOGLE_MAPS_API_KEY")
        if not self.has_tdx:
            miss.append("TDX_CLIENT_ID / TDX_CLIENT_SECRET")
        return miss


def load_settings() -> Settings:
    cache_dir_raw = os.getenv("CACHE_DIR", "").strip()
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else PROJECT_ROOT / "data" / "cache"
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    ttl = CacheTTL(
        static_metadata=_env_int("TTL_STATIC_METADATA", 86400),
        timetable=_env_int("TTL_TIMETABLE", 21600),
        directions=_env_int("TTL_DIRECTIONS", 300),
        bus_eta=_env_int("TTL_BUS_ETA", 30),
        youbike=_env_int("TTL_YOUBIKE", 60),
    )

    return Settings(
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        tdx_client_id=os.getenv("TDX_CLIENT_ID", "").strip(),
        tdx_client_secret=os.getenv("TDX_CLIENT_SECRET", "").strip(),
        enable_realtime=_env_bool("ENABLE_REALTIME", True),
        cache_dir=cache_dir,
        ttl=ttl,
    )


SETTINGS = load_settings()
