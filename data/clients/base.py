"""
data.clients.base — 共用 API client 基底

提供：
- 本地檔案快取（layered TTL，key = source:endpoint:hash(params)）
- 指數退避 retry（最多 3 次嘗試）
- stale-on-error：呼叫失敗時若有舊快取則回傳並標記
- 統一 logging

設計取捨：
- 不引入 Redis；用 JSON 檔即可滿足 demo + 本地開發
- 快取檔案以 hash 命名避免 path injection / 過長檔名
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """快取讀取結果。"""
    data: Any
    fetched_at: float        # unix timestamp
    is_stale: bool           # True = 已過 TTL，但呼叫端決定要不要用


class APIError(Exception):
    """所有 client 對外丟的錯誤；上層 planner 用來決定 fallback。"""


class BaseAPIClient:
    """子類別需定義 source_name。"""

    source_name: str = "base"
    default_timeout: float = 10.0
    max_retries: int = 3
    backoff_base: float = 0.5   # 第 n 次 retry 等 backoff_base * 2^n 秒

    def __init__(self) -> None:
        self.cache_dir: Path = SETTINGS.cache_dir / self.source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    # ── 快取 ───────────────────────────────────────────────────

    def _cache_key(self, endpoint: str, params: dict | None) -> str:
        payload = json.dumps(
            {"endpoint": endpoint, "params": params or {}},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def cache_get(self, endpoint: str, params: dict | None, ttl: int) -> CachedResponse | None:
        path = self._cache_path(self._cache_key(endpoint, params))
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("cache read failed for %s: %s", path, e)
            return None
        fetched_at = float(blob.get("fetched_at", 0))
        age = time.time() - fetched_at
        return CachedResponse(
            data=blob.get("data"),
            fetched_at=fetched_at,
            is_stale=age > ttl,
        )

    def cache_put(self, endpoint: str, params: dict | None, data: Any) -> None:
        path = self._cache_path(self._cache_key(endpoint, params))
        blob = {"fetched_at": time.time(), "endpoint": endpoint, "data": data}
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False)
            tmp.replace(path)
        except OSError as e:
            logger.warning("cache write failed for %s: %s", path, e)

    # ── HTTP with retry ────────────────────────────────────────

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        timeout = timeout or self.default_timeout
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(
                    method, url,
                    params=params, json=json_body, headers=headers, timeout=timeout,
                )
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                logger.warning(
                    "%s %s returned %d (attempt %d/%d)",
                    method, url, resp.status_code, attempt + 1, self.max_retries,
                )
            except requests.RequestException as e:
                last_exc = e
                logger.warning(
                    "%s %s raised %s (attempt %d/%d)",
                    method, url, e, attempt + 1, self.max_retries,
                )
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base * (2 ** attempt))
        if last_exc:
            raise APIError(f"{method} {url} failed after {self.max_retries} attempts: {last_exc}") from last_exc
        raise APIError(f"{method} {url} failed after {self.max_retries} attempts (HTTP error)")

    # ── 對外主 entry：cache + fetch + stale-on-error ───────────

    def cached_call(
        self,
        endpoint: str,
        fetcher: Callable[[], Any],
        *,
        params: dict | None,
        ttl: int,
        allow_stale_on_error: bool = True,
    ) -> tuple[Any, str]:
        """
        通用入口：
        1. 命中且未過期 → 直接回 (data, "cache")
        2. 過期或無快取 → 嘗試呼叫 fetcher() 寫入後回 (data, "live")
        3. fetcher 失敗且有舊快取 → 回 (data, "cache-stale")
        4. fetcher 失敗且無快取 → raise APIError
        """
        cached = self.cache_get(endpoint, params, ttl)
        if cached and not cached.is_stale:
            return cached.data, "cache"

        try:
            data = fetcher()
        except APIError:
            if allow_stale_on_error and cached is not None:
                logger.warning("fetch failed for %s — serving stale cache", endpoint)
                return cached.data, "cache-stale"
            raise

        self.cache_put(endpoint, params, data)
        return data, "live"

    def clear_cache(self) -> int:
        """清空此 client 的快取，回傳刪除檔案數。"""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
        return count
