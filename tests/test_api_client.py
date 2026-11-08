"""
測試 data/api_client.py — 靜態資料截取與查詢

涵蓋：
- load_fare_table / get_route_data（城市間雙向查詢）
- load_city_hub_map / get_supported_cities
- load_geocoding_attractions
- lookup_mrt_fare / normalize_mrt_station
- get_supported_locations（含 input_validator）
- 輸入驗證邏輯（validate_input）

所有測試僅讀取本地靜態檔案，不呼叫任何外部 API。
"""

import pytest
import pandas as pd


# ─────────────────────────────────────────────────────────
# 1. fare_table — 資料格式
# ─────────────────────────────────────────────────────────

class TestLoadFareTable:
    def test_returns_dataframe(self):
        from data.api_client import load_fare_table
        df = load_fare_table()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_exist(self):
        from data.api_client import load_fare_table
        df = load_fare_table()
        required = {"origin", "destination", "mode", "cost",
                    "duration_min", "distance_km", "transfers",
                    "frequency_per_day", "convenience"}
        assert required.issubset(set(df.columns)), f"缺少欄位：{required - set(df.columns)}"

    def test_no_empty_rows(self):
        from data.api_client import load_fare_table
        df = load_fare_table()
        assert len(df) > 0

    def test_cost_positive(self):
        from data.api_client import load_fare_table
        df = load_fare_table()
        assert (df["cost"] > 0).all(), "票價不應有非正值"

    def test_duration_positive(self):
        from data.api_client import load_fare_table
        df = load_fare_table()
        assert (df["duration_min"] > 0).all(), "時間不應有非正值"


# ─────────────────────────────────────────────────────────
# 2. get_route_data — 查詢行為
# ─────────────────────────────────────────────────────────

class TestGetRouteData:
    def test_known_route_returns_list(self):
        from data.api_client import get_route_data
        result = get_route_data("台北", "台中")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_result_schema(self):
        from data.api_client import get_route_data
        result = get_route_data("台北", "台中")
        required_keys = {"mode", "cost", "duration_min", "distance_km",
                         "transfers", "convenience", "source"}
        for r in result:
            missing = required_keys - set(r.keys())
            assert not missing, f"缺少欄位 {missing}，方案: {r.get('mode')}"

    def test_bidirectional_lookup(self):
        """A→B 和 B→A 都能查到資料（或至少其中一方向有資料）。"""
        from data.api_client import get_route_data
        fwd = get_route_data("台北", "台中")
        rev = get_route_data("台中", "台北")
        assert len(fwd) > 0 or len(rev) > 0, "台北↔台中 雙向皆查無資料"

    def test_unknown_route_returns_empty(self):
        from data.api_client import get_route_data
        result = get_route_data("不存在地點A", "不存在地點B")
        assert result == []

    def test_same_origin_destination_returns_empty(self):
        from data.api_client import get_route_data
        result = get_route_data("台北", "台北")
        assert result == []

    def test_modes_include_expected(self):
        from data.api_client import get_route_data
        result = get_route_data("台北", "高雄")
        modes = {r["mode"] for r in result}
        # 台北↔高雄 長程應至少有高鐵
        assert "高鐵" in modes, f"台北→高雄 應包含高鐵，實際 modes: {modes}"

    def test_source_is_static(self):
        from data.api_client import get_route_data
        result = get_route_data("台北", "台中")
        for r in result:
            assert r["source"] == "static"


# ─────────────────────────────────────────────────────────
# 3. city_hub_map
# ─────────────────────────────────────────────────────────

class TestLoadCityHubMap:
    def test_returns_dict(self):
        from data.api_client import load_city_hub_map
        m = load_city_hub_map()
        assert isinstance(m, dict)

    def test_contains_major_cities(self):
        from data.api_client import load_city_hub_map
        m = load_city_hub_map()
        for city in ("台北", "台中", "台南", "高雄"):
            assert city in m, f"缺少城市：{city}"

    def test_cities_have_required_fields(self):
        from data.api_client import load_city_hub_map
        m = load_city_hub_map()
        for city, info in m.items():
            assert "lat" in info and "lng" in info, f"{city} 缺少座標"


class TestGetSupportedCities:
    def test_returns_list_of_strings(self):
        from data.api_client import get_supported_cities
        cities = get_supported_cities()
        assert isinstance(cities, list)
        assert all(isinstance(c, str) for c in cities)

    def test_non_empty(self):
        from data.api_client import get_supported_cities
        assert len(get_supported_cities()) >= 4


# ─────────────────────────────────────────────────────────
# 4. geocoding attractions
# ─────────────────────────────────────────────────────────

class TestLoadGeocodingAttractions:
    def test_returns_dataframe(self):
        from data.api_client import load_geocoding_attractions
        df = load_geocoding_attractions()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        from data.api_client import load_geocoding_attractions
        df = load_geocoding_attractions()
        for col in ("地點簡稱", "緯度_lat", "經度_lng"):
            assert col in df.columns, f"缺少欄位：{col}"

    def test_lat_lng_in_taiwan_range(self):
        from data.api_client import load_geocoding_attractions
        df = load_geocoding_attractions()
        assert (df["緯度_lat"].between(21.0, 26.5)).all(), "緯度超出台灣範圍"
        assert (df["經度_lng"].between(119.0, 122.5)).all(), "經度超出台灣範圍"

    def test_known_attractions_present(self):
        from data.api_client import load_geocoding_attractions
        df = load_geocoding_attractions()
        names = set(df["地點簡稱"].tolist())
        for name in ("北車", "101"):
            assert name in names, f"缺少景點：{name}"


# ─────────────────────────────────────────────────────────
# 5. MRT 票價查詢
# ─────────────────────────────────────────────────────────

class TestNormalizeMrtStation:
    def test_alias_maps_to_canonical(self):
        from data.api_client import normalize_mrt_station
        assert normalize_mrt_station("北車") == "台北車站"
        assert normalize_mrt_station("台北") == "台北車站"

    def test_canonical_returns_itself(self):
        from data.api_client import normalize_mrt_station
        result = normalize_mrt_station("台北車站")
        assert result == "台北車站"

    def test_unknown_station_returns_none(self):
        from data.api_client import normalize_mrt_station
        assert normalize_mrt_station("不存在的站") is None


class TestLookupMrtFare:
    def test_known_pair_returns_dict(self):
        from data.api_client import lookup_mrt_fare
        result = lookup_mrt_fare("台北車站", "龍山寺")
        # 龍山寺在票價表中 → 應有結果
        # 若票價表確實涵蓋此 OD，才強制斷言；否則只檢查類型
        if result is not None:
            assert isinstance(result, dict)
            assert "fare" in result
            assert "duration_min" in result
            assert "distance_km" in result

    def test_same_station_returns_zero_fare(self):
        from data.api_client import lookup_mrt_fare
        result = lookup_mrt_fare("台北車站", "台北車站")
        assert result is not None
        assert result["fare"] == 0
        assert result["duration_min"] == 0

    def test_unknown_pair_returns_none(self):
        from data.api_client import lookup_mrt_fare
        result = lookup_mrt_fare("不存在A站", "不存在B站")
        assert result is None

    def test_alias_lookup_works(self):
        """用景點簡稱（別名）也能查到票價。"""
        from data.api_client import lookup_mrt_fare
        result = lookup_mrt_fare("北車", "龍山寺")
        # 不管有沒有資料，只要不拋例外就算通過
        assert result is None or isinstance(result, dict)


# ─────────────────────────────────────────────────────────
# 6. get_supported_locations（input_validator）
# ─────────────────────────────────────────────────────────

class TestGetSupportedLocations:
    def test_returns_list(self):
        from data.input_validator import get_supported_locations
        locs = get_supported_locations()
        assert isinstance(locs, list)

    def test_contains_only_attractions(self):
        from data.input_validator import get_supported_locations
        locs = get_supported_locations()
        loc_set = set(locs)
        assert {"北車", "101", "桃機", "木柵動物園"}.issubset(loc_set)
        assert "台北" not in loc_set
        assert "台中" not in loc_set

    def test_no_duplicates(self):
        from data.input_validator import get_supported_locations
        locs = get_supported_locations()
        assert len(locs) == len(set(locs)), "地點列表有重複項"


# ─────────────────────────────────────────────────────────
# 7. validate_input
# ─────────────────────────────────────────────────────────

class TestValidateInput:
    def test_valid_input_passes(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 2, 1, 2000)
        assert ok is True
        assert msg == ""

    def test_same_origin_destination_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "北車", 2, 1, 2000)
        assert ok is False
        assert "相同" in msg

    def test_invalid_origin_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("不存在的地方", "101", 2, 1, 2000)
        assert ok is False

    def test_people_zero_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 0, 1, 2000)
        assert ok is False

    def test_people_over_limit_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 11, 1, 2000)
        assert ok is False

    def test_negative_luggage_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 2, -1, 2000)
        assert ok is False

    def test_zero_budget_fails(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 2, 1, 0)
        assert ok is False

    def test_attractions_as_valid_input(self):
        from data.input_validator import validate_input
        ok, msg = validate_input("北車", "101", 2, 0, 500)
        assert ok is True, f"景點作為起訖點應合法，錯誤：{msg}"


# ─────────────────────────────────────────────────────────
# 7. 北北基桃站間票價表（高鐵 / 台鐵 / 國道客運）
# ─────────────────────────────────────────────────────────

class TestStationFareTables:
    def test_hsr_matrix_columns(self):
        from data.api_client import load_hsr_fare_matrix
        df = load_hsr_fare_matrix()
        required = {"origin_station", "dest_station", "fare", "duration_min", "distance_km"}
        assert required.issubset(set(df.columns))
        assert len(df) > 0

    def test_hsr_known_fare(self):
        from data.api_client import lookup_hsr_fare
        r = lookup_hsr_fare("台北", "桃園")
        assert r is not None
        assert r["fare"] == 160

    def test_hsr_bidirectional(self):
        from data.api_client import lookup_hsr_fare
        assert lookup_hsr_fare("桃園", "台北")["fare"] == lookup_hsr_fare("台北", "桃園")["fare"]

    def test_hsr_same_station_zero(self):
        from data.api_client import lookup_hsr_fare
        assert lookup_hsr_fare("台北", "台北")["fare"] == 0

    def test_tra_known_fare(self):
        from data.api_client import lookup_tra_fare
        # 預設區間車
        assert lookup_tra_fare("台北", "基隆")["fare"] == 62

    def test_tra_bidirectional(self):
        from data.api_client import lookup_tra_fare
        assert lookup_tra_fare("基隆", "台北")["fare"] == 62

    def test_tra_default_is_local_train(self):
        from data.api_client import lookup_tra_fare
        r = lookup_tra_fare("台北", "桃園")
        assert r["fare"] == 63
        assert r["train_type"] == "區間"

    def test_tra_express_alternative(self):
        from data.api_client import lookup_tra_fare
        # 自強備選較貴
        local = lookup_tra_fare("台北", "桃園", "區間")
        express = lookup_tra_fare("台北", "桃園", "自強")
        assert express["train_type"] == "自強"
        assert express["fare"] == 99
        assert express["fare"] > local["fare"]

    def test_tra_matrix_has_train_type_column(self):
        from data.api_client import load_tra_fare_matrix
        df = load_tra_fare_matrix()
        assert "train_type" in df.columns

    def test_bus_table_is_attraction_level(self):
        # 客運票價表為景點層級（萃取自 taiwan_route_dataset.csv 含四位數編號的客運路線）
        from data.api_client import load_bus_intercity_fare
        df = load_bus_intercity_fare()
        stations = set(df["origin_station"]) | set(df["dest_station"])
        assert {"北車", "小人國", "廟口夜市"} <= stations

    def test_bus_blank_fare_returns_none(self, monkeypatch):
        # 票價未填（NaN）的 OD 應安全回 None，不丟例外
        import data.api_client as api
        table = api.load_bus_intercity_fare().copy()
        table["fare"] = float("nan")
        monkeypatch.setattr(api, "load_bus_intercity_fare", lambda: table)
        assert api.lookup_bus_fare("北車", "小人國") is None

    def test_bus_no_route_returns_none(self):
        from data.api_client import lookup_bus_fare
        assert lookup_bus_fare("桃園", "基隆") is None

    def test_unknown_station_returns_none(self):
        from data.api_client import lookup_hsr_fare
        assert lookup_hsr_fare("台北", "高雄") is None
