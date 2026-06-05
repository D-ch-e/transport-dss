"""
測試計價邏輯 — YouBike 計費與 iRent 租車時租估算。

- _youbike_fee：前 30 分免費，之後每 30 分 10 元
- _estimate_car_cost：依車型分級（時租 × 時數 + 里程費 + 過路費）

僅測試純函式，不呼叫任何外部 API。
"""


class TestYouBikeFee:
    def test_within_30min_free(self):
        from services.multimodal import _youbike_fee, _youbike_leg_minutes
        # 5600m ≈ 30 分鐘，應免費
        assert _youbike_leg_minutes(5600) <= 30
        assert _youbike_fee(5600) == 0

    def test_short_ride_free(self):
        from services.multimodal import _youbike_fee
        assert _youbike_fee(1000) == 0

    def test_just_over_30min_charges_10(self):
        from services.multimodal import _youbike_fee, _youbike_leg_minutes
        assert _youbike_leg_minutes(5800) == 31
        assert _youbike_fee(5800) == 10

    def test_61min_charges_20(self):
        from services.multimodal import _youbike_fee, _youbike_leg_minutes
        assert _youbike_leg_minutes(12000) == 62
        assert _youbike_fee(12000) == 20


class TestEstimateCarCost:
    def test_returns_cost_and_car_type(self):
        from services.route_planner import _estimate_car_cost
        cost, car_type = _estimate_car_cost(50, 30, 2, 40)
        assert isinstance(cost, int)
        assert isinstance(car_type, str)

    def test_hourly_plus_mileage_plus_toll(self):
        from services.route_planner import _estimate_car_cost
        # 最便宜車型 PRIUS C 250/hr × ceil(40/60)=1hr + 50×3.4(車型里程費) + 30×1.2 = 250+170+36 = 456
        cost, _ = _estimate_car_cost(50, 30, 2, 40)
        assert cost == 456

    def test_rental_hours_round_up(self):
        from services.route_planner import _estimate_car_cost
        # 80 分鐘 → 2 小時
        cost_80, _ = _estimate_car_cost(0, 0, 1, 80)
        cost_61, _ = _estimate_car_cost(0, 0, 1, 61)
        assert cost_80 == cost_61  # 同為 2 小時計費

    def test_picks_seven_seater_for_six_people(self):
        from services.route_planner import _estimate_car_cost
        # 6 人 → 需 7 人座，車隊中唯一可載者為 SIENTA
        _, car_type = _estimate_car_cost(0, 0, 6, 60)
        assert car_type == "SIENTA"

    def test_picks_cheapest_fitting_for_small_group(self):
        from services.route_planner import _estimate_car_cost
        # 2 人無行李 → 取最便宜車型 PRIUS C（250/hr）
        _, car_type = _estimate_car_cost(0, 0, 2, 60)
        assert car_type == "PRIUS C"


class TestSelectCarType:
    def test_no_luggage_picks_cheapest(self):
        from services.route_planner import _select_car_type
        car, fits = _select_car_type(2, 0)
        assert fits is True
        assert car["name"] == "PRIUS C"

    def test_heavy_luggage_picks_larger_capacity(self):
        from services.route_planner import _select_car_type
        # 4 件大行李，小型車(2)裝不下 → 應選容量足夠的車型
        car, fits = _select_car_type(2, 4)
        assert fits is True
        assert car["max_luggage"] >= 4

    def test_over_capacity_still_returns_but_marks_unfit(self):
        from services.route_planner import _select_car_type
        # 遠超任何車型行李容量 → 仍回傳容量最大者，但標記裝不下
        car, fits = _select_car_type(2, 99)
        assert fits is False
        assert car is not None

    def test_unfit_luggage_lowers_convenience(self):
        from services.route_planner import _select_car_type
        _, fits_ok = _select_car_type(2, 1)
        _, fits_bad = _select_car_type(2, 99)
        assert fits_ok is True and fits_bad is False
