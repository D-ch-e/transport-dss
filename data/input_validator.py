"""
M2 · Input Validator — 輸入驗證模組

功能：
- 驗證人數是否在合理範圍內（1~10 人）
- 驗證預算是否為有效正數
- 驗證行李量是否合理（0~10 件）
- 檢查起訖地是否存在於北北基桃景點列表
- 檢查起訖地是否相同
- 回傳可供 UI 顯示的繁體中文錯誤訊息
"""

from data.api_client import load_geocoding_attractions


def get_supported_locations() -> list[str]:
    """北北基桃 10 個景點簡稱，作為合法起訖點。"""
    try:
        attractions = load_geocoding_attractions()["地點簡稱"].tolist()
    except Exception:
        attractions = []

    seen, merged = set(), []
    for n in attractions:
        if n not in seen:
            seen.add(n)
            merged.append(n)
    return merged


def validate_input(
    origin: str,
    destination: str,
    num_people: int,
    luggage: int,
    budget: int,
) -> tuple[bool, str]:
    """
    驗證使用者輸入。

    Parameters
    ----------
    origin : str
        起點景點
    destination : str
        終點景點
    num_people : int
        人數
    luggage : int
        行李件數
    budget : int
        預算（元）

    Returns
    -------
    tuple[bool, str]
        (是否有效, 錯誤訊息)。若有效，訊息為空字串。
    """
    supported = get_supported_locations()

    # 檢查起點
    if origin not in supported:
        return False, f"起點「{origin}」不在支援的地點列表中。支援地點：{', '.join(supported)}"

    # 檢查終點
    if destination not in supported:
        return False, f"終點「{destination}」不在支援的地點列表中。支援地點：{', '.join(supported)}"

    # 檢查起訖地相同
    if origin == destination:
        return False, "起點與終點不能相同，請選擇不同的景點。"

    # 檢查人數
    if not isinstance(num_people, int) or num_people < 1:
        return False, "人數至少為 1 人。"
    if num_people > 10:
        return False, "人數上限為 10 人，超過建議分批規劃。"

    # 檢查行李
    if not isinstance(luggage, int) or luggage < 0:
        return False, "行李件數不得為負數。"
    if luggage > 10:
        return False, "行李件數上限為 10 件。"

    # 檢查預算
    if not isinstance(budget, (int, float)) or budget <= 0:
        return False, "預算需為正數。"

    return True, ""
