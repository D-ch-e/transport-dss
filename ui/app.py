"""
M6 · Streamlit UI — 使用者介面模組

功能：
- 提供側邊欄輸入表單（起訖地、人數、行李量、預算）
- 提供成本、時間、便利性權重滑桿
- 顯示最佳推薦方案
- 顯示各交通方式比較表
- 支援重新計算

執行方式：
    streamlit run ui/app.py
"""

import sys
import os

# 確保專案根目錄在 Python 路徑中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

from data.api_client import get_route_data
from data.input_validator import validate_input, get_supported_locations
from features.feature_engineer import compute_features, build_dt_feature_row
from models.scoring_model import compute_scores
from models.decision_tree import predict_best_mode, FEATURE_COLS
from services.route_planner import plan_routes, youbike_context
from xai.report import build_report
from config.settings import SETTINGS

# ──────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Transport DSS — 可解釋交通決策支援系統",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 自訂樣式
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── 全域字體 ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', 'Inter', sans-serif;
    }

    /* ── 隱藏預設 Streamlit 元件 ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── 側邊欄 ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ── 推薦卡片 ── */
    .recommendation-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(99,102,241,0.3);
    }
    .recommendation-card h2 {
        margin: 0 0 6px 0;
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .recommendation-card .mode-name {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .recommendation-card .score-text {
        font-size: 1.05rem;
        opacity: 0.85;
    }

    /* ── 指標卡片 ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-card .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-card .metric-unit {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* ── 比較表 ── */
    .comparison-table {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* ── 分隔線 ── */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 32px 0;
    }

    /* ── 標題區 ── */
    .page-header {
        padding: 12px 0 8px 0;
    }
    .page-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ecf5ff;
        margin: 0;
    }
    .page-header p {
        color: #64748b;
        font-size: 1rem;
        margin: 4px 0 0 0;
    }

    /* ── 權重顯示 ── */
    .weight-badge {
        display: inline-block;
        background: #000000;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.85rem;
        color: #ffffff;
        font-weight: 500;
        margin-right: 8px;
    }

    /* ── 排名標記 ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }
    .rank-1 { background: linear-gradient(135deg, #f59e0b, #ef4444); }
    .rank-2 { background: #64748b; }
    .rank-3 { background: #94a3b8; }
    .rank-4 { background: #cbd5e1; color: #475569; }
    /* ── 行程路線（多式聯運） ── */
    .itinerary-flow {
        display: flex; flex-wrap: wrap; align-items: center;
        gap: 6px 8px;
        padding: 14px 16px;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        margin-bottom: 14px;
        line-height: 1.4;
    }
    .flow-node {
        display: inline-block;
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 4px 10px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #0f172a;
    }
    .flow-edge {
        display: inline-block;
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        color: #075985;
        text-align: center;
    }
    .flow-edge small { color: #0369a1; font-size: 0.72rem; }
    .flow-arrow { color: #38bdf8; font-weight: 700; padding: 0 2px; }
    .leg-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        height: 100%;
    }
    .leg-card .leg-mode {
        font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 4px;
    }
    .leg-card .leg-from-to {
        font-size: 0.82rem; color: #475569; margin-bottom: 8px;
    }
    .leg-card .leg-stats {
        display: flex; flex-wrap: wrap; gap: 8px;
        font-size: 0.82rem; color: #334155;
    }
    .leg-card .leg-stats span {
        background: #f1f5f9; padding: 2px 8px; border-radius: 6px;
    }
    .fare-breakdown {
        margin-top: 10px;
        padding: 10px 14px;
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #78350f;
    }
    .fare-breakdown b { color: #b45309; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 交通方式 Emoji 對照
# ──────────────────────────────────────────────
MODE_EMOJI = {
    "高鐵": "🚄",
    "台鐵": "🚃",
    "客運": "🚌",
    "租車": "🚗",
    "捷運": "🚇",
    "公車": "🚌",
    "大眾運輸": "🚌",
    "YouBike": "🚲",
    "步行/接駁": "🚶",
    "高鐵+YouBike": "🚄🚲",
    "台鐵+YouBike": "🚃🚲",
    "高鐵+捷運": "🚄🚇",
    "台鐵+捷運": "🚃🚇",
    "高鐵+捷運+YouBike": "🚄🚇🚲",
    "台鐵+捷運+YouBike": "🚃🚇🚲",
}


_MODE_DISPLAY_MAP = {
    "大眾運輸": "公車/客運",
}


def _mode_display(mode: str) -> str:
    return _MODE_DISPLAY_MAP.get(mode, mode)


def _seg_emoji(mode: str) -> str:
    return MODE_EMOJI.get(mode, "🚏")


def render_itinerary(option: dict, *, key_prefix: str = "", num_people: int = 1) -> None:
    """
    把 RouteOption 的 segments 視覺化呈現：
    - 流程圖（emoji + 站名 + 箭頭）
    - 每段卡片（mode / from→to / 票價 / 時間 / 距離）
    - 票價組成與總計
    """
    segs = option.get("segments") or []
    if not segs:
        st.caption("此方案無詳細分段資料。")
        return

    # ── 流程圖 ──
    parts = []
    if segs:
        parts.append(f"<span class='flow-node'>📍 {segs[0].get('from','起點')}</span>")
        for s in segs:
            mode = s.get("mode", "")
            emoji = _seg_emoji(mode)
            line = s.get("line")
            label = f"{emoji} {_mode_display(mode)}" + (f"<br><small>{line}</small>" if line else "")
            parts.append(f"<span class='flow-edge'>{label}</span>")
            parts.append(f"<span class='flow-node'>{s.get('to','')}</span>")
    flow_html = "<span class='flow-arrow'>→</span>".join(parts)
    st.markdown(
        f"<div class='itinerary-flow'>{flow_html}</div>",
        unsafe_allow_html=True,
    )

    # ── 分段卡片 ──
    cols = st.columns(len(segs))
    for i, s in enumerate(segs):
        mode = s.get("mode", "")
        emoji = _seg_emoji(mode)
        cost = int(s.get("cost", 0))
        dur = int(s.get("duration_min", 0))
        dist = float(s.get("distance_km", 0.0))
        cost_txt = f"${cost}" if cost > 0 else "免費"
        dist_txt = f"{dist:.1f} km" if dist > 0 else "—"
        with cols[i]:
            st.markdown(
                f"""
                <div class="leg-card">
                  <div class="leg-mode">{emoji} {_mode_display(mode)}</div>
                  <div class="leg-from-to">{s.get('from','')} → {s.get('to','')}</div>
                  <div class="leg-stats">
                    <span>🎟️ {cost_txt}</span>
                    <span>⏱️ {dur} 分</span>
                    <span>📏 {dist_txt}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 票價組成 ──
    # 租車為整車費用，需除以總人數換算為「每人」票價
    ppl = max(int(num_people), 1)
    paid_segs = [s for s in segs if int(s.get("cost", 0)) > 0]
    if paid_segs:
        def _per_person_cost(s: dict) -> int:
            cost = int(s.get("cost", 0))
            return cost // ppl if s.get("mode") == "租車" else cost
        parts = [f"{s.get('mode','')} ${_per_person_cost(s)}" for s in paid_segs]
        total = sum(_per_person_cost(s) for s in segs)
        st.markdown(
            f"<div class='fare-breakdown'>💰 票價組成：{' + '.join(parts)} = <b>${total}</b> 元 / 人</div>",
            unsafe_allow_html=True,
        )

SOURCE_LABEL = {
    "static": "靜態",
    "google": "Google 即時",
    "tdx+static": "TDX 即時",
    "tdx+static+youbike": "TDX+YouBike",
    "google+youbike": "Google+YouBike",
    "cache": "快取",
    "cache-stale": "舊快取",
}

# ──────────────────────────────────────────────
# 側邊欄：使用者輸入
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚄 交通決策支援系統")
    st.markdown("---")

    locations = get_supported_locations()

    st.markdown("### 📍 路線設定")
    origin = st.selectbox("起點", locations, index=0, key="origin")
    # 預設終點為不同地點
    default_dest = 1 if len(locations) > 1 else 0
    destination = st.selectbox("終點", locations, index=default_dest, key="destination")

    st.markdown("---")
    st.markdown("### 👥 旅行條件")
    num_people = st.number_input("人數", min_value=1, max_value=10, value=2, step=1, key="people")
    luggage = st.number_input("行李件數", min_value=0, max_value=10, value=1, step=1, key="luggage")
    budget = st.number_input("預算上限（元/人）", min_value=0, max_value=10000, key="budget")

    st.markdown("---")
    st.markdown("### ⚖️ 偏好權重")
    st.caption("調整您對成本、時間、便利性的重視程度")
    w_cost = st.slider("💰 成本權重", 0, 10, 5, key="w_cost")
    w_time = st.slider("⏱️ 時間權重", 0, 10, 5, key="w_time")
    w_conv = st.slider("🎒 便利性權重", 0, 10, 3, key="w_conv")

    # 顯示歸一化後的比例
    w_total = w_cost + w_time + w_conv
    if w_total > 0:
        st.markdown(
            f"""
            <div style="margin-top: 8px;">
                <span class="weight-badge">💰 {w_cost/w_total*100:.0f}%</span>
                <span class="weight-badge">⏱️ {w_time/w_total*100:.0f}%</span>
                <span class="weight-badge">🎒 {w_conv/w_total*100:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("請至少設定一項權重大於 0")

    st.markdown("---")
    st.markdown("### 🔌 即時資料來源")
    api_lines = []
    api_lines.append(("Google Maps", SETTINGS.has_google))
    api_lines.append(("TDX 運輸資料", SETTINGS.has_tdx))
    api_lines.append(("YouBike 開放資料", True))   # 免金鑰
    for name, ok in api_lines:
        st.markdown(
            f"<span class='weight-badge'>{'🟢' if ok else '🔴'} {name}</span>",
            unsafe_allow_html=True,
        )
    if not SETTINGS.enable_realtime:
        st.warning("即時模式已關閉（ENABLE_REALTIME=false），全部走靜態資料")

    st.markdown("---")
    col_run, col_refresh = st.columns([3, 2])
    with col_run:
        run_button = st.button("🔍 開始分析", use_container_width=True, type="primary")
    with col_refresh:
        refresh_button = st.button("🔄 重新查詢", use_container_width=True,
                                    help="清除本地 API 快取後重新呼叫")
    if refresh_button:
        # 清掉 google / tdx / youbike 三個來源的本地快取
        cleared = 0
        for sub in ("google", "tdx", "youbike"):
            p = SETTINGS.cache_dir / sub
            if p.exists():
                for f in p.glob("*.json"):
                    try:
                        f.unlink(); cleared += 1
                    except OSError:
                        pass
        st.success(f"已清除 {cleared} 個快取檔案，下次查詢會打即時 API")
        st.session_state["has_run"] = True
        run_button = True

# ──────────────────────────────────────────────
# 主頁面標題
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <h1>🚄 Transport DSS — 交通決策支援系統</h1>
        <p>根據您的條件與偏好，智慧推薦最適合的交通方式，並說明推薦依據。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────
# 分析邏輯
# ──────────────────────────────────────────────
if run_button or st.session_state.get("has_run", False):
    st.session_state["has_run"] = True

    # 輸入驗證
    valid, error_msg = validate_input(origin, destination, num_people, luggage, budget)

    if not valid:
        st.error(f"⚠️ 輸入錯誤：{error_msg}")
    else:
        # 取得路線資料（即時 API + 多式聯運；缺資料時自動 fallback 到靜態）
        with st.spinner(f"正在查詢 {origin} → {destination} 的即時資料..."):
            route_data = plan_routes(
                origin, destination,
                num_people=num_people, luggage=luggage,
            )

        if not route_data:
            st.warning(f"找不到 {origin} → {destination} 的交通資料。請嘗試其他路線。")
        else:
            # 計算特徵
            features = compute_features(route_data, num_people, luggage)

            # 計算分數（傳入預算，過濾超出預算的方案）
            results = compute_scores(features, w_cost, w_time, w_conv, budget=budget)

            if not results:
                st.warning("找不到符合預算上限的交通方案，請提高預算後再試。")
                st.stop()

            # 計算決策樹預測
            distance_km = route_data[0]["distance_km"] if route_data else 0.0
            dt_result = predict_best_mode(
                people=num_people,
                luggage=luggage,
                budget=budget,
                time_pref=w_time,
                cost_pref=w_cost,
                conv_pref=w_conv,
                distance_km=distance_km,
                origin=origin,
                destination=destination,
            )

            import altair as alt

            # 取兩系統首選
            best = results[0]                       # 加權評分首選
            best_emoji = MODE_EMOJI.get(best["mode"], "🚀")

            # 建立 mode → 原始 route_data 的索引（用來抓 source / fetched_at / segments）
            route_by_mode = {rd["mode"]: rd for rd in route_data}

            def _dt_matches(dt_zh: str, score_mode: str) -> bool:
                """寬鬆比對 DT 標籤與評分方案是否指向同一主要交通工具。"""
                if not dt_zh or not score_mode:
                    return False
                if dt_zh == score_mode:
                    return True
                keys = ("高鐵", "台鐵", "租車", "捷運", "客運", "公車")

                def toks(s: str) -> set:
                    t = {k for k in keys if k in s}
                    if "大眾運輸" in s:
                        t |= {"公車", "客運", "捷運"}
                    if "客運" in s:
                        t |= {"公車"}
                    if "公車" in s:
                        t |= {"客運"}
                    return t

                return bool(toks(dt_zh) & toks(score_mode))

            # DT 首選是否落在可比較方案中
            dt_zh = dt_result["mode_zh"] if dt_result else None
            dt_in_table_mode = None
            if dt_zh:
                dt_in_table_mode = next(
                    (r["mode"] for r in results if _dt_matches(dt_zh, r["mode"])),
                    None,
                )

            # ── 兩系統預測摘要（並排小卡） ──
            sum_l, sum_r = st.columns(2)
            with sum_l:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg,#0ea5e9,#6366f1);
                         border-radius:12px; padding:18px 20px; color:white; height:100%;">
                        <div style="font-size:.85rem; opacity:.9;">⚖️ 加權評分首選</div>
                        <div style="font-size:1.8rem; font-weight:700; margin:4px 0;">
                            {best_emoji} {_mode_display(best["mode"])}</div>
                        <div style="font-size:.9rem; opacity:.85;">綜合評分 {best["total_score"]:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sum_r:
                if dt_result:
                    dt_emoji = MODE_EMOJI.get(dt_result["mode_zh"], "🤖")
                    top_prob = max(
                        (v for v in dt_result["probabilities"].values()), default=0.0
                    )
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc; border:1px solid #e2e8f0;
                             border-radius:12px; padding:18px 20px; height:100%;">
                            <div style="font-size:.85rem; color:#64748b;">🤖 決策樹 (DT) 首選</div>
                            <div style="font-size:1.8rem; font-weight:700; margin:4px 0; color:#0f172a;">
                                {dt_emoji} {_mode_display(dt_result["mode_zh"])}</div>
                            <div style="font-size:.9rem; color:#64748b;">預測機率 {top_prob*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="background:#f8fafc; border:1px solid #e2e8f0;
                             border-radius:12px; padding:18px 20px; height:100%;">
                            <div style="font-size:.85rem; color:#64748b;">🤖 決策樹 (DT) 首選</div>
                            <div style="font-size:1.1rem; color:#94a3b8; margin-top:8px;">模型未載入</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── 多方案比較表 ──
            st.markdown("### 📊 多方案比較")
            st.caption("點選表格中的任一列，下方會顯示該方案的行程路線與 XAI 解釋。")
            if dt_zh and dt_in_table_mode is None:
                st.caption(f"ℹ️ DT 首選為 **{_mode_display(dt_zh)}**（不在可比較方案中，未於下表標記）。")

            def _fare_composition(rd: dict) -> str:
                segs = rd.get("segments") or []
                paid = [s for s in segs if int(s.get("cost", 0)) > 0]
                if not paid:
                    return "—"
                from collections import Counter
                parts = []
                modes = [s.get("mode", "") for s in paid]
                counts = Counter(modes)
                for m, c in counts.items():
                    total_cost = sum(int(s.get("cost", 0)) for s in paid if s.get("mode") == m)
                    if m == "租車":
                        per_person = total_cost // max(num_people, 1)
                        label = f"{m} ${per_person}/人"
                    else:
                        label = f"{m}×{c}" if c > 1 else m
                    parts.append(label)
                return " + ".join(parts)

            def _source_badge(mode: str) -> str:
                tags = []
                if mode == best["mode"]:
                    tags.append("⚖️ 評分首選")
                if dt_in_table_mode and mode == dt_in_table_mode:
                    tags.append("🤖 DT首選")
                return " · ".join(tags) if tags else "—"

            comparison_data = []
            for r in results:
                emoji = MODE_EMOJI.get(r["mode"], "")
                in_budget = "✅" if r["cost_raw"] <= budget else "❌"
                raw = route_by_mode.get(r["mode"], {})
                src = SOURCE_LABEL.get(raw.get("source", "static"), raw.get("source", "—"))

                comparison_data.append({
                    "排名": f"#{r['rank']}",
                    "交通方式": f"{emoji} {_mode_display(r['mode'])}",
                    "推薦來源": _source_badge(r["mode"]),
                    "人均成本（元）": f"${r['cost_raw']:.0f}",
                    "預估時間（分鐘）": f"{r['time_raw']:.0f}",
                    "轉乘次數": int(r.get("transfers", raw.get("transfers", 0))),
                    "票價組成": _fare_composition(raw),
                    "便利性分數": f"{r['convenience_raw']:.0f}",
                    "加權總分": f"{r['total_score']:.4f}",
                    "資料來源": src,
                    "預算": in_budget,
                })

            df_comparison = pd.DataFrame(comparison_data)
            table_event = st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="cmp_table",
            )

            # 取得被選列；未選取時預設為評分首選（results[0]）
            sel_rows = []
            try:
                sel_rows = table_event.selection.rows  # type: ignore[attr-defined]
            except AttributeError:
                sel_rows = []
            sel_idx = sel_rows[0] if sel_rows else 0
            selected = results[sel_idx]
            sel_emoji = MODE_EMOJI.get(selected["mode"], "🚀")

            st.markdown("---")

            # ── 選定方案：行程路線 + XAI 解釋（tabs） ──
            st.markdown(f"### 🔎 方案詳情：{sel_emoji} {_mode_display(selected['mode'])}")
            tab_route, tab_xai = st.tabs(["🧭 行程路線", "🔍 XAI 解釋"])

            # ── Tab：行程路線 ──
            with tab_route:
                sel_route = route_by_mode.get(selected["mode"])
                if sel_route and sel_route.get("segments"):
                    transfers = sel_route.get("transfers", 0)
                    st.caption(
                        f"含 **{len(sel_route['segments'])}** 段行程"
                        f"｜**{transfers}** 次轉乘"
                    )
                    render_itinerary(sel_route, key_prefix=f"sel{sel_idx}", num_people=num_people)
                else:
                    st.caption("此方案無分段資料可顯示。")

            # ── Tab：XAI 解釋 ──
            with tab_xai:
                # 用 DT 中對應 selected mode 的路線列建立 feature_row（對不到則 None）
                feature_row = None
                if dt_result and dt_result.get("top_k") and FEATURE_COLS:
                    dt_item = next(
                        (it for it in dt_result["top_k"]
                         if _dt_matches(it["mode_zh"], selected["mode"])),
                        None,
                    ) or dt_result["top_k"][0]
                    feature_row = build_dt_feature_row(
                        route_row=dt_item["_route_row"],
                        people=num_people,
                        luggage=luggage,
                        budget=float(budget),
                        time_pref=w_time,
                        cost_pref=w_cost,
                        conv_pref=w_conv,
                        feature_cols=FEATURE_COLS,
                    )

                report = build_report(
                    results=results,
                    features=features,
                    feature_row=feature_row,
                    w_cost=w_cost,
                    w_time=w_time,
                    w_conv=w_conv,
                    budget=budget,
                    best=selected,
                )

                st.markdown("#### 📌 推薦原因（加權貢獻分解）")
                st.info(report["top_reason"])

                contrib_df = pd.DataFrame({
                    "維度": [c["dim_zh"] for c in report["contributions"]],
                    "貢獻": [c["contribution"] for c in report["contributions"]],
                })
                contrib_chart = (
                    alt.Chart(contrib_df)
                    .mark_bar(cornerRadiusEnd=4, color="#6366f1")
                    .encode(
                        x=alt.X("貢獻:Q", title="對綜合評分的貢獻"),
                        y=alt.Y("維度:N", sort="-x", title=None),
                    )
                    .properties(height=160)
                )
                st.altair_chart(contrib_chart, use_container_width=True)

                st.markdown("#### 🌳 決策路徑（決策樹）")
                if report["decision_path"]:
                    st.markdown(f"> {report['decision_path']}")
                else:
                    st.caption("決策樹模型尚未載入或此方案無對應節點，無法顯示決策路徑。")

                if report["shap"]:
                    with st.expander("🧬 SHAP 特徵推力（決策樹）"):
                        shap_df = pd.DataFrame({
                            "特徵": [v["feature_zh"] for v in report["shap"]["values"]],
                            "推力": [v["value"] for v in report["shap"]["values"]],
                        })
                        shap_chart = (
                            alt.Chart(shap_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("推力:Q", title=f"對「{report['shap']['pred_zh']}」的推力"),
                                y=alt.Y("特徵:N", sort="-x", title=None),
                                color=alt.condition(
                                    alt.datum.推力 > 0,
                                    alt.value("#10b981"),
                                    alt.value("#ef4444"),
                                ),
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(shap_chart, use_container_width=True)
                else:
                    st.caption("（未安裝 shap 套件，略過 SHAP 解釋。）")

                st.markdown("#### 🔁 反事實提示")
                if report["counterfactuals"]:
                    for tip in report["counterfactuals"]:
                        st.warning(tip)
                else:
                    st.caption("在目前條件下，未找到會改變推薦的反事實情境。")

            st.markdown("---")

            # ── 特徵分數視覺化 ──
            st.markdown("### 📈 特徵分數比較")

            chart_df = pd.DataFrame({
                "交通方式": [_mode_display(r["mode"]) for r in results],
                "成本分數": [r["cost_score"] for r in results],
                "時間分數": [r["time_score"] for r in results],
                "便利性分數": [r["convenience_score"] for r in results],
            })

            chart_melted = chart_df.melt(
                id_vars="交通方式", var_name="特徵", value_name="分數"
            )

            chart = (
                alt.Chart(chart_melted)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("交通方式:N", axis=alt.Axis(labelAngle=0), title=None),
                    y=alt.Y("分數:Q", title="正規化分數"),
                    color=alt.Color("特徵:N", title="特徵"),
                    xOffset="特徵:N",
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

            st.markdown("---")

            # ── 兩端 YouBike 即時資訊（D5） ──
            try:
                yb_ctx = youbike_context(origin, destination, k=3)
                if yb_ctx["origin"] or yb_ctx["destination"]:
                    st.markdown("### 🚲 起訖兩端 YouBike 即時狀態")
                    col_yo, col_yd = st.columns(2)
                    with col_yo:
                        st.markdown(f"**起點：{origin} 附近**")
                        if yb_ctx["origin"]:
                            for s in yb_ctx["origin"]:
                                st.markdown(
                                    f"- `{s['distance_m']}m`  {s['name']}  ·  🚲 可借 **{s['available_rent_bikes']}** ／ 空位 {s['available_return_bikes']}"
                                )
                        else:
                            st.caption("附近 800m 內無可借車站點")
                    with col_yd:
                        st.markdown(f"**終點：{destination} 附近**")
                        if yb_ctx["destination"]:
                            for s in yb_ctx["destination"]:
                                st.markdown(
                                    f"- `{s['distance_m']}m`  {s['name']}  ·  🅿️ 空位 **{s['available_return_bikes']}** ／ 可借 {s['available_rent_bikes']}"
                                )
                        else:
                            st.caption("附近無 YouBike 站點")
                    st.markdown("---")
            except Exception as e:
                st.caption(f"YouBike 即時資料暫時無法取得：{e}")

            # ── 系統資訊 ──
            with st.expander("ℹ️ 系統資訊與聲明"):
                st.markdown(
                    """
                    **TRANSPORT DSS — 可解釋交通決策支援系統**

                    - 📊 資料來源：靜態票價與時間參考表、即時資料(Googles Maps、TDX、Youbike)
                    - 🧮 模型方法：加權評分模型（Weighted Scoring Model），並提供決策樹（Decision Tree）預測作為對照
                    - 🗺️ 支援範圍：北北基桃 10 處景點

                    ---

                    ⚠️ **決策支援聲明**

                    本系統僅作為交通方式選擇的分析與建議工具。
                    推薦結果會受到資料來源、模型假設與使用者輸入條件影響，不構成最終決策或結果保證。
                    本系統顯示之票價、交通狀況、班次資料不保證其準確性。
                    使用者仍應依實際交通狀況、票價、班次與個人需求自行判斷。
                    """
                )

else:
    # 未開始分析時的引導畫面
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px;">
            <h1 style="color: #475569; font-weight: 600; margin-bottom: 8px;">
                歡迎使用交通決策支援系統
            </h2>
            <p style="color: #94a3b8; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                請在左側面板設定您的旅行條件與偏好權重，<br>
                然後點擊「開始分析」查看推薦結果。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
