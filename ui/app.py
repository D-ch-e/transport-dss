"""
M6 · Streamlit UI — 使用者介面模組（Week 1 MVP）

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

from data.api_client import get_supported_cities, get_route_data
from data.input_validator import validate_input
from features.feature_engineer import compute_features
from models.scoring_model import compute_scores

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
        color: #475569;
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
}

# ──────────────────────────────────────────────
# 側邊欄：使用者輸入
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚄 交通決策支援系統")
    st.markdown("---")

    cities = get_supported_cities()

    st.markdown("### 📍 路線設定")
    origin = st.selectbox("起點", cities, index=0, key="origin")
    # 預設終點為不同城市
    default_dest = 1 if len(cities) > 1 else 0
    destination = st.selectbox("終點", cities, index=default_dest, key="destination")

    st.markdown("---")
    st.markdown("### 👥 旅行條件")
    num_people = st.number_input("人數", min_value=1, max_value=10, value=2, step=1, key="people")
    luggage = st.number_input("行李件數", min_value=0, max_value=10, value=1, step=1, key="luggage")
    budget = st.number_input("預算上限（元/人）", min_value=100, max_value=10000, value=2000, step=100, key="budget")

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
    run_button = st.button("🔍 開始分析", use_container_width=True, type="primary")

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
        # 取得路線資料
        route_data = get_route_data(origin, destination)

        if not route_data:
            st.warning(f"找不到 {origin} → {destination} 的交通資料。請嘗試其他路線。")
        else:
            # 計算特徵
            features = compute_features(route_data, num_people, luggage)

            # 計算分數
            results = compute_scores(features, w_cost, w_time, w_conv)

            # 取最佳方案
            best = results[0]
            best_emoji = MODE_EMOJI.get(best["mode"], "🚀")

            # ── 推薦卡片 ──
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <h2>🏆 最佳推薦交通方式</h2>
                    <div class="mode-name">{best_emoji} {best["mode"]}</div>
                    <div class="score-text">
                        綜合評分 {best["total_score"]:.2f} ／ 路線：{origin} → {destination}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 最佳方案指標 ──
            cols = st.columns(4)
            with cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">人均成本</div>
                        <div class="metric-value">${best["cost_raw"]:.0f}</div>
                        <div class="metric-unit">元 / 人</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cols[1]:
                hours = int(best["time_raw"]) // 60
                mins = int(best["time_raw"]) % 60
                time_display = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">預估時間</div>
                        <div class="metric-value">{time_display}</div>
                        <div class="metric-unit">含候車與轉乘</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">便利性</div>
                        <div class="metric-value">{best["convenience_raw"]:.0f}</div>
                        <div class="metric-unit">/ 100 分</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cols[3]:
                budget_ok = best["cost_raw"] <= budget
                budget_icon = "✅" if budget_ok else "⚠️"
                budget_text = "在預算內" if budget_ok else "超出預算"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">預算檢查</div>
                        <div class="metric-value">{budget_icon}</div>
                        <div class="metric-unit">{budget_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── 多方案比較表 ──
            st.markdown("### 📊 多方案比較")

            comparison_data = []
            for r in results:
                emoji = MODE_EMOJI.get(r["mode"], "")
                in_budget = "✅" if r["cost_raw"] <= budget else "❌"

                comparison_data.append({
                    "排名": f"#{r['rank']}",
                    "交通方式": f"{emoji} {r['mode']}",
                    "人均成本（元）": f"${r['cost_raw']:.0f}",
                    "預估時間（分鐘）": f"{r['time_raw']:.0f}",
                    "便利性分數": f"{r['convenience_raw']:.0f}",
                    "成本分數": f"{r['cost_score']:.2f}",
                    "時間分數": f"{r['time_score']:.2f}",
                    "便利性正規化": f"{r['convenience_score']:.2f}",
                    "加權總分": f"{r['total_score']:.4f}",
                    "預算": in_budget,
                })

            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")

            # ── 特徵分數視覺化 ──
            st.markdown("### 📈 特徵分數比較")

            import altair as alt

            chart_df = pd.DataFrame({
                "交通方式": [r["mode"] for r in results],
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

            # ── 分析說明 ──
            st.markdown("### 💡 初步分析說明")

            # 找出影響最大的權重
            weights = best["weights"]
            max_weight_name = max(weights, key=weights.get)
            weight_names = {"cost": "成本", "time": "時間", "convenience": "便利性"}

            st.info(
                f"📌 在您目前的偏好設定下，**{weight_names[max_weight_name]}**"
                f"的權重最高（{weights[max_weight_name]*100:.0f}%），"
                f"因此 **{best['mode']}** 獲得了最高的綜合評分。"
            )

            # 預算提醒
            over_budget = [r for r in results if r["cost_raw"] > budget]
            if over_budget:
                modes_str = "、".join([r["mode"] for r in over_budget])
                st.warning(
                    f"💸 以目前的預算上限 ${budget}/人，"
                    f"以下交通方式超出預算：**{modes_str}**"
                )

            # 條件提示
            if num_people >= 3 and best["mode"] != "租車":
                car = next((r for r in results if r["mode"] == "租車"), None)
                if car:
                    st.info(
                        f"🚗 多人同行提示：{num_people} 人同行時，"
                        f"租車人均成本為 ${car['cost_raw']:.0f}/人，"
                        f"可考慮租車方案。"
                    )

            st.markdown("---")

            # ── 系統資訊 ──
            with st.expander("ℹ️ 系統資訊與聲明"):
                st.markdown(
                    """
                    **XAI DSS — 可解釋交通決策支援系統 v0.1（Week 1 MVP）**

                    - 📊 資料來源：靜態票價與時間參考表
                    - 🧮 模型方法：加權評分模型（Weighted Scoring Model）
                    - 🗺️ 支援路線：台北、台中、台南、高雄（雙向）

                    ---

                    ⚠️ **決策支援聲明**

                    本系統僅作為交通方式選擇的分析與建議工具。推薦結果會受到資料來源、
                    模型假設與使用者輸入條件影響，不構成最終決策或結果保證。使用者仍應依
                    實際交通狀況、票價、班次與個人需求自行判斷。
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
