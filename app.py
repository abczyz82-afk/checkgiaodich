"""
Ứng dụng Phân tích Dòng tiền Thị trường Chứng khoán Việt Nam
=============================================================
Thu thập dữ liệu khớp lệnh intraday qua vnstock (source=VCI),
phân nhóm lệnh theo giá trị vào 3 nhóm dòng tiền:
  - Tạo lập (Market Maker): > 1 tỷ VND
  - Quỹ (Fund):             200 triệu – 1 tỷ VND
  - Nhỏ lẻ (Retail):        < 200 triệu VND
Tính khối lượng mua/bán, ròng, tỷ trọng, và chỉ số chi phối
cho toàn thị trường và từng mã cổ phiếu.

Nguồn dữ liệu : VCI (qua vnstock 4.x)
Hàm API        : Quote(symbol, source='VCI').provider.intraday(page_size=...)
Đơn vị giá VCI : nghìn VND (nhân × 1 000 để ra VND)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Suppress vnstock upgrade banners & noisy logs ────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("vnstock").setLevel(logging.CRITICAL)
logging.getLogger("vnai").setLevel(logging.CRITICAL)

from vnstock import Quote, Listing

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CẤU HÌNH MẶC ĐỊNH (biến cấu hình – dễ chỉnh trên sidebar)        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
PRICE_UNIT = 1_000                   # VCI trả giá theo nghìn VND
DEFAULT_MM_THRESHOLD  = 1_000_000_000   # Tạo lập  > 1 tỷ
DEFAULT_FUND_THRESHOLD = 200_000_000    # Quỹ  200 triệu – 1 tỷ
                                        # Nhỏ lẻ   < 200 triệu
PAGE_SIZE = 10_000                   # Số lệnh tối đa mỗi mã
MAX_WORKERS = 6                      # Luồng song song tối đa
DOMINANCE_W_NET  = 0.6               # Trọng số net volume
DOMINANCE_W_SHARE = 0.4              # Trọng số volume share

GROUP_COLORS = {
    "Tạo lập": "#FFD700",   # Gold
    "Quỹ":     "#4FC3F7",   # Light Blue
    "Nhỏ lẻ":  "#66BB6A",   # Green
}

GROUP_ORDER = ["Tạo lập", "Quỹ", "Nhỏ lẻ"]

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  HÀM THU THẬP DỮ LIỆU                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@st.cache_data(ttl=600, show_spinner=False)
def get_all_symbols() -> list[str]:
    """Lấy toàn bộ mã cổ phiếu niêm yết."""
    try:
        ls = Listing()
        df = ls.all_symbols()
        return sorted(df["symbol"].tolist())
    except Exception as e:
        st.error(f"Không thể lấy danh sách mã: {e}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_vn30_symbols() -> list[str]:
    """Lấy danh sách mã VN30."""
    try:
        ls = Listing()
        s = ls.symbols_by_group("VN30")
        return sorted(s.tolist())
    except Exception:
        return [
            "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB",
            "HPG", "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB",
            "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB",
            "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
        ]


def fetch_intraday_single(symbol: str) -> pd.DataFrame | None:
    """
    Lấy dữ liệu khớp lệnh trong ngày cho 1 mã.
    Gọi trực tiếp provider.intraday() để tránh bug wrapper truyền `page`.
    Trả None nếu lỗi hoặc không có dữ liệu.
    """
    try:
        q = Quote(symbol=symbol, source="VCI")
        df = q.provider.intraday(page_size=PAGE_SIZE)
        if df is not None and not df.empty:
            df["symbol"] = symbol
            return df
    except ValueError:
        # Ngoài giờ / chuẩn bị phiên → bỏ qua
        pass
    except Exception:
        pass
    return None


def fetch_all_intraday(
    symbols: list[str],
    progress_bar=None,
    status_text=None,
) -> pd.DataFrame:
    """
    Lấy dữ liệu intraday cho toàn bộ danh sách mã (song song).
    Trả về DataFrame gộp có cột `symbol`.
    """
    results: list[pd.DataFrame] = []
    total = len(symbols)
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_intraday_single, sym): sym
            for sym in symbols
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            done += 1
            try:
                df = future.result()
                if df is not None:
                    results.append(df)
                else:
                    failed += 1
            except Exception:
                failed += 1

            if progress_bar is not None:
                progress_bar.progress(
                    done / total,
                    text=f"Đang tải: {done}/{total} mã  |  Thành công: {done - failed}  |  Lỗi: {failed}  |  Đang xử lý: {sym}",
                )

    if status_text is not None:
        status_text.info(
            f"✅ Hoàn tất: {done - failed}/{total} mã thành công, "
            f"{failed} mã lỗi/không có dữ liệu."
        )

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  HÀM PHÂN NHÓM & TÍNH TOÁN                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def classify_orders(
    df: pd.DataFrame,
    mm_threshold: float,
    fund_threshold: float,
) -> pd.DataFrame:
    """
    Thêm cột phân nhóm vào DataFrame intraday.
    - order_value_vnd = price × PRICE_UNIT × volume
    - investor_group  = Tạo lập / Quỹ / Nhỏ lẻ
    - side            = Buy / Sell (ATO/ATC gán riêng)
    """
    df = df.copy()
    df["order_value_vnd"] = df["price"] * PRICE_UNIT * df["volume"]

    conditions = [
        df["order_value_vnd"] > mm_threshold,
        df["order_value_vnd"] >= fund_threshold,
    ]
    choices = ["Tạo lập", "Quỹ"]
    df["investor_group"] = np.select(conditions, choices, default="Nhỏ lẻ")

    # Phân loại phía mua/bán
    df["side"] = df["match_type"].map({
        "Buy":  "Buy",
        "Sell": "Sell",
        "ATO":  "Neutral",
        "ATC":  "Neutral",
    }).fillna("Neutral")

    return df


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thống kê tổng hợp theo nhóm dòng tiền.
    Trả về DataFrame với mỗi dòng là 1 nhóm.
    """
    if df.empty:
        return pd.DataFrame()

    # Chỉ tính Buy/Sell, bỏ Neutral
    df_bs = df[df["side"].isin(["Buy", "Sell"])]

    stats = []
    total_vol = df_bs["volume"].sum()
    total_val = df_bs["order_value_vnd"].sum()

    for group in GROUP_ORDER:
        gdf = df_bs[df_bs["investor_group"] == group]
        buy_vol  = gdf.loc[gdf["side"] == "Buy",  "volume"].sum()
        sell_vol = gdf.loc[gdf["side"] == "Sell", "volume"].sum()
        buy_val  = gdf.loc[gdf["side"] == "Buy",  "order_value_vnd"].sum()
        sell_val = gdf.loc[gdf["side"] == "Sell", "order_value_vnd"].sum()
        net_vol  = buy_vol - sell_vol
        net_val  = buy_val - sell_val
        grp_vol  = buy_vol + sell_vol
        grp_val  = buy_val + sell_val

        vol_pct = (grp_vol / total_vol * 100) if total_vol > 0 else 0
        val_pct = (grp_val / total_val * 100) if total_val > 0 else 0

        stats.append({
            "Nhóm":            group,
            "KL Mua":          buy_vol,
            "KL Bán":          sell_vol,
            "KL Ròng":         net_vol,
            "GT Mua (tỷ)":     buy_val  / 1e9,
            "GT Bán (tỷ)":     sell_val / 1e9,
            "GT Ròng (tỷ)":    net_val  / 1e9,
            "Tỷ trọng KL %":   round(vol_pct, 2),
            "Tỷ trọng GT %":   round(val_pct, 2),
        })

    return pd.DataFrame(stats)


def compute_dominance_score(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính chỉ số chi phối (0–100) cho mỗi nhóm.
    Score = w1 × (|net_vol_ratio|) + w2 × (vol_share)
    Chia theo 3 mức: Mạnh (≥70), Trung bình (40–69), Yếu (<40).
    """
    if stats_df.empty:
        return stats_df

    df = stats_df.copy()

    total_abs_net = df["KL Ròng"].abs().sum()
    total_vol     = (df["KL Mua"] + df["KL Bán"]).sum()

    scores = []
    for _, row in df.iterrows():
        grp_vol = row["KL Mua"] + row["KL Bán"]
        net_ratio  = abs(row["KL Ròng"]) / total_abs_net if total_abs_net > 0 else 0
        vol_share  = grp_vol / total_vol if total_vol > 0 else 0

        raw = DOMINANCE_W_NET * net_ratio + DOMINANCE_W_SHARE * vol_share
        score = round(raw * 100, 1)
        scores.append(score)

    df["Điểm chi phối"] = scores
    df["Mức độ"] = df["Điểm chi phối"].apply(
        lambda x: "🔴 Mạnh" if x >= 70 else ("🟡 Trung bình" if x >= 40 else "🟢 Yếu")
    )

    return df


def compute_per_symbol_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính thống kê cho từng mã cổ phiếu: tổng KL mua/bán/ròng mỗi nhóm.
    Trả về DataFrame wide (mỗi dòng = 1 mã).
    """
    if df.empty:
        return pd.DataFrame()

    df_bs = df[df["side"].isin(["Buy", "Sell"])]

    records = []
    for symbol, sdf in df_bs.groupby("symbol"):
        total_vol = sdf["volume"].sum()
        total_val = sdf["order_value_vnd"].sum()
        row = {"Mã": symbol, "Tổng KL": total_vol, "Tổng GT (tỷ)": round(total_val / 1e9, 2)}

        for group in GROUP_ORDER:
            gdf = sdf[sdf["investor_group"] == group]
            buy_vol  = gdf.loc[gdf["side"] == "Buy",  "volume"].sum()
            sell_vol = gdf.loc[gdf["side"] == "Sell", "volume"].sum()
            buy_val  = gdf.loc[gdf["side"] == "Buy",  "order_value_vnd"].sum()
            sell_val = gdf.loc[gdf["side"] == "Sell", "order_value_vnd"].sum()
            net_vol  = buy_vol - sell_vol
            net_val  = buy_val - sell_val
            grp_vol  = buy_vol + sell_vol
            vol_pct  = round(grp_vol / total_vol * 100, 2) if total_vol > 0 else 0

            prefix = group[:2]  # TL / Qu / Nh
            row[f"{prefix}_Mua_KL"]     = buy_vol
            row[f"{prefix}_Bán_KL"]     = sell_vol
            row[f"{prefix}_Ròng_KL"]    = net_vol
            row[f"{prefix}_Ròng_GT(tỷ)"] = round(net_val / 1e9, 4)
            row[f"{prefix}_TT%"]        = vol_pct

        # Dominance per symbol
        abs_nets = []
        for group in GROUP_ORDER:
            prefix = group[:2]
            abs_nets.append(abs(row[f"{prefix}_Ròng_KL"]))
        total_abs_net = sum(abs_nets)

        for i, group in enumerate(GROUP_ORDER):
            prefix = group[:2]
            grp_vol_sym = row[f"{prefix}_Mua_KL"] + row[f"{prefix}_Bán_KL"]
            net_r = abs_nets[i] / total_abs_net if total_abs_net > 0 else 0
            vol_r = grp_vol_sym / total_vol if total_vol > 0 else 0
            score = round((DOMINANCE_W_NET * net_r + DOMINANCE_W_SHARE * vol_r) * 100, 1)
            row[f"{prefix}_Score"] = score

        # Nhóm chi phối nhất
        best_group = max(
            GROUP_ORDER,
            key=lambda g: row[f"{g[:2]}_Score"],
        )
        row["Nhóm chi phối"] = best_group

        records.append(row)

    return pd.DataFrame(records)


def compute_time_series(df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """
    Resample dữ liệu intraday theo khoảng thời gian,
    tính net volume mỗi nhóm theo từng interval.
    """
    if df.empty:
        return pd.DataFrame()

    df_bs = df[df["side"].isin(["Buy", "Sell"])].copy()
    if df_bs.empty:
        return pd.DataFrame()

    df_bs["signed_vol"] = df_bs.apply(
        lambda r: r["volume"] if r["side"] == "Buy" else -r["volume"], axis=1
    )

    # Ensure time is datetime & set as index
    df_bs["time"] = pd.to_datetime(df_bs["time"], utc=True)
    df_bs = df_bs.set_index("time")

    freq = f"{interval_minutes}min"
    records = []
    for group in GROUP_ORDER:
        gdf = df_bs[df_bs["investor_group"] == group]
        if gdf.empty:
            continue
        resampled = gdf["signed_vol"].resample(freq).sum().fillna(0)
        for t, val in resampled.items():
            records.append({
                "Thời gian": t,
                "Nhóm": group,
                "KL Ròng": val,
            })

    return pd.DataFrame(records)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FORMAT HELPERS                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fmt_volume(v):
    """Format volume với dấu phân cách nghìn."""
    if pd.isna(v) or v == 0:
        return "0"
    return f"{int(v):,}"

def fmt_billion(v):
    """Format giá trị tỷ VND."""
    if pd.isna(v):
        return "0"
    return f"{v:,.2f}"

def color_net(val):
    """CSS cho giá trị ròng: xanh dương (+) / đỏ (-)."""
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: #4FC3F7; font-weight: bold"
        elif val < 0:
            return "color: #FF5252; font-weight: bold"
    return ""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GIAO DIỆN STREAMLIT                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    st.set_page_config(
        page_title="Phân tích Dòng tiền TTCK Việt Nam",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 50%, #2C3E50 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(79, 195, 247, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        background: linear-gradient(90deg, #FFD700, #4FC3F7, #66BB6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #B0BEC5;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(79, 195, 247, 0.15);
    }
    .metric-label {
        color: #78909C;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-value.gold   { color: #FFD700; }
    .metric-value.blue   { color: #4FC3F7; }
    .metric-value.green  { color: #66BB6A; }
    .metric-value.red    { color: #FF5252; }
    .metric-value.white  { color: #ECEFF1; }

    .group-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .group-badge.taolap  { background: rgba(255,215,0,0.15); color: #FFD700; }
    .group-badge.quy     { background: rgba(79,195,247,0.15); color: #4FC3F7; }
    .group-badge.nhole   { background: rgba(102,187,106,0.15); color: #66BB6A; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ECEFF1;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(79, 195, 247, 0.3);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .info-box {
        background: rgba(79, 195, 247, 0.08);
        border: 1px solid rgba(79, 195, 247, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #B0BEC5;
    }

    .legend-row {
        display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem;
    }
    .legend-item {
        display: flex; align-items: center; gap: 0.4rem;
        font-size: 0.85rem; color: #B0BEC5;
    }
    .legend-dot {
        width: 12px; height: 12px; border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📊 Phân tích Dòng tiền TTCK Việt Nam</h1>
        <p>Thu thập & phân loại lệnh khớp intraday theo 3 nhóm dòng tiền — Tạo lập · Quỹ · Nhỏ lẻ</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cấu hình")

        st.markdown("#### 📐 Ngưỡng phân nhóm (VND)")
        mm_threshold = st.number_input(
            "Tạo lập (>)",
            value=DEFAULT_MM_THRESHOLD,
            step=100_000_000,
            format="%d",
            help="Lệnh có giá trị > ngưỡng này → Tạo lập",
        )
        fund_threshold = st.number_input(
            "Quỹ (≥)",
            value=DEFAULT_FUND_THRESHOLD,
            step=50_000_000,
            format="%d",
            help="Lệnh có giá trị ≥ ngưỡng này và ≤ Tạo lập → Quỹ. Dưới ngưỡng → Nhỏ lẻ",
        )

        st.markdown("---")
        st.markdown("#### 📋 Phạm vi thu thập")

        scope = st.radio(
            "Chọn nhóm cổ phiếu",
            ["Toàn thị trường (~1500 mã)", "VN30 (30 mã)", "Tùy chọn"],
            index=0,
        )

        custom_symbols = []
        if scope == "Tùy chọn":
            custom_input = st.text_area(
                "Nhập mã (phân cách bởi dấu phẩy)",
                value="VNM, FPT, HPG, VCB, MBB",
                help="Ví dụ: VNM, FPT, HPG",
            )
            custom_symbols = [
                s.strip().upper()
                for s in custom_input.split(",")
                if s.strip()
            ]

        st.markdown("---")
        st.markdown("#### ⏱️ Tùy chọn khác")
        time_interval = st.selectbox(
            "Khoảng thời gian diễn biến",
            [5, 10, 15, 30, 60],
            index=0,
            format_func=lambda x: f"{x} phút",
        )

        st.markdown("---")
        st.markdown(
            '<div class="info-box">'
            "💡 <b>Lưu ý</b>: Quét toàn thị trường (~1500 mã) có thể "
            "mất 5–15 phút tùy tốc độ mạng và rate limit API VCI. "
            "Dữ liệu được cache 10 phút."
            "</div>",
            unsafe_allow_html=True,
        )

        fetch_btn = st.button("🔄 Thu thập dữ liệu", type="primary", use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="legend-row">
        <div class="legend-item">
            <div class="legend-dot" style="background: {GROUP_COLORS['Tạo lập']}"></div>
            Tạo lập (> {mm_threshold/1e9:.1f} tỷ)
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {GROUP_COLORS['Quỹ']}"></div>
            Quỹ ({fund_threshold/1e6:.0f}tr – {mm_threshold/1e9:.1f} tỷ)
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {GROUP_COLORS['Nhỏ lẻ']}"></div>
            Nhỏ lẻ (< {fund_threshold/1e6:.0f}tr)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ────────────────────────────────────────────────
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "classified_data" not in st.session_state:
        st.session_state.classified_data = None
    if "fetch_time" not in st.session_state:
        st.session_state.fetch_time = None

    # ── Fetch logic ──────────────────────────────────────────────────
    if fetch_btn:
        # Determine symbol list
        if scope == "Toàn thị trường (~1500 mã)":
            symbols = get_all_symbols()
        elif scope == "VN30 (30 mã)":
            symbols = get_vn30_symbols()
        else:
            symbols = custom_symbols

        if not symbols:
            st.error("Không tìm thấy mã cổ phiếu nào.")
            return

        st.info(f"🚀 Bắt đầu thu thập dữ liệu cho **{len(symbols)} mã**...")
        progress = st.progress(0, text="Đang khởi tạo...")
        status_box = st.empty()

        t0 = time.time()
        raw = fetch_all_intraday(symbols, progress_bar=progress, status_text=status_box)
        elapsed = time.time() - t0

        if raw.empty:
            st.error(
                "Không có dữ liệu! Có thể ngoài giờ giao dịch (9:00–15:00) "
                "hoặc đang chuẩn bị phiên mới."
            )
            return

        classified = classify_orders(raw, mm_threshold, fund_threshold)
        st.session_state.raw_data = raw
        st.session_state.classified_data = classified
        st.session_state.fetch_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        st.session_state.elapsed = round(elapsed, 1)

        st.success(
            f"✅ Thu thập xong trong **{elapsed:.1f}s**  •  "
            f"**{classified['symbol'].nunique()}** mã  •  "
            f"**{len(classified):,}** lệnh khớp"
        )

    # ── Display ──────────────────────────────────────────────────────
    if st.session_state.classified_data is None:
        st.markdown(
            '<div class="info-box">'
            "👈 Nhấn <b>Thu thập dữ liệu</b> ở sidebar để bắt đầu."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    df_all = st.session_state.classified_data
    st.caption(
        f"🕐 Dữ liệu lúc: {st.session_state.fetch_time}  •  "
        f"Thời gian tải: {st.session_state.elapsed}s  •  "
        f"{df_all['symbol'].nunique()} mã  •  "
        f"{len(df_all):,} lệnh"
    )

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_market, tab_stock = st.tabs([
        "🌐 Toàn thị trường",
        "🔍 Theo mã cổ phiếu",
    ])

    # ═════════════════════════════════════════════════════════════════
    # TAB 1: TOÀN THỊ TRƯỜNG
    # ═════════════════════════════════════════════════════════════════
    with tab_market:
        market_stats = compute_group_stats(df_all)
        market_stats = compute_dominance_score(market_stats)

        if market_stats.empty:
            st.warning("Không có dữ liệu mua/bán để phân tích.")
            return

        # ── KPI Cards ────────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Tổng quan thị trường</div>', unsafe_allow_html=True)

        total_val = market_stats["GT Mua (tỷ)"].sum() + market_stats["GT Bán (tỷ)"].sum()
        total_net_val = market_stats["GT Ròng (tỷ)"].sum()
        dominant_group = market_stats.loc[market_stats["Điểm chi phối"].idxmax(), "Nhóm"]
        dominant_score = market_stats["Điểm chi phối"].max()

        cols = st.columns(4)
        kpis = [
            ("Tổng GTGD", f"{total_val:,.1f} tỷ", "white"),
            ("GT Ròng toàn TT", f"{total_net_val:+,.1f} tỷ", "blue" if total_net_val >= 0 else "red"),
            ("Nhóm chi phối", dominant_group, "gold" if dominant_group == "Tạo lập" else ("blue" if dominant_group == "Quỹ" else "green")),
            ("Điểm chi phối", f"{dominant_score}", "gold"),
        ]
        for col, (label, value, color) in zip(cols, kpis):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Bảng tổng hợp ───────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Phân bổ theo nhóm dòng tiền</div>', unsafe_allow_html=True)

        display_cols = [
            "Nhóm", "KL Mua", "KL Bán", "KL Ròng",
            "GT Mua (tỷ)", "GT Bán (tỷ)", "GT Ròng (tỷ)",
            "Tỷ trọng KL %", "Tỷ trọng GT %",
            "Điểm chi phối", "Mức độ",
        ]
        st.dataframe(
            market_stats[display_cols].style
                .format({
                    "KL Mua":       "{:,.0f}",
                    "KL Bán":       "{:,.0f}",
                    "KL Ròng":      "{:+,.0f}",
                    "GT Mua (tỷ)":  "{:,.2f}",
                    "GT Bán (tỷ)":  "{:,.2f}",
                    "GT Ròng (tỷ)": "{:+,.2f}",
                    "Tỷ trọng KL %": "{:.2f}%",
                    "Tỷ trọng GT %": "{:.2f}%",
                    "Điểm chi phối": "{:.1f}",
                })
                .map(color_net, subset=["KL Ròng", "GT Ròng (tỷ)"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Diễn biến theo thời gian ────────────────────────────────
        st.markdown('<div class="section-title">⏱️ Diễn biến dòng tiền ròng theo thời gian</div>', unsafe_allow_html=True)

        ts = compute_time_series(df_all, interval_minutes=time_interval)
        if not ts.empty:
            ts_pivot = ts.pivot_table(
                index="Thời gian", columns="Nhóm", values="KL Ròng", aggfunc="sum"
            ).fillna(0)
            # Reorder columns
            ts_pivot = ts_pivot[[c for c in GROUP_ORDER if c in ts_pivot.columns]]
            st.dataframe(
                ts_pivot.style.format("{:+,.0f}").map(color_net),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("Không có dữ liệu diễn biến theo thời gian.")

        # ── Bảng chi tiết từng mã ────────────────────────────────────
        st.markdown('<div class="section-title">📋 Chi tiết từng mã cổ phiếu</div>', unsafe_allow_html=True)

        per_sym = compute_per_symbol_stats(df_all)
        if not per_sym.empty:
            # Sort by total value descending
            per_sym = per_sym.sort_values("Tổng GT (tỷ)", ascending=False).reset_index(drop=True)

            # Column renaming for display
            rename_map = {
                "Tạ_Mua_KL": "TL Mua KL", "Tạ_Bán_KL": "TL Bán KL",
                "Tạ_Ròng_KL": "TL Ròng KL", "Tạ_Ròng_GT(tỷ)": "TL Ròng GT(tỷ)",
                "Tạ_TT%": "TL TT%", "Tạ_Score": "TL Score",
                "Qu_Mua_KL": "Quỹ Mua KL", "Qu_Bán_KL": "Quỹ Bán KL",
                "Qu_Ròng_KL": "Quỹ Ròng KL", "Qu_Ròng_GT(tỷ)": "Quỹ Ròng GT(tỷ)",
                "Qu_TT%": "Quỹ TT%", "Qu_Score": "Quỹ Score",
                "Nh_Mua_KL": "NL Mua KL", "Nh_Bán_KL": "NL Bán KL",
                "Nh_Ròng_KL": "NL Ròng KL", "Nh_Ròng_GT(tỷ)": "NL Ròng GT(tỷ)",
                "Nh_TT%": "NL TT%", "Nh_Score": "NL Score",
            }
            per_sym_display = per_sym.rename(columns=rename_map)

            # Format dict
            fmt_dict = {"Tổng GT (tỷ)": "{:,.2f}", "Tổng KL": "{:,.0f}"}
            for prefix in ["TL", "Quỹ", "NL"]:
                fmt_dict[f"{prefix} Mua KL"]     = "{:,.0f}"
                fmt_dict[f"{prefix} Bán KL"]     = "{:,.0f}"
                fmt_dict[f"{prefix} Ròng KL"]    = "{:+,.0f}"
                fmt_dict[f"{prefix} Ròng GT(tỷ)"] = "{:+,.4f}"
                fmt_dict[f"{prefix} TT%"]        = "{:.2f}%"
                fmt_dict[f"{prefix} Score"]      = "{:.1f}"

            net_cols = [c for c in per_sym_display.columns if "Ròng" in c]

            st.dataframe(
                per_sym_display.style
                    .format(fmt_dict)
                    .map(color_net, subset=net_cols),
                use_container_width=True,
                height=600,
            )

            # ── Download CSV ─────────────────────────────────────────
            csv = per_sym_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải CSV toàn thị trường",
                csv,
                file_name=f"dong_tien_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    # ═════════════════════════════════════════════════════════════════
    # TAB 2: THEO MÃ CỔ PHIẾU
    # ═════════════════════════════════════════════════════════════════
    with tab_stock:
        available_symbols = sorted(df_all["symbol"].unique())

        if not available_symbols:
            st.warning("Chưa có dữ liệu. Vui lòng thu thập trước.")
            return

        selected = st.selectbox(
            "🔎 Chọn mã cổ phiếu",
            available_symbols,
            index=available_symbols.index("VNM") if "VNM" in available_symbols else 0,
        )

        sym_df = df_all[df_all["symbol"] == selected]

        if sym_df.empty:
            st.warning(f"Không có dữ liệu cho {selected}.")
            return

        # ── KPI for selected stock ───────────────────────────────────
        sym_stats = compute_group_stats(sym_df)
        sym_stats = compute_dominance_score(sym_stats)

        st.markdown(f'<div class="section-title">📈 Tổng quan {selected}</div>', unsafe_allow_html=True)

        sym_total_val = sym_stats["GT Mua (tỷ)"].sum() + sym_stats["GT Bán (tỷ)"].sum()
        sym_net_val = sym_stats["GT Ròng (tỷ)"].sum()
        sym_total_vol = sym_stats["KL Mua"].sum() + sym_stats["KL Bán"].sum()
        sym_dominant = sym_stats.loc[sym_stats["Điểm chi phối"].idxmax(), "Nhóm"] if not sym_stats.empty else "N/A"

        cols = st.columns(4)
        sym_kpis = [
            ("Tổng KL", f"{sym_total_vol:,.0f}", "white"),
            ("Tổng GT", f"{sym_total_val:,.2f} tỷ", "white"),
            ("GT Ròng", f"{sym_net_val:+,.2f} tỷ", "blue" if sym_net_val >= 0 else "red"),
            ("Nhóm chi phối", sym_dominant, "gold" if sym_dominant == "Tạo lập" else ("blue" if sym_dominant == "Quỹ" else "green")),
        ]
        for col, (label, value, color) in zip(cols, sym_kpis):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Bảng chi tiết ────────────────────────────────────────────
        st.markdown(f'<div class="section-title">📊 Phân bổ dòng tiền — {selected}</div>', unsafe_allow_html=True)

        st.dataframe(
            sym_stats[display_cols].style
                .format({
                    "KL Mua":       "{:,.0f}",
                    "KL Bán":       "{:,.0f}",
                    "KL Ròng":      "{:+,.0f}",
                    "GT Mua (tỷ)":  "{:,.2f}",
                    "GT Bán (tỷ)":  "{:,.2f}",
                    "GT Ròng (tỷ)": "{:+,.2f}",
                    "Tỷ trọng KL %": "{:.2f}%",
                    "Tỷ trọng GT %": "{:.2f}%",
                    "Điểm chi phối": "{:.1f}",
                })
                .map(color_net, subset=["KL Ròng", "GT Ròng (tỷ)"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── So sánh với thị trường ───────────────────────────────────
        st.markdown(f'<div class="section-title">⚖️ So sánh với trung bình thị trường</div>', unsafe_allow_html=True)

        market_stats_cmp = compute_group_stats(df_all)
        market_stats_cmp = compute_dominance_score(market_stats_cmp)

        comparison = []
        for _, mrow in market_stats_cmp.iterrows():
            group = mrow["Nhóm"]
            srow = sym_stats[sym_stats["Nhóm"] == group]
            if srow.empty:
                continue
            srow = srow.iloc[0]
            comparison.append({
                "Nhóm":                       group,
                f"{selected} - Tỷ trọng KL%": srow["Tỷ trọng KL %"],
                "Thị trường - Tỷ trọng KL%":  mrow["Tỷ trọng KL %"],
                f"{selected} - GT Ròng (tỷ)": srow["GT Ròng (tỷ)"],
                f"{selected} - Điểm CP":       srow["Điểm chi phối"],
                "Thị trường - Điểm CP":        mrow["Điểm chi phối"],
            })

        if comparison:
            cmp_df = pd.DataFrame(comparison)
            st.dataframe(
                cmp_df.style.format({
                    col: "{:.2f}%" for col in cmp_df.columns if "KL%" in col
                } | {
                    col: "{:+,.2f}" for col in cmp_df.columns if "Ròng" in col
                } | {
                    col: "{:.1f}" for col in cmp_df.columns if "Điểm" in col
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Diễn biến theo thời gian cho mã ─────────────────────────
        st.markdown(f'<div class="section-title">⏱️ Diễn biến dòng tiền ròng — {selected}</div>', unsafe_allow_html=True)

        ts_sym = compute_time_series(sym_df, interval_minutes=time_interval)
        if not ts_sym.empty:
            ts_sym_pivot = ts_sym.pivot_table(
                index="Thời gian", columns="Nhóm", values="KL Ròng", aggfunc="sum"
            ).fillna(0)
            ts_sym_pivot = ts_sym_pivot[[c for c in GROUP_ORDER if c in ts_sym_pivot.columns]]
            st.dataframe(
                ts_sym_pivot.style.format("{:+,.0f}").map(color_net),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("Không có dữ liệu diễn biến.")

        # ── Top lệnh lớn nhất ────────────────────────────────────────
        st.markdown(f'<div class="section-title">🏆 Top 20 lệnh lớn nhất — {selected}</div>', unsafe_allow_html=True)

        top_orders = (
            sym_df[sym_df["side"].isin(["Buy", "Sell"])]
            .nlargest(20, "order_value_vnd")
            [["time", "price", "volume", "match_type", "investor_group", "order_value_vnd"]]
            .copy()
        )
        if not top_orders.empty:
            top_orders["GT (triệu)"] = top_orders["order_value_vnd"] / 1e6
            top_orders["time"] = top_orders["time"].dt.strftime("%H:%M:%S")
            top_orders = top_orders.rename(columns={
                "time": "Thời gian",
                "price": "Giá (nghìn)",
                "volume": "Khối lượng",
                "match_type": "Loại",
                "investor_group": "Nhóm",
            })
            st.dataframe(
                top_orders[["Thời gian", "Giá (nghìn)", "Khối lượng", "Loại", "Nhóm", "GT (triệu)"]]
                .style.format({
                    "Giá (nghìn)": "{:,.1f}",
                    "Khối lượng": "{:,.0f}",
                    "GT (triệu)": "{:,.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
