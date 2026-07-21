import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import time

# ============================================================
# 페이지 기본 설정
# ============================================================
st.set_page_config(page_title="SK하이닉스 자동 FGI", page_icon="📈", layout="wide")

st.title("🤖 SK하이닉스 공포탐욕지수(FGI) 자동 계산 웹앱")
st.markdown(
    "야후 파이낸스(Yahoo Finance) API를 통해 최신 주가 데이터를 **자동으로 수집**하여 "
    "투자 심리를 분석합니다. (수동 입력 불필요)"
)


# ============================================================
# 사이드바: 텔레그램 알림 설정  [추가 기능 4]
# ============================================================
st.sidebar.header("🔔 텔레그램 알림 설정")
st.sidebar.caption("FGI가 '극도의 공포/탐욕' 구간에 진입하면 텔레그램으로 알림을 보냅니다.")
telegram_enabled = st.sidebar.checkbox("알림 활성화", value=False)
telegram_token = st.sidebar.text_input("Bot Token", type="password", help="@BotFather로 발급받은 봇 토큰")
telegram_chat_id = st.sidebar.text_input("Chat ID", help="알림을 받을 텔레그램 채팅방 ID")
st.sidebar.caption(
    "⚠️ 참고: Streamlit 세션이 재시작되면(앱 슬립/재배포 등) 중복 발송 방지 기록이 초기화될 수 있습니다. "
    "운영 환경에서는 외부 DB에 발송 이력을 저장하는 것을 권장합니다."
)


def send_telegram_alert(token: str, chat_id: str, message: str) -> bool:
    """텔레그램 봇 API로 알림 메시지를 전송한다. 실패 시 False를 반환한다."""
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# 데이터 로딩 (에러 핸들링 + 재시도 로직)  [개선 5]
# ============================================================
@st.cache_data(ttl=60)  # 60초 단위로 데이터 캐싱 (API 호출 최적화)
def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """야후 파이낸스 데이터를 재시도 로직과 함께 안전하게 가져온다."""
    last_error = None
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, progress=False)
            # yfinance 최신 버전의 MultiIndex 컬럼 평탄화 처리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            if data.empty:
                raise ValueError("빈 데이터가 반환되었습니다.")
            return data
        except Exception as e:  # noqa: BLE001 - 재시도를 위해 광범위하게 포착
            last_error = e
            time.sleep(1.5)
    raise RuntimeError(f"'{ticker}' 데이터 수집에 3회 실패했습니다. (마지막 오류: {last_error})")


try:
    with st.spinner("금융 시장에서 최신 주가 데이터를 자동으로 분석하는 중입니다..."):
        df = get_price_data("000660.KS", period="1y")
except Exception as e:
    st.error(
        "⚠️ SK하이닉스 주가 데이터를 불러오지 못했습니다.\n\n"
        "주말/공휴일 장마감, 야후 파이낸스 API 일시 장애, 혹은 네트워크 문제일 수 있습니다. "
        "잠시 후 새로고침해 주세요.\n\n"
        f"상세 오류: {e}"
    )
    st.stop()

# 보조 데이터(환율/삼성전자)는 실패해도 메인 FGI 기능이 죽지 않도록 개별적으로 예외 처리
usdkrw_df, samsung_df = None, None
try:
    usdkrw_df = get_price_data("KRW=X", period="6mo")
except Exception:
    st.sidebar.warning("⚠️ 원/달러 환율 데이터를 불러오지 못했습니다.")

try:
    samsung_df = get_price_data("005930.KS", period="6mo")
except Exception:
    st.sidebar.warning("⚠️ 삼성전자 데이터를 불러오지 못했습니다.")

# KRX 투자자별 수급 데이터 (외국인 순매수 / 공매도 비중)  [추가 기능 3]
investor_df, short_df = None, None
try:
    from pykrx import stock as krx_stock

    @st.cache_data(ttl=3600)  # KRX 데이터는 하루 한 번 정도만 갱신되므로 1시간 캐싱
    def get_krx_supply_demand(ticker: str = "000660", lookback_days: int = 90):
        end = datetime.today()
        start = end - pd.Timedelta(days=lookback_days)
        end_s, start_s = end.strftime("%Y%m%d"), start.strftime("%Y%m%d")
        inv = krx_stock.get_market_trading_value_by_date(start_s, end_s, ticker)
        short = krx_stock.get_shorting_volume_by_date(start_s, end_s, ticker)
        return inv, short

    investor_df, short_df = get_krx_supply_demand("000660", 90)
except Exception:
    investor_df, short_df = None, None


# ============================================================
# 지표 계산
# ============================================================
# RSI (14일) - 0으로 나누기 방지 처리  [개선 3]
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
rs = gain / loss.replace(0, np.nan)  # loss가 0이면 나눗셈 대신 NaN 처리
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].fillna(100)  # 연속 상승으로 loss가 0인 구간은 극단적 과매수(RSI=100)로 간주

# 이격도 (60일)
df["MA60"] = df["Close"].rolling(window=60).mean()
df["Disparity"] = (df["Close"] / df["MA60"] - 1) * 100
df["Disparity_Score"] = ((df["Disparity"] + 20) / 40 * 100).clip(0, 100)

# 볼린저 밴드 (20일)
df["MA20"] = df["Close"].rolling(window=20).mean()
df["STD20"] = df["Close"].rolling(window=20).std()
df["Upper"] = df["MA20"] + (df["STD20"] * 2)
df["Lower"] = df["MA20"] - (df["STD20"] * 2)
df["BB_Score"] = (((df["Close"] - df["Lower"]) / (df["Upper"] - df["Lower"])) * 100).clip(0, 100)

# 거래량 강도 - 방향성(상승일/하락일) 반영  [개선 4]
# 기존 로직(Vol5/Vol20 비율)은 "거래량 급증=탐욕"으로 단순화했지만,
# 패닉 매도(하락일 거래량 급증) 역시 거래량이 늘어나므로 방향을 구분해야 정확함.
price_change = df["Close"].diff()
df["UpVol"] = np.where(price_change > 0, df["Volume"], 0)
df["DownVol"] = np.where(price_change < 0, df["Volume"], 0)
up_vol_ma5 = pd.Series(df["UpVol"], index=df.index).rolling(window=5).mean()
down_vol_ma5 = pd.Series(df["DownVol"], index=df.index).rolling(window=5).mean()
vol_total = (up_vol_ma5 + down_vol_ma5).replace(0, np.nan)
# 상승일 거래량 비중이 높을수록 탐욕(100에 가까움), 하락일 거래량 비중이 높을수록 공포(0에 가까움)
df["Vol_Score"] = (up_vol_ma5 / vol_total * 100).fillna(50).clip(0, 100)

# 2. 최종 FGI 지수 산출 (가중치 적용)
df["FGI"] = (
    (df["RSI"] * 0.3)
    + (df["Disparity_Score"] * 0.3)
    + (df["BB_Score"] * 0.2)
    + (df["Vol_Score"] * 0.2)
)

# 결측치 제거 및 최근 데이터 추출
df = df.dropna(subset=["RSI", "Disparity_Score", "BB_Score", "Vol_Score", "FGI"])
latest = df.iloc[-1]
prev = df.iloc[-2]

fgi_val = latest["FGI"]

if fgi_val <= 24:
    status, color = "극도의 공포 (과매도 구간 / 반등 기대)", "#ff4b4b"
elif fgi_val <= 49:
    status, color = "공포 (투자 심리 위축)", "#ffa500"
elif fgi_val <= 74:
    status, color = "탐욕 (매수세 우위)", "#90ee90"
else:
    status, color = "극도의 탐욕 (과매수 구간 / 단기 고점 주의)", "#008000"

# ============================================================
# 텔레그램 알림 발송 로직  [추가 기능 4]
# ============================================================
if telegram_enabled and telegram_token and telegram_chat_id:
    today_str = datetime.now().strftime("%Y-%m-%d")
    alert_key = f"alert_sent_{today_str}"
    if fgi_val <= 24 or fgi_val >= 75:
        if not st.session_state.get(alert_key, False):
            zone = "극도의 공포 🔴" if fgi_val <= 24 else "극도의 탐욕 🟢"
            msg = (
                f"[SK하이닉스 FGI 알림]\n"
                f"현재 FGI: {fgi_val:.1f} ({zone})\n"
                f"현재가: {int(latest['Close']):,}원\n"
                f"기준일: {latest.name.strftime('%Y-%m-%d')}"
            )
            if send_telegram_alert(telegram_token, telegram_chat_id, msg):
                st.session_state[alert_key] = True
                st.sidebar.success("✅ 오늘자 알림이 발송되었습니다.")
            else:
                st.sidebar.error("❌ 알림 발송에 실패했습니다. 토큰/Chat ID를 확인해 주세요.")
    else:
        st.session_state[alert_key] = False

# ============================================================
# 데이터 신선도 표시  [개선 6 / 요청 반영]
# ============================================================
st.caption(
    f"📅 데이터 기준일: **{latest.name.strftime('%Y-%m-%d')}**  ·  "
    f"마지막 확인 시각: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**  ·  "
    "Yahoo Finance 기준 약 15~20분 지연된 일봉 데이터입니다."
)

# --- UI 렌더링 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("최신 주가 및 상태")
    price_diff = latest["Close"] - prev["Close"]
    price_pct = (price_diff / prev["Close"]) * 100
    st.metric(
        label=f"기준일: {latest.name.strftime('%Y-%m-%d')}",
        value=f"{int(latest['Close']):,} 원",
        delta=f"{int(price_diff):,}원 ({price_pct:.2f}%)",
    )

    st.markdown(
        f"**현재 시장 심리:** <span style='color:{color}; font-size:1.2em; font-weight:bold;'>{status}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("세부 지표 기여도")
    st.write(f"- **RSI (30% 반영):** {latest['RSI']:.1f} 점")
    st.progress(latest["RSI"] / 100)
    st.write(f"- **이격도 (30% 반영):** {latest['Disparity_Score']:.1f} 점")
    st.progress(latest["Disparity_Score"] / 100)
    st.write(f"- **볼린저 밴드 (20% 반영):** {latest['BB_Score']:.1f} 점")
    st.progress(latest["BB_Score"] / 100)
    st.write(f"- **거래량 강도 (20% 반영, 방향성 반영):** {latest['Vol_Score']:.1f} 점")
    st.progress(latest["Vol_Score"] / 100)

with col2:
    # 게이지 차트 (메인 인디케이터)
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fgi_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "통합 공포탐욕지수", "font": {"size": 20}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "rgba(0,0,0,0)"},  # 바늘 형태로 만들기 위해 바 색상 투명 처리
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 24], "color": "#ff4b4b", "name": "극도의 공포"},
                    {"range": [24, 49], "color": "#ffa500", "name": "공포"},
                    {"range": [49, 74], "color": "#90ee90", "name": "탐욕"},
                    {"range": [74, 100], "color": "#008000", "name": "극도의 탐욕"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 6},
                    "thickness": 0.75,
                    "value": fgi_val,
                },
            },
        )
    )
    fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# 히스토리컬 차트 + 정량적 백테스트  [추가 기능 2]
# ============================================================
st.markdown("---")
st.subheader("📊 FGI 히스토리 및 백테스트")
st.markdown("과거 FGI 지수가 바닥(극도의 공포)일 때 주가의 움직임과, 고점(극도의 탐욕)일 때 주가의 움직임을 비교해보세요.")

df_3mo = df.tail(60)  # 약 3개월(영업일 기준 60일)

fig_line = go.Figure()
fig_line.add_trace(
    go.Scatter(
        x=df_3mo.index, y=df_3mo["FGI"], name="FGI 지수 (좌측 축)",
        line=dict(color="purple", width=3), yaxis="y1",
    )
)
fig_line.add_trace(
    go.Scatter(
        x=df_3mo.index, y=df_3mo["Close"], name="종가 (우측 축)",
        line=dict(color="gray", width=2, dash="dot"), yaxis="y2",
    )
)
fig_line.update_layout(
    xaxis=dict(title="날짜"),
    yaxis=dict(title="FGI 지수 (0~100)", range=[0, 100], side="left"),
    yaxis2=dict(title="주가 (KRW)", side="right", overlaying="y", showgrid=False),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("#### 🔍 극단 구간 진입 이후 실제 수익률 (정량적 검증)")
st.caption(
    "FGI가 극도의 공포(≤24) 또는 극도의 탐욕(≥75)에 도달했던 날 이후, "
    "N거래일 뒤 평균/중앙값 수익률입니다. 표본 수(발생 횟수)가 적으면 통계적 신뢰도가 낮을 수 있으니 참고용으로만 활용하세요."
)


def forward_return_stats(price: pd.Series, mask: pd.Series, horizons=(5, 20, 60)) -> pd.DataFrame:
    """조건을 만족한 시점 이후 N거래일 뒤의 수익률 분포(평균/중앙값/표본 수)를 계산한다."""
    positions = np.where(mask.to_numpy())[0]
    rows = []
    for h in horizons:
        rets = [
            (price.iloc[p + h] / price.iloc[p] - 1) * 100
            for p in positions
            if p + h < len(price)
        ]
        rows.append(
            {
                "기간": f"{h}거래일 후",
                "표본 수": len(rets),
                "평균 수익률(%)": round(float(np.mean(rets)), 2) if rets else None,
                "중앙값 수익률(%)": round(float(np.median(rets)), 2) if rets else None,
            }
        )
    return pd.DataFrame(rows)


fear_mask = df["FGI"] <= 24
greed_mask = df["FGI"] >= 75

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**🔴 극도의 공포 진입 이후**")
    st.dataframe(forward_return_stats(df["Close"], fear_mask), use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**🟢 극도의 탐욕 진입 이후**")
    st.dataframe(forward_return_stats(df["Close"], greed_mask), use_container_width=True, hide_index=True)

# ============================================================
# 반도체 업종 특화 지표  [추가 기능 3]
# ============================================================
st.markdown("---")
st.subheader("🏭 반도체 업종 특화 지표")

tab1, tab2, tab3 = st.tabs(["💱 원/달러 환율", "📈 삼성전자 상대강도", "🏦 수급 동향 (외국인/공매도)"])

with tab1:
    if usdkrw_df is not None and not usdkrw_df.empty:
        fig_fx = go.Figure()
        fig_fx.add_trace(go.Scatter(x=usdkrw_df.index, y=usdkrw_df["Close"], name="USD/KRW", line=dict(color="teal")))
        fig_fx.update_layout(yaxis_title="원/달러 환율", xaxis_title="날짜")
        st.plotly_chart(fig_fx, use_container_width=True)
        st.caption("반도체는 수출 비중이 높아 원화 약세(환율 상승)가 실적에 우호적으로 작용하는 경향이 있습니다.")
    else:
        st.info("환율 데이터를 현재 불러올 수 없습니다.")

with tab2:
    if samsung_df is not None and not samsung_df.empty:
        common_len = min(120, len(df), len(samsung_df))
        sk_norm = df["Close"].tail(common_len)
        ss_norm = samsung_df["Close"].tail(common_len)
        merged = pd.DataFrame(
            {
                "SK하이닉스": sk_norm / sk_norm.iloc[0] * 100,
                "삼성전자": ss_norm / ss_norm.iloc[0] * 100,
            }
        ).dropna()
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(x=merged.index, y=merged["SK하이닉스"], name="SK하이닉스 (지수화)", line=dict(color="crimson")))
        fig_rel.add_trace(go.Scatter(x=merged.index, y=merged["삼성전자"], name="삼성전자 (지수화)", line=dict(color="navy")))
        fig_rel.update_layout(yaxis_title="상대 수익률 (구간 시작일=100)", xaxis_title="날짜")
        st.plotly_chart(fig_rel, use_container_width=True)
        st.caption("두 종목 모두 비교 구간 시작일을 100으로 지수화하여 상대적 강도를 비교합니다.")
    else:
        st.info("삼성전자 비교 데이터를 현재 불러올 수 없습니다.")

with tab3:
    if investor_df is not None and short_df is not None and not investor_df.empty and not short_df.empty:
        try:
            if "외국인합계" in investor_df.columns:
                fig_supply = go.Figure()
                fig_supply.add_trace(
                    go.Bar(x=investor_df.index, y=investor_df["외국인합계"], name="외국인 순매수대금(원)")
                )
                fig_supply.update_layout(yaxis_title="순매수대금 (원)", xaxis_title="날짜")
                st.plotly_chart(fig_supply, use_container_width=True)
                st.caption("외국인 순매수대금이 지속적으로 (+)이면 수급상 우호적 신호로 해석할 수 있습니다.")

            if "비중" in short_df.columns:
                fig_short = go.Figure()
                fig_short.add_trace(
                    go.Scatter(x=short_df.index, y=short_df["비중"], name="공매도 비중(%)", line=dict(color="orange"))
                )
                fig_short.update_layout(yaxis_title="공매도 비중 (%)", xaxis_title="날짜")
                st.plotly_chart(fig_short, use_container_width=True)
                st.caption("공매도 비중이 급등하면 하락 베팅이 늘고 있다는 뜻으로, 심리적 공포 신호로 참고할 수 있습니다.")
        except Exception:
            st.info("KRX 수급 데이터 형식을 처리하는 중 문제가 발생했습니다.")
    else:
        st.info(
            "ℹ️ KRX 공매도/외국인 순매수 데이터를 현재 이용할 수 없습니다. "
            "(requirements.txt에 `pykrx` 추가 필요 및 KRX 서버 응답 지연 가능성 있음)"
        )
