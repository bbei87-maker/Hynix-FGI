import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(page_title="SK하이닉스 자동 FGI", page_icon="📈", layout="wide")

st.title("🤖 SK하이닉스 공포탐욕지수(FGI) 자동 계산 웹앱")
st.markdown("야후 파이낸스(Yahoo Finance) API를 통해 최신 주가 데이터를 **자동으로 수집**하여 실시간에 가까운 투자 심리를 분석합니다. (수동 입력 불필요)")

@st.cache_data(ttl=60) # 1 단위로 데이터 캐싱 (API 호출 최적화)
def get_hynix_data():
    df = yf.download("000660.KS", period="1y", progress=False)
    # yfinance 최신 버전의 MultiIndex 컬럼 평탄화 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

with st.spinner('금융 시장에서 최신 주가 데이터를 자동으로 분석하는 중입니다...'):
    df = get_hynix_data()
    
    if df.empty:
        st.error("데이터를 불러오는데 실패했습니다. 주말이나 장 마감 후 일시적 현상일 수 있습니다.")
        st.stop()

    # 1. 지표 계산 (PRD 로직 적용)
    # RSI (14일)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # 이격도 (60일)
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Disparity'] = (df['Close'] / df['MA60'] - 1) * 100
    df['Disparity_Score'] = ((df['Disparity'] + 20) / 40 * 100).clip(0, 100)

    # 볼린저 밴드 (20일)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower'] = df['MA20'] - (df['STD20'] * 2)
    df['BB_Score'] = (((df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])) * 100).clip(0, 100)

    # 거래량 강도 (5일 vs 20일)
    df['Vol5'] = df['Volume'].rolling(window=5).mean()
    df['Vol20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Score'] = (((df['Vol5'] / df['Vol20']) - 0.5) / 1.0 * 100).clip(0, 100)

    # 2. 최종 FGI 지수 산출 (가중치 적용)
    df['FGI'] = (df['RSI'] * 0.3) + (df['Disparity_Score'] * 0.3) + (df['BB_Score'] * 0.2) + (df['Vol_Score'] * 0.2)
    
    # 결측치 제거 및 최근 데이터 추출
    df = df.dropna()
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    fgi_val = latest['FGI']
    
    if fgi_val <= 24: 
        status, color = "극도의 공포 (과매도 구간 / 반등 기대)", "#ff4b4b"
    elif fgi_val <= 49: 
        status, color = "공포 (투자 심리 위축)", "#ffa500"
    elif fgi_val <= 74: 
        status, color = "탐욕 (매수세 우위)", "#90ee90"
    else: 
        status, color = "극도의 탐욕 (과매수 구간 / 단기 고점 주의)", "#008000"

    # --- UI 렌더링 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("최신 주가 및 상태")
        price_diff = latest['Close'] - prev['Close']
        price_pct = (price_diff / prev['Close']) * 100
        st.metric(label=f"기준일: {latest.name.strftime('%Y-%m-%d')}", 
                  value=f"{int(latest['Close']):,} 원", 
                  delta=f"{int(price_diff):,}원 ({price_pct:.2f}%)")
        
        st.markdown(f"**현재 시장 심리:** <span style='color:{color}; font-size:1.2em; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("세부 지표 기여도")
        st.write(f"- **RSI (30% 반영):** {latest['RSI']:.1f} 점")
        st.progress(latest['RSI'] / 100)
        st.write(f"- **이격도 (30% 반영):** {latest['Disparity_Score']:.1f} 점")
        st.progress(latest['Disparity_Score'] / 100)
        st.write(f"- **볼린저 밴드 (20% 반영):** {latest['BB_Score']:.1f} 점")
        st.progress(latest['BB_Score'] / 100)
        st.write(f"- **거래량 강도 (20% 반영):** {latest['Vol_Score']:.1f} 점")
        st.progress(latest['Vol_Score'] / 100)

    with col2:
        # 게이지 차트 (메인 인디케이터)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fgi_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "통합 공포탐욕지수", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"}, # 바늘 형태로 만들기 위해 바 색상 투명 처리
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 24], 'color': "#ff4b4b", 'name': '극도의 공포'},
                    {'range': [24, 49], 'color': "#ffa500", 'name': '공포'},
                    {'range': [49, 74], 'color': "#90ee90", 'name': '탐욕'},
                    {'range': [74, 100], 'color': "#008000", 'name': '극도의 탐욕'}],
                'threshold': {
                    'line': {'color': "black", 'width': 6},
                    'thickness': 0.75,
                    'value': fgi_val
                }
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 하단: 히스토리컬 차트
    st.markdown("---")
    st.subheader("📊 최근 3개월 지수 흐름 백테스팅")
    st.markdown("과거 FGI 지수가 바닥(극도의 공포)일 때 주가의 움직임과, 고점(극도의 탐욕)일 때 주가의 움직임을 비교해보세요.")
    
    df_3mo = df.tail(60) # 약 3개월(영업일 기준 60일)
    
    fig_line = go.Figure()
    # FGI 라인
    fig_line.add_trace(go.Scatter(x=df_3mo.index, y=df_3mo['FGI'], name="FGI 지수 (좌측 축)", line=dict(color='purple', width=3), yaxis="y1"))
    # 주가 라인
    fig_line.add_trace(go.Scatter(x=df_3mo.index, y=df_3mo['Close'], name="종가 (우측 축)", line=dict(color='gray', width=2, dash='dot'), yaxis="y2"))
    
    fig_line.update_layout(
        xaxis=dict(title="날짜"),
        yaxis=dict(title="FGI 지수 (0~100)", range=[0, 100], side="left"),
        yaxis2=dict(title="주가 (KRW)", side="right", overlaying="y", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)
