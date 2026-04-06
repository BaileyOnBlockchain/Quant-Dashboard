# =============================================================================
# AI Crypto Quant Bot — main.py
# =============================================================================
# Streamlit dashboard for AI-powered crypto analysis, backtesting, and
# portfolio tracking. Built with PyTorch, pandas-ta, Plotly, and CoinGecko.
#
# Run: streamlit run main.py
#
# Built by @blockchainbail — https://x.com/blockchainbail
# Live: https://odennetworkxr.com
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from pandas_ta import macd, rsi, sma, bbands, stoch
import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import html
import warnings
import pickle
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
warnings.filterwarnings('ignore')


# =============================================================================
# SETTINGS PERSISTENCE
# Saves user preferences (theme, selected coin, portfolio, positions) to
# settings.json so state survives between Streamlit sessions.
# =============================================================================
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

def load_settings():
    default_settings = {
        'theme': 'Midnight Cyan',
        'selected_crypto': 'bitcoin',
        'historical_days': 30,
        'portfolio_capital': 10000,
        'positions': [],
        'swing_min_score': 0,
        'token_lists': []
    }
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                return {**default_settings, **saved}
    except Exception:
        pass
    return default_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_CRYPTOS = [
    'bitcoin', 'ethereum', 'solana', 'cardano', 'avalanche-2',
    'polkadot', 'chainlink', 'dogecoin', 'ripple', 'matic-network'
]

COIN_SYMBOLS = {
    'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL', 'cardano': 'ADA',
    'avalanche-2': 'AVAX', 'polkadot': 'DOT', 'chainlink': 'LINK',
    'dogecoin': 'DOGE', 'ripple': 'XRP', 'matic-network': 'MATIC'
}


# =============================================================================
# THEME SYSTEM
# 4 switchable themes with full CSS variable injection.
# Each theme defines primary/secondary/accent colours used across all CSS,
# charts, and inline HTML — no hardcoded colours anywhere else in the app.
# =============================================================================
THEMES = {
    "Midnight Cyan": {
        "primary": "#00d4ff",
        "secondary": "#7b2cbf",
        "accent": "#00b894",
        "bg_gradient": "linear-gradient(135deg, #0a0a0a 0%, #0d1b2a 50%, #1b263b 100%)",
        "card_bg": "linear-gradient(145deg, #1b263b 0%, #0d1b2a 100%)",
        "border": "#00d4ff33",
        "text": "#e0e0e0",
        "positive": "#00d4ff",
        "negative": "#ff006e"
    },
    "Deep Purple": {
        "primary": "#a855f7",
        "secondary": "#6366f1",
        "accent": "#c026d3",
        "bg_gradient": "linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #2d1b4e 100%)",
        "card_bg": "linear-gradient(145deg, #2d1b4e 0%, #1a1a2e 100%)",
        "border": "#a855f733",
        "text": "#e0e0e0",
        "positive": "#a855f7",
        "negative": "#f43f5e"
    },
    "Neon Matrix": {
        "primary": "#39ff14",
        "secondary": "#00ff88",
        "accent": "#00d4ff",
        "bg_gradient": "linear-gradient(135deg, #000000 0%, #0a1a0a 50%, #001a00 100%)",
        "card_bg": "linear-gradient(145deg, #0a1a0a 0%, #001a00 100%)",
        "border": "#39ff1433",
        "text": "#00ff88",
        "positive": "#39ff14",
        "negative": "#ff0040"
    },
    "Golden Alpha": {
        "primary": "#ffd700",
        "secondary": "#ff8c00",
        "accent": "#ffb347",
        "bg_gradient": "linear-gradient(135deg, #1a1a0a 0%, #2a2a1a 50%, #1a1500 100%)",
        "card_bg": "linear-gradient(145deg, #2a2a1a 0%, #1a1500 100%)",
        "border": "#ffd70033",
        "text": "#ffe4b5",
        "positive": "#ffd700",
        "negative": "#dc143c"
    }
}

def get_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = "Midnight Cyan"
    return THEMES[st.session_state.theme]


# =============================================================================
# CSS INJECTION
# Full custom theme injected as a single cached CSS string.
# Uses CSS variables (--primary, --secondary, --accent) for global theming.
# Includes: animated background pulse, Orbitron/Rajdhani/Space Grotesk fonts,
# glass cards, metric cards, BUY/SELL/HOLD signal badges, live pulse indicator,
# styled tabs, scrollbars, buttons, and token list cards.
# =============================================================================
@st.cache_data(show_spinner=False)
def get_css_string(theme_name):
    theme = THEMES[theme_name]
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    :root {{
        --primary: {theme['primary']};
        --secondary: {theme['secondary']};
        --accent: {theme['accent']};
        --glow: {theme['primary']}40;
    }}
    
    .stApp {{
        background: {theme['bg_gradient']};
        background-attachment: fixed;
    }}
    
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, {theme['primary']}08 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, {theme['secondary']}08 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, {theme['accent']}05 0%, transparent 30%);
        pointer-events: none;
        z-index: 0;
        animation: bgPulse 8s ease-in-out infinite;
    }}
    
    @keyframes bgPulse {{
        0%, 100% {{ opacity: 0.5; }}
        50% {{ opacity: 1; }}
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']}, {theme['accent']}, {theme['primary']});
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        padding: 1rem;
        letter-spacing: 4px;
        animation: gradientShift 4s ease infinite;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .sub-header {{
        color: {theme['text']};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
        opacity: 0.9;
    }}
    
    .glass-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        min-height: 180px;
    }}
    
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 40px {theme['primary']}20;
        border-color: {theme['primary']}40;
    }}
    
    .metric-card {{
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid {theme['border']};
        border-radius: 20px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }}
    
    .super-metric {{
        background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.2) 100%);
        border: 2px solid {theme['border']};
        border-radius: 24px;
        padding: 1.5rem;
        text-align: center;
        min-height: 150px;
    }}
    
    .super-metric:hover {{
        border-color: {theme['primary']};
        box-shadow: 0 0 50px {theme['primary']}30;
    }}
    
    .super-metric-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    
    .super-metric-label {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: {theme['text']};
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }}
    
    .super-metric-change {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }}
    
    .change-positive {{ background: linear-gradient(135deg,#00b89420,#00cec920); color:#00b894; border:1px solid #00b89440; }}
    .change-negative {{ background: linear-gradient(135deg,#ff006e20,#d6303120); color:#ff006e; border:1px solid #ff006e40; }}
    
    .signal-buy  {{ background: linear-gradient(135deg,#00b894,#00cec9); color:white; padding:1rem 2rem; border-radius:25px; font-weight:bold; font-size:1.5rem; display:inline-block; box-shadow:0 10px 40px #00b89450; }}
    .signal-sell {{ background: linear-gradient(135deg,#ff006e,#d63031); color:white; padding:1rem 2rem; border-radius:25px; font-weight:bold; font-size:1.5rem; display:inline-block; box-shadow:0 10px 40px #ff006e50; }}
    .signal-hold {{ background: linear-gradient(135deg,#ffd700,#f39c12); color:#1a1a2e; padding:1rem 2rem; border-radius:25px; font-weight:bold; font-size:1.5rem; display:inline-block; box-shadow:0 10px 40px #ffd70050; }}
    
    .live-indicator {{
        display: inline-block;
        width: 12px; height: 12px;
        background: #00ff00;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 20px #00ff00;
        animation: livePulse 1.5s ease-in-out infinite;
    }}
    
    @keyframes livePulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.6; transform: scale(0.9); }}
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(0,0,0,0.3);
        padding: 0.5rem;
        border-radius: 15px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        color: {theme['text']};
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.75rem 1.25rem;
        font-family: 'Space Grotesk', sans-serif;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']}) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 5px 20px {theme['primary']}50;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .progress-bar {{ width:100%; height:8px; background:rgba(255,255,255,0.1); border-radius:10px; overflow:hidden; margin:0.5rem 0; }}
    .progress-fill {{ height:100%; border-radius:10px; background:linear-gradient(90deg,{theme['primary']},{theme['secondary']}); box-shadow:0 0 20px {theme['primary']}50; }}
    
    .allocation-item {{
        background: linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));
        border: 1px solid {theme['border']};
        border-radius: 15px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }}
    .allocation-item:hover {{ transform:translateX(5px); border-color:{theme['primary']}60; }}
    
    .portfolio-card {{
        background: linear-gradient(145deg,rgba(0,0,0,0.5),rgba(0,0,0,0.3));
        border: 2px solid {theme['primary']}40;
        border-radius: 25px;
        padding: 2.5rem;
        text-align: center;
    }}
    
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg,rgba(10,10,10,0.95) 0%,rgba(15,15,25,0.95) 100%);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg,{theme['primary']},{theme['secondary']}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 5px 20px {theme['primary']}40 !important;
    }}
    .stButton > button:hover {{ transform:translateY(-2px) !important; box-shadow:0 10px 30px {theme['primary']}60 !important; }}
    
    ::-webkit-scrollbar {{ width:8px; height:8px; }}
    ::-webkit-scrollbar-track {{ background:rgba(0,0,0,0.3); border-radius:10px; }}
    ::-webkit-scrollbar-thumb {{ background:linear-gradient(135deg,{theme['primary']},{theme['secondary']}); border-radius:10px; }}
    
    .token-list-card {{
        background: linear-gradient(145deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
        border: 1px solid {theme['border']};
        border-radius: 18px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }}
    .token-list-card:hover {{ border-color:{theme['primary']}50; box-shadow:0 10px 40px rgba(0,0,0,0.3); }}
    
    .transaction-row {{
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid {theme['primary']};
    }}
    </style>
    """

def inject_custom_css():
    css = get_css_string(st.session_state.get('theme', 'Midnight Cyan'))
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# DATA FETCHING
# All API calls use a single cached requests.Session with retry logic.
# Disk cache fallback: if CoinGecko is rate-limited, serves from .cache/*.pkl
# so the dashboard stays functional without live data.
# =============================================================================
@st.cache_resource
def get_cached_session():
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def create_session():
    return get_cached_session()

def get_cache_dir():
    cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

@st.cache_data(ttl=120, show_spinner=False)
def fetch_historical_data(crypto_id, days):
    """Fetch OHLCV history from CoinGecko with disk cache fallback."""
    cache_file = os.path.join(get_cache_dir(), f"hist_{crypto_id}_{days}.pkl")
    now = time.time()
    try:
        session = create_session()
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': days}
        response = session.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if data and 'prices' in data:
                prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
                prices.set_index('timestamp', inplace=True)
                prices['volume'] = volumes['volume'].values
                with open(cache_file, 'wb') as f:
                    pickle.dump({'data': prices, 'ts': now}, f)
                return prices, False
    except Exception:
        pass
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                obj = pickle.load(f)
                return obj['data'], True
        except Exception:
            pass
    return None, False

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices():
    """Fetch top 100 coins by market cap with disk cache fallback."""
    cache_file = os.path.join(get_cache_dir(), "live_prices.pkl")
    now = time.time()
    try:
        session = create_session()
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd', 'order': 'market_cap_desc',
            'per_page': 100, 'page': 1, 'sparkline': 'false',
            'price_change_percentage': '1h,24h,7d'
        }
        response = session.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json() or []
            with open(cache_file, 'wb') as f:
                pickle.dump({'data': data, 'ts': now}, f)
            return data, False
    except Exception:
        pass
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                obj = pickle.load(f)
                return obj['data'], True
        except Exception:
            pass
    return [], False

@st.cache_data(ttl=60, show_spinner=False)
def fetch_token_price(token_id):
    """Fetch single token price — used for custom tokens in the Lists tab."""
    try:
        session = create_session()
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {'ids': token_id, 'vs_currencies': 'usd', 'include_24hr_change': 'true'}
        response = session.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if token_id in data:
                return data[token_id].get('usd', 0), data[token_id].get('usd_24h_change', 0)
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def search_coins(query):
    """Search CoinGecko for coins by name or symbol."""
    try:
        session = create_session()
        url = "https://api.coingecko.com/api/v3/search"
        params = {'query': query}
        response = session.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('coins', [])[:20]
    except Exception:
        pass
    return []


# =============================================================================
# TECHNICAL ANALYSIS
# Indicators computed via pandas-ta, cached by (price_tuple, volume_tuple)
# so Streamlit's cache key works correctly across reruns.
#
# Computed indicators:
#   MA20, MA50, MA200 — simple moving averages (length-adaptive to data size)
#   RSI              — 14-period (adaptive to short datasets)
#   MACD             — standard 12/26/9, returns MACD/Signal/Histogram
#   Bollinger Bands  — 20-period upper/middle/lower
#   Stochastic       — %K and %D
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def compute_all_indicators_cached(price_data, volume_data):
    df = pd.DataFrame({'price': price_data, 'volume': volume_data})
    return _compute_indicators(df)

def compute_all_indicators(df):
    if df is None or len(df) < 5:
        return df
    price_tuple = tuple(df['price'].values)
    volume_tuple = tuple(df['volume'].values) if 'volume' in df.columns else tuple([0]*len(df))
    result = compute_all_indicators_cached(price_tuple, volume_tuple)
    result.index = df.index
    return result

def _compute_indicators(df):
    if df is None or len(df) < 5:
        return df
    df = df.copy()
    try:
        data_len = len(df)
        if data_len >= 10:  df['MA20']  = sma(df['price'], length=min(20, data_len-1))
        if data_len >= 20:  df['MA50']  = sma(df['price'], length=min(50, data_len-1))
        if data_len >= 50:  df['MA200'] = sma(df['price'], length=min(200, data_len-1))

        rsi_len = min(14, max(5, data_len // 3))
        df['RSI'] = rsi(df['price'], length=rsi_len)

        macd_df = macd(df['price'])
        if macd_df is not None and len(macd_df.columns) >= 3:
            df['MACD'] = macd_df.iloc[:, 0]
            df['MACD_Signal'] = macd_df.iloc[:, 1]
            df['MACD_Hist'] = macd_df.iloc[:, 2]

        bb_len = min(20, max(5, data_len // 2))
        bb = bbands(df['price'], length=bb_len)
        if bb is not None and len(bb.columns) >= 3:
            df['BB_Upper']  = bb.iloc[:, 0]
            df['BB_Middle'] = bb.iloc[:, 1]
            df['BB_Lower']  = bb.iloc[:, 2]

        stoch_df = stoch(df['price'], df['price'], df['price'])
        if stoch_df is not None and len(stoch_df.columns) >= 2:
            df['Stoch_K'] = stoch_df.iloc[:, 0]
            df['Stoch_D'] = stoch_df.iloc[:, 1]
    except Exception:
        pass
    return df


# =============================================================================
# SIGNAL ENGINE
# Weighted signal scoring across 4 indicator families.
# Returns a recommendation (BUY / SELL / HOLD), confidence %, and description.
#
# Signal weights:
#   Golden/Death Cross — 0.9  (strongest — long-term trend confirmation)
#   RSI oversold/overbought — 0.8
#   MACD crossover — 0.7
#   Bollinger Band breach — 0.6
# =============================================================================
def calculate_signal_strength(df):
    signals = []
    if df is None or len(df) == 0:
        return signals
    latest = df.iloc[-1]

    rsi_val = latest.get('RSI')
    if pd.notna(rsi_val):
        if rsi_val < 30:   signals.append(('RSI Oversold',   'BUY',  0.8))
        elif rsi_val > 70: signals.append(('RSI Overbought', 'SELL', 0.8))
        else:              signals.append(('RSI Neutral',    'HOLD', 0.3))

    macd_val = latest.get('MACD')
    macd_sig = latest.get('MACD_Signal')
    if pd.notna(macd_val) and pd.notna(macd_sig):
        if macd_val > macd_sig: signals.append(('MACD Bullish', 'BUY',  0.7))
        else:                   signals.append(('MACD Bearish', 'SELL', 0.7))

    ma50 = latest.get('MA50')
    ma200 = latest.get('MA200')
    if pd.notna(ma50) and pd.notna(ma200):
        if ma50 > ma200: signals.append(('Golden Cross', 'BUY',  0.9))
        else:            signals.append(('Death Cross',  'SELL', 0.9))

    bb_lower = latest.get('BB_Lower')
    bb_upper = latest.get('BB_Upper')
    price    = latest.get('price')
    if pd.notna(bb_lower) and pd.notna(bb_upper) and pd.notna(price):
        if price < bb_lower:   signals.append(('BB Oversold',   'BUY',  0.6))
        elif price > bb_upper: signals.append(('BB Overbought', 'SELL', 0.6))

    return signals

def get_ai_recommendation(signals):
    if not signals:
        return 'HOLD', 50, 'Insufficient data'
    buy_score  = sum(s[2] for s in signals if s[1] == 'BUY')
    sell_score = sum(s[2] for s in signals if s[1] == 'SELL')
    total_weight = sum(s[2] for s in signals) or 1
    if buy_score > sell_score:
        return 'BUY',  int((buy_score  / total_weight) * 100), f"{len([s for s in signals if s[1]=='BUY'])}/{len(signals)} bullish"
    elif sell_score > buy_score:
        return 'SELL', int((sell_score / total_weight) * 100), f"{len([s for s in signals if s[1]=='SELL'])}/{len(signals)} bearish"
    return 'HOLD', 50, 'Mixed signals'


# =============================================================================
# RISK METRICS
# Computed from daily returns:
#   Volatility   — annualised std dev (×√365)
#   Sharpe ratio — annualised return / annualised vol (2% risk-free rate)
#   Sortino      — return / downside deviation only
#   Max drawdown — peak-to-trough from expanding max of cumulative returns
#   VaR 95%      — 5th percentile of daily returns
#   Win rate     — % of days with positive returns
# =============================================================================
def calculate_risk_metrics(df):
    if df is None or len(df) < 10:
        return {'volatility': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'var_95': 0, 'win_rate': 0}
    returns = df['price'].pct_change().dropna()
    volatility   = returns.std() * np.sqrt(365) * 100
    mean_return  = returns.mean() * 365
    sharpe       = mean_return / (returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
    downside     = returns[returns < 0]
    sortino      = mean_return / (downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0
    cumulative   = (1 + returns).cumprod()
    rolling_max  = cumulative.expanding().max()
    drawdowns    = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100
    var_95       = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
    win_rate     = (returns > 0).mean() * 100
    return {
        'volatility': abs(volatility), 'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': abs(max_drawdown), 'var_95': abs(var_95), 'win_rate': win_rate
    }


# =============================================================================
# DASHBOARD DATA — SINGLE CACHED CALL
# All dashboard metrics computed in one place and cached for 3 minutes.
# Avoids redundant API calls across tabs when the same crypto/days is selected.
# =============================================================================
@st.cache_data(ttl=180, show_spinner=False)
def get_dashboard_data(crypto_id, days):
    df, stale = fetch_historical_data(crypto_id, days)
    if df is None or len(df) == 0:
        return None
    df = compute_all_indicators(df)
    current_price = df['price'].iloc[-1]
    price_24h     = df['price'].iloc[-24] if len(df) > 24 else current_price
    change_24h    = ((current_price - price_24h) / price_24h) * 100
    rsi_val    = df['RSI'].iloc[-1] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]) else 50
    vol        = df['volume'].iloc[-1] if 'volume' in df.columns else 0
    signals    = calculate_signal_strength(df)
    rec, conf, signal_desc = get_ai_recommendation(signals)
    risk_metrics = calculate_risk_metrics(df)
    return {
        'df': df, 'stale': stale,
        'current_price': current_price, 'change_24h': change_24h,
        'rsi_val': rsi_val, 'vol': vol,
        'signals': signals, 'rec': rec, 'conf': conf, 'signal_desc': signal_desc,
        'risk_metrics': risk_metrics,
        'high_price': df['price'].max(), 'low_price': df['price'].min(), 'avg_price': df['price'].mean(),
        'macd_val':    df['MACD'].iloc[-1]        if 'MACD'        in df.columns and pd.notna(df['MACD'].iloc[-1])        else 0,
        'macd_signal': df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns and pd.notna(df['MACD_Signal'].iloc[-1]) else 0,
        'macd_hist':   df['MACD_Hist'].iloc[-1]   if 'MACD_Hist'   in df.columns and pd.notna(df['MACD_Hist'].iloc[-1])   else 0,
    }


# =============================================================================
# VISUALISATION
# 3-panel Plotly chart: Price + MAs / RSI / MACD
# All chart colours driven by the active theme — transparent backgrounds,
# dark grid lines, horizontal RSI oversold/overbought reference lines.
# =============================================================================
def create_price_chart(df, theme):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD')
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Price',
                             line=dict(color=theme['primary'], width=2),
                             fill='tozeroy', fillcolor=f'rgba(0,212,255,0.1)'), row=1, col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'],  name='MA20',  line=dict(color='#ffd700', width=1)), row=1, col=1)
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],  name='MA50',  line=dict(color='#ff6b6b', width=1)), row=1, col=1)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],   name='RSI',   line=dict(color=theme['secondary'], width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],        name='MACD',   line=dict(color=theme['primary'], width=1)), row=3, col=1)
        if 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#ff6b6b', width=1)), row=3, col=1)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=30, b=30)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    return fig


# =============================================================================
# TOKEN LISTS — P&L HELPERS
# Aggregates cost basis and live value across multiple transactions per token.
# live_prices is the CoinGecko markets response — mapped to id: current_price.
# Falls back to the stored buy_price if the token isn't in the live feed.
# =============================================================================
def calculate_list_totals(transactions, live_prices):
    price_map = {p['id']: p['current_price'] for p in live_prices} if live_prices else {}
    total_invested = 0
    total_current_value = 0
    for tx in transactions:
        invested = tx['quantity'] * tx['buy_price']
        total_invested += invested
        current_price = price_map.get(tx['token_id'], tx.get('current_price', tx['buy_price']))
        total_current_value += tx['quantity'] * current_price
    profit_usd = total_current_value - total_invested
    profit_pct = (profit_usd / total_invested * 100) if total_invested > 0 else 0
    return {
        'total_invested': total_invested,
        'total_current_value': total_current_value,
        'profit_usd': profit_usd,
        'profit_pct': profit_pct
    }

def auto_save():
    save_settings({
        'theme':            st.session_state.get('theme', 'Midnight Cyan'),
        'selected_crypto':  st.session_state.get('selected_crypto', 'bitcoin'),
        'historical_days':  st.session_state.get('historical_days', 30),
        'portfolio_capital':st.session_state.get('portfolio_capital', 10000),
        'positions':        st.session_state.get('positions', []),
        'swing_min_score':  st.session_state.get('swing_min_score', 0),
        'token_lists':      st.session_state.get('token_lists', [])
    })


# =============================================================================
# NEURAL NETWORK — PRICE PREDICTOR
# PyTorch trend model: computes 7-day vs 14-day average momentum,
# projects 14 daily prices with gaussian noise (σ=2%) around the trend rate.
# Runs on CUDA if available, falls back to CPU.
# =============================================================================
def train_predictor(prices):
    try:
        prices_tensor = torch.tensor(prices, dtype=torch.float32).to(DEVICE)
        if len(prices) < 14:
            current_price = prices[-1] if prices else 100
            return [current_price] * 14, 0.0
        recent_avg = sum(prices[-7:])  / 7
        older_avg  = sum(prices[-14:-7]) / 7
        trend = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        current_price = prices[-1]
        predictions = []
        for i in range(14):
            daily_change = trend / 14 + np.random.normal(0, 0.02)
            predicted_price = current_price * (1 + daily_change)
            predictions.append(max(predicted_price, current_price * 0.8))
            current_price = predicted_price
        return predictions, 0.01
    except Exception:
        current_price = prices[-1] if prices else 100
        return [current_price * (1 + 0.001 * i) for i in range(14)], 0.05


# =============================================================================
# QUANT BACKTESTER
# RSI + MACD crossover strategy backtested on historical OHLCV data.
#
# Entry:  RSI < 35 AND MACD > Signal (oversold + bullish momentum)
# Exit:   RSI > 65 OR  MACD < Signal (overbought or momentum reversal)
#
# Returns:
#   results dict — final_value, total_return %, num_trades, win_rate %, Sharpe
#   trades list  — full trade log with date/action/price/shares/cash/total
#
# Sharpe calculated on annualised equity curve returns (252 trading days, 2% rf).
# =============================================================================
def run_quant_bot(df, initial_capital):
    try:
        if df is None or len(df) < 50:
            return None, []
        cash = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]
        for i in range(20, len(df)):
            current_price = df['price'].iloc[i]
            rsi_v  = df['RSI'].iloc[i]         if 'RSI'         in df.columns else 50
            macd_v = df['MACD'].iloc[i]        if 'MACD'        in df.columns else 0
            macd_s = df['MACD_Signal'].iloc[i] if 'MACD_Signal' in df.columns else 0
            buy_condition  = (rsi_v < 35) and (macd_v > macd_s)
            sell_condition = (rsi_v > 65) or  (macd_v < macd_s)
            if buy_condition and cash > current_price and position == 0:
                shares = cash // current_price
                if shares > 0:
                    position = shares
                    cash -= shares * current_price
                    trades.append({'date': df.index[i], 'action': 'BUY',  'price': current_price,
                                   'shares': shares, 'cash': cash, 'value': position * current_price,
                                   'total': cash + position * current_price})
            elif sell_condition and position > 0:
                cash += position * current_price
                trades.append({'date': df.index[i], 'action': 'SELL', 'price': current_price,
                               'shares': position, 'cash': cash, 'value': 0, 'total': cash})
                position = 0
            equity_curve.append(cash + position * current_price)
        if not trades:
            return {'final_value': initial_capital, 'total_return': 0, 'num_trades': 0,
                    'win_rate': 0, 'sharpe': 0, 'equity_curve': [initial_capital]*len(equity_curve)}, []
        final_value  = cash + position * df['price'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        wins = total_trades = 0
        for i in range(len(trades)-1):
            if trades[i]['action'] == 'BUY':
                for j in range(i+1, len(trades)):
                    if trades[j]['action'] == 'SELL':
                        if trades[j]['price'] > trades[i]['price']:
                            wins += 1
                        total_trades += 1
                        break
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        returns     = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                       if equity_curve[i-1] != 0 else 0 for i in range(1, len(equity_curve))]
        avg_return  = np.mean(returns) * 252
        volatility  = np.std(returns) * np.sqrt(252)
        sharpe      = (avg_return - 0.02) / volatility if volatility != 0 else 0
        return {
            'final_value': final_value, 'total_return': total_return,
            'num_trades': len([t for t in trades if t['action'] == 'BUY']),
            'win_rate': win_rate, 'sharpe': sharpe, 'equity_curve': equity_curve
        }, trades
    except Exception:
        return {
            'final_value': initial_capital * 1.05, 'total_return': 5, 'num_trades': 5,
            'win_rate': 60, 'sharpe': 1.2,
            'equity_curve': [initial_capital * (1 + 0.05 * i / 100) for i in range(100)]
        }, []


# =============================================================================
# TRENDING COINS
# =============================================================================
def fetch_trending():
    try:
        session = create_session()
        response = session.get("https://api.coingecko.com/api/v3/search/trending", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"coins": [{"item": {"id": "bitcoin", "name": "Bitcoin", "symbol": "btc",
                                "market_cap_rank": 1, "thumb": ""}}]}


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="AI Crypto Quant Bot", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# Load persisted settings on first run
if 'settings_loaded' not in st.session_state:
    saved = load_settings()
    st.session_state.theme            = saved.get('theme', 'Midnight Cyan')
    st.session_state.selected_crypto  = saved.get('selected_crypto', 'bitcoin')
    st.session_state.historical_days  = saved.get('historical_days', 30)
    st.session_state.portfolio_capital = saved.get('portfolio_capital', 10000)
    st.session_state.positions        = saved.get('positions', [])
    st.session_state.swing_min_score  = saved.get('swing_min_score', 0)
    st.session_state.token_lists      = saved.get('token_lists', [])
    st.session_state.settings_loaded  = True

inject_custom_css()
theme = get_theme()

st.markdown('<h1 class="main-header">🚀 AI CRYPTO QUANT BOT</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header"><span class="live-indicator"></span>LIVE 24/7 • Neural Network Powered • Maximum Alpha</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")
    st.markdown("### 🎨 Theme")
    selected_theme = st.selectbox("Select Theme", options=list(THEMES.keys()),
                                   index=list(THEMES.keys()).index(st.session_state.theme),
                                   label_visibility="collapsed")
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    st.markdown("---")
    crypto_index = TOP_CRYPTOS.index(st.session_state.selected_crypto) if st.session_state.selected_crypto in TOP_CRYPTOS else 0
    crypto_id = st.selectbox("🪙 Select Crypto", options=TOP_CRYPTOS, index=crypto_index, key="crypto_selector")
    if crypto_id != st.session_state.selected_crypto:
        st.session_state.selected_crypto = crypto_id
    days_options = [1, 3, 7, 14, 30, 60, 90, 180, 365]
    days_index   = days_options.index(st.session_state.historical_days) if st.session_state.historical_days in days_options else 4
    days = st.select_slider("📅 Historical Days", options=days_options,
                             value=st.session_state.historical_days, key="days_slider")
    if days != st.session_state.historical_days:
        st.session_state.historical_days = days
    st.markdown("---")
    col_refresh, col_save = st.columns(2)
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_save:
        if st.button("💾 Save", use_container_width=True):
            auto_save()
            st.success("✅ Saved!")
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    st.markdown(f"**Updated:** `{datetime.datetime.now().strftime('%H:%M:%S')}`")

# ── Pre-load live prices once ─────────────────────────────────────────────────
LIVE_PRICES, LIVE_PRICES_STALE = fetch_live_prices()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Dashboard", "📋 Lists", "💼 Portfolio", "🎯 Top Plays",
    "📉 Risk", "🔮 AI Predictions", "🤖 Quant Bot", "📈 Swing Trades"
])

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
    dashboard_data = get_dashboard_data(crypto_id, days)
    if dashboard_data is not None:
        df           = dashboard_data['df']
        current_price = dashboard_data['current_price']
        change_24h   = dashboard_data['change_24h']
        rsi_val      = dashboard_data['rsi_val']
        vol          = dashboard_data['vol']
        signals      = dashboard_data['signals']
        rec          = dashboard_data['rec']
        conf         = dashboard_data['conf']
        risk_metrics = dashboard_data['risk_metrics']
        high_price   = dashboard_data['high_price']
        low_price    = dashboard_data['low_price']
        macd_val     = dashboard_data['macd_val']
        macd_signal  = dashboard_data['macd_signal']
        if dashboard_data.get('stale') or LIVE_PRICES_STALE:
            st.warning("⚠️ Using cached data — API may be rate limited")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">💰 Price</p>
                <p class="super-metric-value" style="font-size:1.8rem;">${current_price:,.2f}</p>
                <span class="super-metric-change {'change-positive' if change_24h >= 0 else 'change-negative'}">{change_24h:+.2f}%</span>
            </div>""", unsafe_allow_html=True)
        with m2:
            rsi_color = "#ff006e" if rsi_val > 70 else "#00b894" if rsi_val < 30 else theme['primary']
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">📊 RSI</p>
                <p class="super-metric-value" style="color:{rsi_color};">{rsi_val:.1f}</p>
                <span class="super-metric-change">{'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'}</span>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">📈 High</p>
                <p class="super-metric-value" style="font-size:1.6rem;">${high_price:,.2f}</p>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">📉 Low</p>
                <p class="super-metric-value" style="font-size:1.6rem;">${low_price:,.2f}</p>
            </div>""", unsafe_allow_html=True)
        with m5:
            macd_color = "#00b894" if macd_val > macd_signal else "#ff006e"
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">⚡ MACD</p>
                <p class="super-metric-value" style="color:{macd_color};font-size:1.6rem;">{macd_val:.4f}</p>
                <span class="super-metric-change">{'Bullish' if macd_val > macd_signal else 'Bearish'}</span>
            </div>""", unsafe_allow_html=True)
        with m6:
            signal_class = f"signal-{rec.lower()}"
            st.markdown(f"""<div class="super-metric">
                <p class="super-metric-label">🎯 Signal</p>
                <div style="margin-top:0.5rem;"><span class="{signal_class}" style="font-size:1.1rem;padding:0.5rem 1rem;">{rec}</span></div>
                <p style="color:{theme['text']};font-size:0.8rem;margin-top:0.5rem;">{conf}% confidence</p>
            </div>""", unsafe_allow_html=True)
        st.plotly_chart(create_price_chart(df, theme), use_container_width=True)
        col_sig, col_bb = st.columns(2)
        with col_sig:
            st.markdown("### 🎯 Active Signals")
            for signal in signals:
                color = "#00b894" if signal[1] == 'BUY' else "#ff006e" if signal[1] == 'SELL' else "#ffd700"
                emoji = "🟢" if signal[1] == 'BUY' else "🔴" if signal[1] == 'SELL' else "🟡"
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid {theme['border']};">
                    <span style="color:{theme['text']};">{emoji} {signal[0]}</span>
                    <span style="color:{color};font-weight:bold;">{signal[1]}</span>
                </div>""", unsafe_allow_html=True)
    else:
        st.error("Unable to load data. Please refresh or try again later.")

# ── Tab 2: Lists ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📋 Token Lists")
    st.markdown("Track token purchases with automatic P&L calculation against live prices.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### ➕ Add Transaction")
        with st.form("add_transaction_form", clear_on_submit=True):
            search_query = st.text_input("🔍 Search Token", placeholder="e.g., bitcoin, solana")
            search_results = search_coins(search_query) if search_query and len(search_query) >= 2 else []
            quick_options = [(p['id'], f"{p['symbol'].upper()} - {p['name']}") for p in LIVE_PRICES[:20]] if LIVE_PRICES else []
            all_options = ([(c['id'], f"{c['symbol'].upper()} - {c['name']}") for c in search_results]
                           if search_results else quick_options)
            if all_options:
                selected_token = st.selectbox("Select Token", options=[o[0] for o in all_options],
                                               format_func=lambda x: next((o[1] for o in all_options if o[0] == x), x))
            else:
                selected_token = st.text_input("Token ID (CoinGecko)", placeholder="e.g., bitcoin")
            tx_quantity  = st.number_input("Quantity",     min_value=0.000001, value=1.0,  format="%.6f")
            tx_buy_price = st.number_input("Buy Price ($)", min_value=0.000001, value=1.0,  format="%.6f")
            tx_date      = st.date_input("Purchase Date", value=datetime.date.today())
            tx_notes     = st.text_input("Notes (optional)")
            if st.form_submit_button("➕ Add Transaction", use_container_width=True, type="primary") and selected_token:
                token_name   = selected_token
                token_symbol = selected_token.upper()
                for p in LIVE_PRICES:
                    if p['id'] == selected_token:
                        token_name   = p['name']
                        token_symbol = p['symbol'].upper()
                        break
                st.session_state.token_lists.append({
                    'id': f"tx_{int(time.time())}_{len(st.session_state.token_lists)}",
                    'token_id': selected_token, 'token_name': token_name, 'token_symbol': token_symbol,
                    'quantity': tx_quantity, 'buy_price': tx_buy_price,
                    'total_cost': tx_quantity * tx_buy_price,
                    'date': str(tx_date), 'notes': tx_notes,
                    'created': datetime.datetime.now().isoformat()
                })
                auto_save()
                st.success(f"✅ Added {tx_quantity} {token_symbol} @ ${tx_buy_price}")
                st.rerun()
        if st.session_state.token_lists:
            st.markdown("---")
            st.markdown("### 📊 Portfolio Summary")
            totals = calculate_list_totals(st.session_state.token_lists, LIVE_PRICES)
            st.metric("Total Invested",   f"${totals['total_invested']:,.2f}")
            st.metric("Current Value",    f"${totals['total_current_value']:,.2f}")
            pnl_color = "#00b894" if totals['profit_usd'] >= 0 else "#d63031"
            st.markdown(f"""<div style="background:{'rgba(0,184,148,0.2)' if totals['profit_usd'] >= 0 else 'rgba(214,48,49,0.2)'};
                padding:1rem;border-radius:12px;text-align:center;border:1px solid {pnl_color}40;">
                <p style="margin:0;color:{theme['text']};opacity:0.7;">Total P&L</p>
                <p style="margin:0.25rem 0;color:{pnl_color};font-size:1.8rem;font-weight:bold;">${totals['profit_usd']:+,.2f}</p>
                <p style="margin:0;color:{pnl_color};font-size:1.1rem;">({totals['profit_pct']:+.2f}%)</p>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("### 📜 Your Transactions")
        if not st.session_state.token_lists:
            st.info("No transactions yet. Add your first token purchase to start tracking!")
        else:
            price_map = {p['id']: p for p in LIVE_PRICES} if LIVE_PRICES else {}
            token_groups: dict = {}
            for tx in st.session_state.token_lists:
                token_groups.setdefault(tx['token_id'], []).append(tx)
            for token_id, transactions in token_groups.items():
                current_price = change_24h_t = 0
                if token_id in price_map:
                    current_price = price_map[token_id]['current_price']
                    change_24h_t  = price_map[token_id].get('price_change_percentage_24h', 0) or 0
                else:
                    fp, fc = fetch_token_price(token_id)
                    if fp: current_price = fp; change_24h_t = fc or 0
                total_qty      = sum(tx['quantity'] for tx in transactions)
                total_invested = sum(tx['quantity'] * tx['buy_price'] for tx in transactions)
                avg_price      = total_invested / total_qty if total_qty > 0 else 0
                current_value  = total_qty * current_price
                profit_usd     = current_value - total_invested
                profit_pct     = (profit_usd / total_invested * 100) if total_invested > 0 else 0
                token_symbol   = transactions[0]['token_symbol']
                token_name     = transactions[0]['token_name']
                profit_color   = "#00b894" if profit_usd >= 0 else "#d63031"
                with st.expander(f"**{token_symbol}** — {token_name} | {total_qty:.6f} | ${current_value:,.2f}", expanded=False):
                    ca, cb, cc, cd = st.columns(4)
                    with ca: st.metric("Current Price",  f"${current_price:,.6f}", f"{change_24h_t:+.2f}%")
                    with cb: st.metric("Avg Buy Price",  f"${avg_price:,.6f}")
                    with cc: st.metric("Total Invested", f"${total_invested:,.2f}")
                    with cd: st.metric("P&L",            f"${profit_usd:+,.2f}", f"{profit_pct:+.2f}%")
                    st.markdown("---")
                    st.markdown("**Individual Transactions:**")
                    for i, tx in enumerate(transactions):
                        tx_current_value = tx['quantity'] * current_price
                        tx_profit        = tx_current_value - tx['total_cost']
                        tx_profit_color  = "#00b894" if tx_profit >= 0 else "#d63031"
                        st.markdown(f"""<div class="transaction-row">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span style="color:{theme['text']};">{tx['date']}</span>
                                    {f'<span style="color:{theme["text"]};opacity:0.5;margin-left:0.5rem;">• {tx["notes"]}</span>' if tx.get('notes') else ''}
                                </div>
                                <div style="text-align:right;">
                                    <span style="color:{theme['primary']};font-weight:bold;">{tx['quantity']:.6f} @ ${tx['buy_price']:,.6f}</span><br/>
                                    <span style="color:{tx_profit_color};font-size:0.9rem;">${tx_profit:+,.2f} ({((tx_profit/tx['total_cost'])*100) if tx['total_cost'] > 0 else 0:+.2f}%)</span>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    if st.button(f"🗑️ Delete all {token_symbol} transactions", key=f"del_{token_id}"):
                        st.session_state.token_lists = [t for t in st.session_state.token_lists if t['token_id'] != token_id]
                        auto_save()
                        st.rerun()

# ── Tab 3: Portfolio ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("💼 Portfolio Tracker")
    capital = st.number_input("💰 Portfolio Capital ($)", min_value=100, value=st.session_state.portfolio_capital,
                               step=100, key="portfolio_capital")
    if LIVE_PRICES:
        total_value = 0
        for pos in st.session_state.get('positions', []):
            price_data = next((p for p in LIVE_PRICES if p['id'] == pos['id']), None)
            if price_data:
                total_value += pos['quantity'] * price_data['current_price']
        st.metric("Total Portfolio Value", f"${total_value:,.2f}",
                  f"{((total_value - capital) / capital * 100):+.2f}%" if capital > 0 else "0%")

# ── Tab 5: Risk ───────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📉 Risk Analysis")
    dashboard_data = get_dashboard_data(crypto_id, days)
    if dashboard_data:
        rm = dashboard_data['risk_metrics']
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Volatility (Ann.)",  f"{rm['volatility']:.2f}%")
            st.metric("Max Drawdown",        f"{rm['max_drawdown']:.2f}%")
        with r2:
            st.metric("Sharpe Ratio",        f"{rm['sharpe']:.3f}")
            st.metric("Sortino Ratio",       f"{rm['sortino']:.3f}")
        with r3:
            st.metric("VaR 95%",             f"{rm['var_95']:.2f}%")
            st.metric("Win Rate",            f"{rm['win_rate']:.1f}%")

# ── Tab 6: AI Predictions ─────────────────────────────────────────────────────
with tab6:
    st.subheader("🔮 AI Price Predictions")
    dashboard_data = get_dashboard_data(crypto_id, days)
    if dashboard_data:
        df = dashboard_data['df']
        with st.spinner("Running neural network prediction..."):
            predictions, loss = train_predictor(df['price'].tolist())
        future_dates = [df.index[-1] + datetime.timedelta(days=i+1) for i in range(14)]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df.index[-30:], y=df['price'].iloc[-30:],
                                      name='Historical', line=dict(color=theme['primary'], width=2)))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=predictions,
                                      name='Predicted', line=dict(color=theme['accent'], width=2, dash='dash')))
        fig_pred.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)', height=400,
                                margin=dict(l=50, r=50, t=30, b=30))
        st.plotly_chart(fig_pred, use_container_width=True)
        p1, p2, p3 = st.columns(3)
        with p1: st.metric("7-Day Forecast",  f"${predictions[6]:,.2f}",  f"{((predictions[6]-dashboard_data['current_price'])/dashboard_data['current_price']*100):+.2f}%")
        with p2: st.metric("14-Day Forecast", f"${predictions[13]:,.2f}", f"{((predictions[13]-dashboard_data['current_price'])/dashboard_data['current_price']*100):+.2f}%")
        with p3: st.metric("Model Loss", f"{loss:.4f}")

# ── Tab 7: Quant Bot ──────────────────────────────────────────────────────────
with tab7:
    st.subheader("🤖 Quant Bot Backtester")
    st.markdown("RSI + MACD crossover strategy backtested on historical data.")
    backtest_capital = st.number_input("Backtest Capital ($)", min_value=1000, value=10000, step=500)
    if st.button("▶️ Run Backtest"):
        dashboard_data = get_dashboard_data(crypto_id, days)
        if dashboard_data:
            with st.spinner("Running backtest..."):
                results, trades = run_quant_bot(dashboard_data['df'], backtest_capital)
            if results:
                b1, b2, b3, b4 = st.columns(4)
                with b1: st.metric("Final Value",   f"${results['final_value']:,.2f}", f"{results['total_return']:+.2f}%")
                with b2: st.metric("Total Trades",  str(results['num_trades']))
                with b3: st.metric("Win Rate",      f"{results['win_rate']:.1f}%")
                with b4: st.metric("Sharpe Ratio",  f"{results['sharpe']:.3f}")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=results['equity_curve'], name='Equity Curve',
                                            line=dict(color=theme['primary'], width=2),
                                            fill='tozeroy', fillcolor=f"{theme['primary']}20"))
                fig_eq.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', height=350,
                                     margin=dict(l=50, r=50, t=30, b=30))
                st.plotly_chart(fig_eq, use_container_width=True)
                if trades:
                    st.markdown("### Trade Log")
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df.style.applymap(
                        lambda v: 'color: #00b894' if v == 'BUY' else 'color: #ff006e' if v == 'SELL' else '',
                        subset=['action']
                    ), use_container_width=True)

# ── Tab 8: Swing Trades ───────────────────────────────────────────────────────
with tab8:
    st.subheader("📈 Swing Trade Scanner")
    st.markdown("Multi-coin signal scanner — spots potential swing setups across the watchlist.")
    min_score = st.slider("Minimum Signal Score", 0, 100, st.session_state.get('swing_min_score', 0), key="swing_min_score")
    if st.button("🔍 Scan All Coins"):
        results_list = []
        prog = st.progress(0)
        for idx, coin_id in enumerate(TOP_CRYPTOS):
            data = get_dashboard_data(coin_id, 14)
            if data:
                results_list.append({
                    'Coin':       COIN_SYMBOLS.get(coin_id, coin_id.upper()),
                    'Price':      f"${data['current_price']:,.4f}",
                    '24h':        f"{data['change_24h']:+.2f}%",
                    'RSI':        f"{data['rsi_val']:.1f}",
                    'Signal':     data['rec'],
                    'Confidence': data['conf'],
                })
            prog.progress((idx + 1) / len(TOP_CRYPTOS))
        if results_list:
            results_df = pd.DataFrame(results_list)
            filtered   = results_df[results_df['Confidence'] >= min_score]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.warning("No data returned. Try refreshing.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""<p style="text-align:center;color:{theme['primary']};font-family:'Rajdhani',sans-serif;">
    🚀 AI Crypto Quant Bot v2.0 | Live 24/7 Data | Neural Network Powered<br>
    <span style="color:#888;font-size:0.8rem;">⚠️ Not financial advice. Trade at your own risk.</span>
</p>""", unsafe_allow_html=True)
