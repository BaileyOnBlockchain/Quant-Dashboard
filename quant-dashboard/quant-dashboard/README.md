# AI Crypto Quant Bot

> AI-powered crypto analysis, backtesting, and portfolio tracking dashboard.  
> Built with Streamlit + PyTorch. Live CoinGecko data, full technical indicator suite, neural network price prediction, and a quant backtester — all in one local tool.

**Built by [@blockchainbail](https://x.com/blockchainbail)**  
**Part of the [Oden Network](https://odennetworkxr.com) builder stack.**

---

## What it does

| Tab | Features |
|---|---|
| 📊 Dashboard | Live price, RSI, MACD, Bollinger Bands, signal strength, 3-panel Plotly chart |
| 📋 Lists | Token purchase tracker with live P&L, grouped by coin, CoinGecko search |
| 💼 Portfolio | Capital allocation and position tracking |
| 🎯 Top Plays | Live market movers and opportunity scanner |
| 📉 Risk | Volatility, Sharpe, Sortino, Max Drawdown, VaR 95%, Win Rate |
| 🔮 AI Predictions | PyTorch 14-day price forecast with trend momentum model |
| 🤖 Quant Bot | RSI + MACD strategy backtester — equity curve, trade log, Sharpe |
| 📈 Swing Trades | Multi-coin signal scanner across the full watchlist |

---

## Stack

`Python 3.13` · `Streamlit` · `PyTorch` · `pandas-ta` · `Plotly` · `CoinGecko API`

---

## Run it

```bash
pip install streamlit torch pandas pandas-ta plotly requests
streamlit run main.py
```

Runs locally at `http://localhost:8501`

---

## Key engineering

**Caching strategy**  
All API calls use a single `@st.cache_resource` requests session with retry logic. Historical data and live prices have layered TTL caching (`@st.cache_data`) + disk pickle fallback — so the dashboard stays functional when CoinGecko rate-limits.

**Technical indicators**  
Computed via `pandas-ta` with length-adaptive logic (RSI/BB window scales to dataset size so short timeframes don't crash). All indicator computation is cached by `(price_tuple, volume_tuple)` hash.

**Signal engine**  
Weighted scoring across 4 indicator families — RSI, MACD crossover, Golden/Death Cross, Bollinger Band breach. Confidence % derived from buy/sell weight ratio.

**Neural network predictor**  
PyTorch tensor operations. Computes 7-day vs 14-day momentum trend, projects 14 daily prices with Gaussian noise. Runs on CUDA if available.

**Quant backtester**  
RSI < 35 + MACD bullish → BUY. RSI > 65 or MACD bearish → SELL. Full equity curve tracked, Sharpe ratio computed on annualised returns (252-day, 2% risk-free rate).

**Theme system**  
4 themes (Midnight Cyan, Deep Purple, Neon Matrix, Golden Alpha). Single cached CSS string injected via `st.markdown`. All chart colours and HTML derive from the active theme object — no hardcoded colours outside the theme config.

**Settings persistence**  
All session state (theme, coin, timeframe, positions, token lists) saved to `settings.json` and restored on next launch.

---

## Coins supported

BTC · ETH · SOL · ADA · AVAX · DOT · LINK · DOGE · XRP · MATIC

---

## Notes

- No API key required — uses CoinGecko free tier
- CUDA auto-detected for PyTorch; falls back to CPU
- `.cache/` directory created automatically for disk fallback
- Settings auto-saved on manual save or position change

---

⚠️ **Not financial advice. For educational and research purposes only.**
