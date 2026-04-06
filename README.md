# AI Crypto Quant Bot

> A full-stack crypto analysis and backtesting dashboard built with Streamlit, PyTorch, and the CoinGecko API. Live technical indicators, neural network price forecasting, quant strategy backtesting, and multi-coin portfolio tracking — all running locally with no API key required.

Built by [@blockchainbail](https://x.com/blockchainbail) · Part of the [Oden Network](https://odennetworkxr.com) builder stack.

---

Eight tabs. One file. Zero fluff.

Live RSI, MACD, Bollinger Bands, and Golden/Death Cross signals scored and weighted into a single BUY / SELL / HOLD recommendation with confidence percentage. PyTorch momentum model projects 14-day price forecasts on CUDA or CPU. A full quant backtester runs an RSI + MACD crossover strategy against historical data and returns an equity curve, trade log, win rate, and annualised Sharpe ratio. Multi-coin swing scanner checks the entire watchlist in parallel. Token purchase tracker calculates live P&L per position against real-time CoinGecko prices.

All API calls run through a single cached session with retry logic and disk pickle fallback — the dashboard stays functional even when rate-limited. Four switchable themes with full CSS injection. Settings persist across sessions via `settings.json`.

---

⚠️ Not financial advice. For educational and research purposes only.
