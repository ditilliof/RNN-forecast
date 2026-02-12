# RNN Trade Forecast

Production-grade forecasting system for cryptocurrencies and ETFs using a deterministic recurrent neural network regressor with Huber loss and residual-based prediction intervals.

## Algorithm Overview

This system uses a **deterministic RNN regressor** that differs from probabilistic approaches like DeepAR:

- **Architecture**: LSTM/GRU encoder with linear output head for point predictions
- **Loss Function**: Huber loss (robust to outliers, combines L1 and L2)
- **Training**: Teacher forcing on past + future context → deterministic regression
- **Inference**: Autoregressive prediction feeding its own outputs forward
- **Uncertainty Quantification**: Post-hoc prediction intervals from validation residual statistics (not learned distributions)

Unlike probabilistic models (Student-t, Gaussian), this approach:
- Produces single point forecasts per timestep
- Uses robust regression loss instead of likelihood maximization
- Derives confidence intervals from empirical residuals rather than parametric distributions
- Provides computational efficiency and interpretability

## Architecture

```
src/rnn_forecast/
├── data/           # Data providers (CCXT, yfinance) + SQLite storage
├── features/       # Feature engineering (log-returns, indicators, no leakage)
├── models/         # RNN regressor with Huber loss, training loop
├── evaluation/     # Metrics (MAE, RMSE, coverage, calibration)
├── backtest/       # Walk-forward backtesting with transaction costs
├── app_api/        # FastAPI REST endpoints (/ingest, /train, /forecast, /backtest)
└── app_ui/         # Streamlit dashboards (main forecast + backtest mode)
```

## Quick Start

### 1. Install

```bash
poetry install
cp .env.example .env          # optional, defaults work out of the box
```

### 2. Start the API server

```bash
poetry run uvicorn rnn_forecast.app_api.main:app --host 0.0.0.0 --port 8000
```

API docs: <http://localhost:8000/docs>

### 3. Start the main Streamlit UI

```bash
poetry run streamlit run src/rnn_forecast/app_ui/main.py --server.port 8501
```

Open <http://localhost:8501> — follow the 3-step wizard (select asset → configure → run).

### 4. Start the backtest UI (optional)

```bash
poetry run streamlit run src/rnn_forecast/app_ui/main_backtest.py --server.port 8502
```

Open <http://localhost:8502> — train with a cutoff date and compare forecast vs actual.
