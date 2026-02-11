"""
RNN Trade Forecast

Production-grade forecasting system for cryptocurrencies and ETFs using
a deterministic recurrent neural network regressor with Huber loss.
"""

__version__ = "0.2.0"

from . import backtest, data, evaluation, features, models

__all__ = ["data", "features", "models", "evaluation", "backtest"]
