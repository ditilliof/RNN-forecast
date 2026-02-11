"""
Deterministic RNN Regressor for financial time series forecasting.

Replaces the probabilistic DeepAR (Student-t) approach with a standard
regression model that outputs point forecasts and uses Huber loss.

Prediction intervals are generated post-hoc from validation residual
statistics rather than from a learned distribution.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class RNNRegressor(nn.Module):
    """
    Deterministic RNN regressor for financial time series.

    Architecture:
    - Embeddings for symbol, timeframe, asset_type (categorical features)
    - LSTM/GRU encoder for autoregressive context encoding
    - Linear output head producing a single point prediction per timestep

    Training:
    - Teacher forcing on past + future context → Huber regression loss
    - No distributional output — deterministic predictions only

    Inference:
    - Autoregressive: feed own predictions back as input for each horizon step
    - Prediction intervals derived from stored residual std (not model output)
    """

    def __init__(
        self,
        input_size: int,  # Number of exogenous features
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "lstm",  # 'lstm' or 'gru'
        embedding_dim: int = 8,
        num_symbols: int = 10,
        num_timeframes: int = 5,
        num_asset_types: int = 3,
    ):
        """
        Initialize RNN Regressor.

        Args:
            input_size: Number of exogenous features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
            embedding_dim: Dimension of categorical embeddings
            num_symbols: Number of unique symbols for embedding
            num_timeframes: Number of unique timeframes
            num_asset_types: Number of unique asset types
        """
        super().__init__()

        # GUARD: input_size must NEVER be 0 — clamp to 1 minimum
        if input_size < 1:
            logger.warning(
                f"RNNRegressor.__init__ received input_size={input_size}, clamping to 1"
            )
            input_size = 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Embeddings for categorical features
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.timeframe_embedding = nn.Embedding(num_timeframes, embedding_dim)
        self.asset_type_embedding = nn.Embedding(num_asset_types, embedding_dim)

        # RNN encoder
        # Input: target(1) + exogenous features + 3 embeddings
        rnn_input_size = 1 + input_size + 3 * embedding_dim

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        # Deterministic output head — single value per timestep
        self.output_layer = nn.Linear(hidden_size, 1)

        logger.info(
            f"Initialized RNNRegressor: {rnn_type.upper()} "
            f"hidden={hidden_size}, layers={num_layers}, input={input_size}"
        )

    # ------------------------------------------------------------------ #
    #  Embedding helper (shared by forward / predict)
    # ------------------------------------------------------------------ #

    def _get_embeddings(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        symbol_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        asset_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (symbol_emb, timeframe_emb, asset_type_emb) each (B, T, E)."""
        if symbol_ids is None:
            symbol_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if timeframe_ids is None:
            timeframe_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if asset_type_ids is None:
            asset_type_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        sym = self.symbol_embedding(symbol_ids).unsqueeze(1).expand(-1, seq_len, -1)
        tf = self.timeframe_embedding(timeframe_ids).unsqueeze(1).expand(-1, seq_len, -1)
        at = self.asset_type_embedding(asset_type_ids).unsqueeze(1).expand(-1, seq_len, -1)
        return sym, tf, at

    # ------------------------------------------------------------------ #
    #  Forward (training with teacher forcing)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        past_target: torch.Tensor,      # (batch, context_length, 1)
        past_features: torch.Tensor,     # (batch, context_length, input_size)
        future_target: Optional[torch.Tensor] = None,  # (batch, horizon, 1)
        symbol_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        asset_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Returns:
            predictions: (batch, context_length + horizon, 1) when future_target
                         is provided, else (batch, context_length, 1).
        """
        batch_size = past_target.size(0)
        context_length = past_target.size(1)
        device = past_target.device

        sym, tf, at = self._get_embeddings(
            batch_size, context_length, device,
            symbol_ids, timeframe_ids, asset_type_ids,
        )

        # Context RNN input
        rnn_input = torch.cat([past_target, past_features, sym, tf, at], dim=-1)
        rnn_output, hidden = self.rnn(rnn_input)

        if future_target is not None:
            horizon = future_target.size(1)
            sym_f, tf_f, at_f = self._get_embeddings(
                batch_size, horizon, device,
                symbol_ids, timeframe_ids, asset_type_ids,
            )
            future_features = torch.zeros(
                batch_size, horizon, self.input_size, device=device,
            )
            future_input = torch.cat(
                [future_target, future_features, sym_f, tf_f, at_f], dim=-1,
            )
            future_output, _ = self.rnn(future_input, hidden)
            full_output = torch.cat([rnn_output, future_output], dim=1)
        else:
            full_output = rnn_output

        # Linear projection → point prediction per timestep
        predictions = self.output_layer(full_output)  # (B, T, 1)
        return predictions

    # ------------------------------------------------------------------ #
    #  Deterministic autoregressive prediction (inference)
    # ------------------------------------------------------------------ #

    def predict(
        self,
        past_target: torch.Tensor,      # (batch, context_length, 1)
        past_features: torch.Tensor,     # (batch, context_length, input_size)
        horizon: int,
        symbol_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        asset_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deterministic autoregressive forecast.

        Args:
            past_target: (batch, context_length, 1) past log-returns
            past_features: (batch, context_length, input_size)
            horizon: number of future steps

        Returns:
            forecast: (batch, horizon, 1) predicted log-returns
        """
        self.eval()
        with torch.no_grad():
            past_target = past_target.to(dtype=torch.float32)
            past_features = past_features.to(dtype=torch.float32)

            batch_size = past_target.size(0)
            context_length = past_target.size(1)
            device = past_target.device

            # Defensive: feature dim must be >= 1
            safe_input = max(self.input_size, 1)
            if past_features.size(-1) != safe_input:
                past_features = torch.zeros(
                    batch_size, context_length, safe_input,
                    device=device, dtype=torch.float32,
                )

            sym, tf, at = self._get_embeddings(
                batch_size, context_length, device,
                symbol_ids, timeframe_ids, asset_type_ids,
            )

            rnn_input = torch.cat([past_target, past_features, sym, tf, at], dim=-1)
            _, hidden = self.rnn(rnn_input)

            # Autoregressive roll-forward
            preds = []
            current = past_target[:, -1:, :]  # last observed target

            for _ in range(horizon):
                sym_t, tf_t, at_t = self._get_embeddings(
                    batch_size, 1, device,
                    symbol_ids, timeframe_ids, asset_type_ids,
                )
                feat_t = torch.zeros(
                    batch_size, 1, self.input_size, device=device, dtype=torch.float32,
                )
                inp_t = torch.cat([current, feat_t, sym_t, tf_t, at_t], dim=-1)
                out_t, hidden = self.rnn(inp_t, hidden)
                pred_t = self.output_layer(out_t)           # (B, 1, 1)
                pred_t = torch.clamp(pred_t, -1.0, 1.0)    # safety clamp
                preds.append(pred_t)
                current = pred_t                             # feed own prediction

            return torch.cat(preds, dim=1)  # (batch, horizon, 1)
