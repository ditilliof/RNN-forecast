# DeepAR Forecasting — Machine Learning Manual

> Local reference document. Not committed to version control.
> This manual is intended as a comprehensive, self-contained technical reference for
> the DeepAR Trade Forecast system. It is written at a graduate level for readers
> who possess working knowledge of calculus and linear algebra but have never
> examined this particular codebase before.

---

## Table of Contents

1. [Chapter 1 — Theory](#chapter-1--theory)
2. [Chapter 2 — Code Walk-through (Line-by-Line)](#chapter-2--code-walk-through-line-by-line)
3. [Chapter 3 — Feature Engineering Theory](#chapter-3--feature-engineering-theory)
4. [Chapter 4 — Financial Forecasting Interpretation](#chapter-4--financial-forecasting-interpretation)
5. [Chapter 5 — Diagnostics, Failure Modes, and Limitations](#chapter-5--diagnostics-failure-modes-and-limitations)

---

## Chapter 1 — Theory

This chapter lays the full mathematical and conceptual groundwork for every modelling
decision in the system. It begins from first principles — the definition of a time
series and the notion of stochastic processes — and advances through recurrent neural
networks, the LSTM cell, the DeepAR architecture, the Student's t likelihood, and
the negative-log-likelihood loss. No symbol is introduced without an immediate
textual explanation.

### 1.1 Time Series — Formal Definition

A **time series** is a sequence of observations indexed by time:

$$
\{y_t\}_{t=1}^{T} = (y_1, y_2, \ldots, y_T)
$$

where each $y_t \in \mathbb{R}$ (univariate) or $y_t \in \mathbb{R}^d$ (multivariate).
The index $t$ may represent equally spaced clock times (e.g. every hour) or irregular
timestamps. In this project every series is resampled to a regular grid by the data
ingestion layer, so we treat $t$ as an integer index with constant spacing $\Delta t$
(one hour, four hours, or one day depending on the chosen timeframe).

A time series differs from an ordinary sequence in one critical respect: the ordering
is meaningful and irrevocable. Permuting the elements of a time series destroys its
information content because temporal dependencies — the relationships between $y_t$
and $y_{t-1}, y_{t-2}, \ldots$ — are the very thing we wish to learn.

### 1.2 Stochastic Processes

A **stochastic process** is a collection of random variables defined on a common
probability space $(\Omega, \mathcal{F}, P)$ and indexed by a parameter set $\mathcal{T}$
(usually the real line or the integers):

$$
\{Y_t : t \in \mathcal{T}\}
$$

Each $Y_t$ is a random variable; an observed time series $(y_1, \ldots, y_T)$ is a
single **realisation** of this process. When we write a model such as $Y_t = f(Y_{t-1},
Y_{t-2}, \ldots) + \varepsilon_t$, we are specifying the joint distribution
$P(Y_1, Y_2, \ldots, Y_T)$ through the conditional distributions
$P(Y_t \mid Y_{<t})$. The DeepAR model parameterises exactly these conditionals
using a neural network.

### 1.3 Stationarity and Weak Stationarity

A stochastic process is **strictly stationary** if its finite-dimensional
distributions are invariant under time shifts. Formally, for every $k \geq 1$ and
for any indices $t_1 < t_2 < \cdots < t_k$:

$$
P(Y_{t_1}, Y_{t_2}, \ldots, Y_{t_k}) = P(Y_{t_1 + \tau}, Y_{t_2 + \tau}, \ldots, Y_{t_k + \tau})
\quad \forall \tau \in \mathbb{Z}
$$

This is an extremely strong condition. In practice the weaker notion of **weak
stationarity** (also called second-order or covariance stationarity) suffices:

1. The mean function $\mu(t) = E[Y_t]$ is constant: $\mu(t) = \mu$ for all $t$.
2. The variance $\text{Var}(Y_t) = \sigma^2$ is finite and constant.
3. The autocovariance function $\gamma(h) = \text{Cov}(Y_t, Y_{t+h})$ depends only
   on the lag $h$, not on $t$.

Financial prices $P_t$ of assets like BTC/USDT or SPY are non-stationary: they
exhibit trends (upward drift), changing volatility (heteroscedasticity), and the mean
$E[P_t]$ drifts over time. This is why we do not model prices directly. Instead, we
transform them into log-returns, which are approximately weakly stationary. The
derivation and justification are given in Section 1.5.

### 1.4 Autoregressive Models

An **autoregressive model of order $p$** (denoted AR($p$)) expresses the current
value as a linear combination of its $p$ most recent predecessors plus a noise term:

$$
Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \varepsilon_t
$$

where $c$ is a constant intercept, $\phi_1, \ldots, \phi_p$ are the autoregressive
coefficients, and $\varepsilon_t$ is a zero-mean white-noise error term with
$\text{Var}(\varepsilon_t) = \sigma^2_\varepsilon$. The term "autoregressive" means
the variable is regressed on its own past values.

Classical AR models are linear and assume fixed coefficients. The DeepAR model
generalises this idea in three ways:

1. The mapping from past to future is nonlinear, computed by a recurrent neural
   network.
2. The "order" is effectively the entire context window — the model sees $p$
   previous timesteps through the LSTM hidden state, which in principle can
   retain information from any of them.
3. The noise distribution is not fixed; its parameters $(\mu, \sigma, \nu)$ are
   predicted at each step, allowing heteroscedastic, heavy-tailed output.

### 1.5 Log-Returns — Derivation and Properties

#### 1.5.1 Definition

Given a sequence of asset prices $(P_0, P_1, \ldots, P_T)$, the **simple return** at
time $t$ is:

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

The **log-return** (or continuously compounded return) is:

$$
r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right) = \ln P_t - \ln P_{t-1}
$$

Equivalently, $r_t = \ln(1 + R_t)$. For small returns
$|R_t| \ll 1$, the Taylor expansion $\ln(1+x) \approx x$ gives $r_t \approx R_t$.
For typical hourly or daily returns (of order $10^{-3}$ to $10^{-2}$), the
approximation is excellent. For large moves (e.g. a 10 % crash), the distinction
matters: $R_t = -0.10$ corresponds to $r_t \approx -0.1054$.

#### 1.5.2 Additivity

Log-returns are additive over time. The cumulative return from $t$ to $t+k$ equals
the sum of individual log-returns:

$$
\ln\!\left(\frac{P_{t+k}}{P_t}\right)
= \sum_{i=1}^{k} \ln\!\left(\frac{P_{t+i}}{P_{t+i-1}}\right)
= \sum_{i=1}^{k} r_{t+i}
$$

This is a direct consequence of the logarithm transforming multiplication into
addition: $\ln(ab) = \ln a + \ln b$. The price ratio telescopes:

$$
\frac{P_{t+k}}{P_t} = \frac{P_{t+1}}{P_t} \cdot \frac{P_{t+2}}{P_{t+1}} \cdots \frac{P_{t+k}}{P_{t+k-1}}
$$

Taking logarithms converts each factor into the corresponding log-return term.

Additivity is crucial for the DeepAR model because the LSTM produces one log-return
prediction per timestep, and the cumulative sum of those predictions directly gives
the cumulative price change via exponentiation. Without additivity, accumulating
multi-step forecasts would require multiplicative corrections that interact
nonlinearly with the prediction uncertainty.

#### 1.5.3 Approximate Stationarity

Asset prices are non-stationary: they drift upward (equities historically earn a
positive long-run return), and their variance changes over time. The log-return
transformation removes the price level, leaving a series whose mean is approximately
zero (on short horizons) and whose variance, while time-varying, is far more stable
than that of prices.

More precisely, if prices follow a geometric Brownian motion
$dP = \mu P\,dt + \sigma P\,dW$, then the discrete log-return over interval $\Delta t$
is:

$$
r_t = \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t}\, Z_t
$$

where $Z_t \sim \mathcal{N}(0,1)$. This is a constant-mean, constant-variance
process — weakly stationary by construction. In reality, both $\mu$ and $\sigma$
vary slowly, so log-returns are only approximately stationary. The Student's t
likelihood used in this project absorbs some of the non-Gaussianity (heavy tails,
excess kurtosis) that a Gaussian model fails to capture.

#### 1.5.4 Why Not Simple Returns?

Simple returns are not additive: the two-period return
$R_{t:t+2} \neq R_{t+1} + R_{t+2}$. Instead, $1 + R_{t:t+2} = (1+R_{t+1})(1+R_{t+2})$.
This multiplicative structure would require the model to output productisable
quantities, complicating both the loss function and the aggregation of multi-step
forecasts. Log-returns avoid this entirely.

### 1.6 Recurrent Neural Networks (RNNs)

A **recurrent neural network** processes sequential input by maintaining a hidden
state $h_t$ that is updated at each timestep:

$$
h_t = f(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)
$$

where $W_{hh}$ and $W_{xh}$ are weight matrices, $b_h$ is a bias vector, and $f$ is
a nonlinear activation (e.g. $\tanh$). The hidden state $h_t$ serves as a compressed
summary of the input sequence up to time $t$. An output at time $t$ is then:

$$
o_t = g(W_{ho}\, h_t + b_o)
$$

Unrolling the recurrence over $T$ time steps yields a very deep computation graph
with shared weights. Backpropagation through this graph is called
**Backpropagation Through Time (BPTT)**. The gradient of the loss with respect to
parameters at earlier timesteps involves products of Jacobians
$\frac{\partial h_t}{\partial h_{t-1}}$. For a vanilla RNN with $\tanh$ activation,
these Jacobians can have spectral radius less than 1 (causing **vanishing gradients**)
or greater than 1 (causing **exploding gradients**). Vanishing gradients make it
difficult to learn long-range dependencies, because the error signal decays
exponentially as it propagates backward through many timesteps.

### 1.7 Long Short-Term Memory (LSTM) — Full Internal Mechanics

The LSTM (Hochreiter and Schmidhuber, 1997) solves the vanishing gradient problem by
introducing a **cell state** $c_t$ that flows through the network with controlled
additive updates, and three **gates** that regulate the flow of information.

#### 1.7.1 The Gates

At each timestep $t$, the LSTM receives the current input $x_t$ and the previous
hidden state $h_{t-1}$. Three gates are computed:

**Forget gate** — decides what fraction of the old cell state to retain:

$$
f_t = \sigma(W_f [h_{t-1},\, x_t] + b_f)
$$

$\sigma$ is the logistic sigmoid function, so $f_t \in (0, 1)^d$. The notation
$[h_{t-1}, x_t]$ denotes concatenation. $W_f$ has shape $(d_h, d_h + d_x)$ where
$d_h$ is the hidden size and $d_x$ is the input size. Each element of $f_t$ is a
per-dimension gate value: when $f_t^{(j)} \approx 1$, dimension $j$ of the old cell
state is retained; when $f_t^{(j)} \approx 0$, it is erased.

**Input gate** — decides which new values to write into the cell state:

$$
i_t = \sigma(W_i [h_{t-1},\, x_t] + b_i)
$$

**Candidate update** — proposes new content:

$$
\tilde{c}_t = \tanh(W_c [h_{t-1},\, x_t] + b_c)
$$

The $\tanh$ activation bounds the candidate to $(-1, 1)$, preventing unbounded
growth.

#### 1.7.2 Cell State Update

The cell state is updated by forgetting a fraction of the old state and adding a
fraction of the candidate:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

where $\odot$ is element-wise (Hadamard) multiplication. This is the critical design
feature of the LSTM: the cell state update is **additive** (plus a multiplicative
forget). When $f_t = 1$ and $i_t = 0$, the cell state passes through unchanged.
Gradients can therefore flow backward through arbitrary many timesteps along the cell
state "highway" without multiplicative attenuation — this is how the LSTM mitigates
vanishing gradients.

#### 1.7.3 Output Gate and Hidden State

**Output gate** — decides what portion of the cell state to expose:

$$
o_t = \sigma(W_o [h_{t-1},\, x_t] + b_o)
$$

**Hidden state** — the actual output of the LSTM cell at time $t$:

$$
h_t = o_t \odot \tanh(c_t)
$$

The $\tanh$ squashes the cell state to $(-1, 1)$, and the output gate modulates
which dimensions are visible. The hidden state $h_t$ is both the output that
downstream layers consume and the recurrent input for the next timestep.

#### 1.7.4 Parameter Counting in PyTorch's LSTM

In PyTorch, the four weight matrices ($W_f, W_i, W_c, W_o$ for the input-to-hidden
projection, and their hidden-to-hidden counterparts) are concatenated into two
matrices:

- `weight_ih_l{k}` of shape $(4 \cdot d_h,\, d_x)$ for layer $k$ (or $(4 \cdot d_h,\, d_h)$ for $k > 0$)
- `weight_hh_l{k}` of shape $(4 \cdot d_h,\, d_h)$

The factor 4 arises because all four gate computations share a single matrix-multiply
and the result is split into four equal chunks. The corresponding biases are
`bias_ih_l{k}` and `bias_hh_l{k}`, each of shape $(4 \cdot d_h)$.

For this project's default configuration ($d_h = 64$, $d_x = 1 + \text{input\_size} + 3 \times 8 = \text{rnn\_input\_size}$), `weight_ih_l0` has shape $(256, \text{rnn\_input\_size})$. The second stacked layer's `weight_ih_l1` has shape $(256, 64)$ because its input is the hidden state of layer 0.

#### 1.7.5 Stacked LSTMs

When `num_layers > 1`, the output sequence from layer $l$ becomes the input sequence
of layer $l+1$. Dropout is applied between layers (but not within a layer or after
the final layer). Stacking increases the network's capacity to model complex temporal
patterns by introducing hierarchical representations: lower layers may capture
short-range patterns while upper layers integrate longer-range structure.

### 1.8 The DeepAR Architecture

DeepAR (Salinas, Flunkert, Gasthaus, and Januschowski, 2020; arXiv 1704.04110v3)
combines an autoregressive recurrent encoder with a parametric likelihood output.
The key innovations relative to a plain LSTM sequence-to-sequence model are:

1. **Probabilistic output**: instead of predicting a point estimate $\hat{y}_t$,
   the model predicts the parameters of a probability distribution at every step.
2. **Global model, local forecasts**: a single model is trained across all time
   series (all symbols), learning shared temporal patterns, while per-series
   information is injected through covariates and embeddings.
3. **Teacher forcing during training**: the model sees the true $y_t$ at each step,
   which stabilises training by preventing error accumulation.
4. **Ancestral sampling at inference**: future values are sampled step by step from
   the predicted distribution, producing full joint sample paths that capture
   temporal correlations in the forecast.

#### 1.8.1 What This Implementation Adopts from the Paper

The following elements are taken directly from Salinas et al.:

- The autoregressive LSTM encoder that conditions each timestep on the previous
  target and covariates.
- Teacher forcing during training (the observed future target is fed back as input).
- Ancestral sampling during inference (each sample path is generated by drawing from
  the predicted distribution at each step and feeding that draw forward).
- The use of categorical embeddings to enable a global model across multiple series.
- The decomposition of the forecast into a probability distribution parameterised by
  neural network outputs.

#### 1.8.2 What This Implementation Modifies

- **Likelihood**: Salinas et al. use a Gaussian (for real-valued data) or negative
  binomial (for count data) likelihood. This project replaces both with a **Student's t
  distribution**, which is specifically motivated by the heavy tails of financial
  return distributions.
- **Covariates**: The paper discusses time-varying covariates (e.g. promotions, day of
  week). This project uses technical indicators (RSI, MACD, volatility, etc.) as
  covariates, which are domain-specific to finance.
- **Scaling**: The paper applies per-series mean scaling to handle different magnitudes.
  This implementation works in log-return space, where series are already on a common
  scale (approximately zero mean, small variance), so no additional scaling is applied.
- **Embedding vocabulary**: The paper uses item (series) embeddings learned jointly.
  This project adds separate embeddings for symbol, timeframe, and asset type.

#### 1.8.3 Architecture Diagram

At each timestep, the model receives a composite input vector formed by concatenating
the previous target value, the exogenous feature vector, and the categorical
embeddings:

```
x_t = [y_{t-1}, features_t, emb_symbol, emb_timeframe, emb_asset_type]
```

The dimensionality of this vector is $1 + \text{input\_size} + 3 \times \text{embedding\_dim}$.
With the default embedding dimension of 8 and, say, 7 exogenous features, the input
width is $1 + 7 + 24 = 32$.

The LSTM processes this sequence:

```
  x_1 ──→ [ LSTM layer 0 ] ──→ [ LSTM layer 1 ] ──→ h_1 ──→ (μ₁, σ₁, ν₁)
             ↓ c_1, h_1                ↓ c_1', h_1'
  x_2 ──→ [ LSTM layer 0 ] ──→ [ LSTM layer 1 ] ──→ h_2 ──→ (μ₂, σ₂, ν₂)
             ↓ c_2, h_2                ↓ c_2', h_2'
  ...
  x_T ──→ [ LSTM layer 0 ] ──→ [ LSTM layer 1 ] ──→ h_T ──→ (μ_T, σ_T, ν_T)
```

Each $h_t$ (from the final LSTM layer) is passed through three parallel linear
projection heads to produce the distribution parameters $(\mu_t, \sigma_t, \nu_t)$.

### 1.9 The Student's t Distribution — Full Treatment

#### 1.9.1 Historical Background

The Student's t distribution was introduced by William Sealy Gosset in 1908 under the
pseudonym "Student". It arises naturally when a normally distributed variable's
variance is estimated from the data rather than known a priori. In the context of this
project, the Student's t is used not because we are estimating variance from finite
samples in the classical sense, but because its heavier tails provide a materially
better fit to the empirical distribution of financial returns than a Gaussian.

#### 1.9.2 Probability Density Function

The probability density function (PDF) of the Student's t distribution with $\nu$
degrees of freedom, location $\mu$, and scale $\sigma$ is:

$$
p(y \mid \mu, \sigma, \nu) =
\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
     {\Gamma\!\left(\frac{\nu}{2}\right) \sqrt{\pi \nu}\, \sigma}
\left(1 + \frac{1}{\nu}\left(\frac{y - \mu}{\sigma}\right)^{\!2}\right)^{-\frac{\nu+1}{2}}
$$

Let us define each symbol:

- $y$ — the observed value (in our case, a log-return $r_t$).
- $\mu$ — the **location parameter**. This is the mode and, when $\nu > 1$, the mean
  of the distribution. It plays the role analogous to the mean of a Gaussian. The
  neural network predicts $\mu$ through an unrestricted linear layer, so it can take
  any value in $(-\infty, +\infty)$.
- $\sigma$ — the **scale parameter**. This controls the spread of the distribution,
  analogous to the standard deviation of a Gaussian. It must be strictly positive.
  The network ensures this by applying the softplus activation followed by adding a
  small epsilon.
- $\nu$ — the **degrees-of-freedom parameter**. This controls the tail heaviness.
  Lower $\nu$ means heavier tails (more probability mass far from the center). As
  $\nu \to \infty$, the Student's t converges to a Gaussian $\mathcal{N}(\mu, \sigma^2)$.
  The network ensures $\nu > 2$ by applying softplus and adding 2.
- $\Gamma(\cdot)$ — the gamma function, a generalisation of the factorial to
  continuous arguments: $\Gamma(n) = (n-1)!$ for positive integers.

#### 1.9.3 Degrees of Freedom — Role and Constraints

The parameter $\nu$ controls the **kurtosis** (tail weight) of the distribution.
Specifically:

- For $\nu > 1$, the mean exists and equals $\mu$.
- For $\nu > 2$, the variance exists and equals $\sigma^2 \frac{\nu}{\nu - 2}$.
  Note that as $\nu \to 2^+$, the variance diverges to infinity.
- For $\nu > 3$, the skewness exists and is zero (the distribution is symmetric).
- For $\nu > 4$, the excess kurtosis is $\frac{6}{\nu - 4}$, which is always
  positive (heavier tails than Gaussian).

This codebase enforces $\nu \geq 2$ (strictly greater than 2 in practice due to the
softplus activation producing strictly positive values). The constraint $\nu > 2$ is
essential because the loss function involves the variance of the distribution
implicitly: during training, if $\nu$ were allowed to drop below 2, the distribution
would have infinite variance, and the loss landscape would become degenerate. The
model could trivially reduce loss by predicting $\nu \to 2^+$ (inflating variance to
infinity), which is a pathological local minimum that provides no useful forecast.

The floor of 2.0 added after softplus ensures $\nu$ is always at least slightly above
2. In practice, trained models typically learn $\nu$ values in the range of 3 to 10
for financial returns, reflecting the well-documented leptokurtosis (fat tails) of
return distributions.

#### 1.9.4 Comparison: Gaussian vs. Student's t

The Gaussian (normal) distribution $\mathcal{N}(\mu, \sigma^2)$ has PDF:

$$
p_{\text{Gauss}}(y \mid \mu, \sigma) =
\frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)
$$

This has exponentially decaying tails: the probability of observing a value more than
4 standard deviations from the mean is approximately $6.3 \times 10^{-5}$ (a 1-in-
16,000 event). In financial markets, 4σ moves occur far more frequently than this.
For example, the S&P 500 has historically experienced daily returns exceeding 4
standard deviations roughly once per year, not once per 44 years as the Gaussian would
predict.

The Student's t distribution decays as a **power law** in the tails:

$$
p(y) \propto |y|^{-(\nu+1)} \quad \text{as } |y| \to \infty
$$

For $\nu = 4$, the tail probability at 4σ is approximately 50 times larger than
under a Gaussian. This is a far better fit for financial returns.

Quantitatively, the key differences:

| Property | Gaussian $\mathcal{N}(\mu, \sigma^2)$ | Student's $t(\mu, \sigma, \nu)$ |
|---|---|---|
| Tail decay | Exponential $\exp(-y^2)$ | Power law $y^{-(\nu+1)}$ |
| Kurtosis | 0 (mesokurtic) | $\frac{6}{\nu - 4}$ for $\nu > 4$ (leptokurtic) |
| Parameters | 2 ($\mu$, $\sigma$) | 3 ($\mu$, $\sigma$, $\nu$) |
| Variance | $\sigma^2$ | $\sigma^2 \frac{\nu}{\nu-2}$ for $\nu > 2$ |
| Gaussian limit | — | $\nu \to \infty$ recovers Gaussian |
| Extreme event prob | Very low | Substantially higher |

For financial risk management, underestimating tail probabilities leads to
underestimating the risk of large losses. A model that assigns near-zero probability
to events that actually occur will be over-confident in its prediction intervals and
poorly calibrated.

### 1.10 The Gaussian Likelihood — Derivation (for Comparison)

Before presenting the Student's t log-likelihood, it is instructive to derive the
Gaussian case, which is simpler and highlights the structure of likelihood-based
learning.

The Gaussian log-likelihood for a single observation $y$ given parameters $\mu, \sigma$ is:

$$
\log p(y \mid \mu, \sigma) = -\frac{1}{2}\log(2\pi) - \log\sigma - \frac{(y - \mu)^2}{2\sigma^2}
$$

Derivation from the PDF:

$$
p(y \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

Taking the natural logarithm:

$$
\log p = \log\frac{1}{\sigma\sqrt{2\pi}} - \frac{(y-\mu)^2}{2\sigma^2}
= -\log\sigma - \frac{1}{2}\log(2\pi) - \frac{(y-\mu)^2}{2\sigma^2}
$$

The **negative log-likelihood** (NLL) is:

$$
-\log p = \log\sigma + \frac{1}{2}\log(2\pi) + \frac{(y-\mu)^2}{2\sigma^2}
$$

Minimising this over $\mu$ and $\sigma$ (with $\sigma$ predicted by the network) teaches the model to:

1. Predict $\mu$ close to the observed $y$ (reducing the squared residual).
2. Predict $\sigma$ that matches the actual spread of residuals (not too large, not
   too small). If $\sigma$ is too large, the $\log\sigma$ term penalises the model.
   If $\sigma$ is too small, the $(y-\mu)^2 / (2\sigma^2)$ term explodes. The optimum
   is $\sigma$ equal to the actual conditional standard deviation.

### 1.11 The Student's t Log-Likelihood — Full Derivation

Starting from the PDF given in Section 1.9.2:

$$
p(y \mid \mu, \sigma, \nu) =
\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
     {\Gamma\!\left(\frac{\nu}{2}\right) \sqrt{\pi \nu}\, \sigma}
\left(1 + \frac{1}{\nu}\left(\frac{y - \mu}{\sigma}\right)^{\!2}\right)^{-\frac{\nu+1}{2}}
$$

We take its logarithm term by term:

**Term 1** — the gamma function ratio:
$$
\log\Gamma\!\left(\frac{\nu+1}{2}\right) - \log\Gamma\!\left(\frac{\nu}{2}\right)
$$

This is the logarithm of $\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)}$. In PyTorch this is
computed as `torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)`. The `lgamma`
function computes $\log\Gamma(x)$ numerically stably for all positive $x$.

**Term 2** — the normalisation:
$$
-\frac{1}{2}\log(\pi \nu)
$$

This comes from $\log\frac{1}{\sqrt{\pi\nu}} = -\frac{1}{2}\log(\pi\nu)$. In code
this is computed as `-0.5 * torch.log(torch.tensor(np.pi) * nu)`. The factor of
$\pi$ is the mathematical constant, and $\nu$ contributes a data-dependent
normalisation.

**Term 3** — the scale:
$$
-\log\sigma
$$

This comes from $\log\frac{1}{\sigma} = -\log\sigma$ and penalises large scale
predictions just as in the Gaussian case.

**Term 4** — the kernel (the "meat" of the distribution):
$$
-\frac{\nu+1}{2}\,\log\!\left(1 + \frac{1}{\nu}\left(\frac{y - \mu}{\sigma}\right)^2\right)
$$

This term involves the standardised residual $z = (y - \mu) / \sigma$. As
$|z|$ grows (the observation is far from the predicted location), the logarithmic
argument increases, but it does so as $\log(1 + z^2/\nu)$, which grows much more
slowly than the $z^2/(2\sigma^2)$ term in the Gaussian case. This slower growth is
precisely what produces the heavier tails.

The code uses `torch.log1p((z ** 2) / nu)` rather than `torch.log(1 + (z ** 2) / nu)`.
The function `log1p(x)` computes $\log(1 + x)$ with higher numerical precision when
$x$ is small — when $z \approx 0$ (the observation is close to the prediction),
$z^2/\nu$ is very small, and the naive computation $\log(1 + \epsilon)$ for tiny
$\epsilon$ loses significant digits due to floating-point cancellation. `log1p` uses a
numerically stable algorithm that avoids this.

Combining all four terms:

$$
\log p(y \mid \mu, \sigma, \nu) =
\underbrace{\log\Gamma\!\left(\frac{\nu+1}{2}\right) - \log\Gamma\!\left(\frac{\nu}{2}\right)}_{\text{Term 1: gamma ratio}}
\underbrace{- \frac{1}{2}\log(\pi\nu)}_{\text{Term 2: normalisation}}
\underbrace{- \log\sigma}_{\text{Term 3: scale penalty}}
\underbrace{- \frac{\nu+1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right)}_{\text{Term 4: kernel}}
$$

The **negative log-likelihood** is the negation of this expression. The training
objective is to minimise the mean NLL over all timesteps across the batch.

### 1.12 How Uncertainty Enters the Loss

The NLL loss is not merely a measure of prediction accuracy (how close $\mu$ is to
$y$); it is fundamentally a measure of **calibrated confidence**. The loss decomposes
into a prediction error component (Term 4) and a confidence component (Terms 2, 3,
and the influence of $\nu$ in Term 4).

Consider what happens when the model inflates $\sigma$ to a very large value:

- Term 3 ($-\log\sigma$) decreases (becomes more negative) — this penalises large $\sigma$.
- Term 4 ($-\frac{\nu+1}{2}\log(1 + z^2/\nu)$) increases (becomes less negative),
  because $z = (y-\mu)/\sigma$ shrinks when $\sigma$ is large, making the kernel
  approach zero.

The optimal $\sigma$ balances these two forces: the model is penalised for being
over-confident ($\sigma$ too small, Term 4 dominates) and for being under-confident
($\sigma$ too large, Term 3 dominates).

Similarly, reducing $\nu$ (predicting heavier tails) reduces the effective penalty for
large residuals (Term 4 becomes less sensitive to $z$), but also changes the
normalisation constants (Terms 1 and 2). The model learns to set $\nu$ high (thin
tails ≈ Gaussian) during calm periods and lower during volatile periods, reflecting
the conditional tail structure of financial returns.

### 1.13 Teacher Forcing

During training, the model has access to the true future target values
$y_{T+1}, \ldots, y_{T+H}$ (where $T$ is the end of the context window and $H$ is
the forecast horizon). Rather than sampling from the predicted distribution and
feeding that sample forward (which introduces compounding noise), the model feeds the
**observed** future value at each step. This technique is called **teacher forcing**.

Formally, during training the RNN input at future time $t > T$ is:

$$
x_t = [y_{t-1}^{\text{observed}},\, 0,\, \text{embeddings}]
$$

(Note: exogenous features for the future are unknown and set to zero.)

At inference time, the model does not have access to $y_t^{\text{observed}}$ for
$t > T$. Instead, it uses its own sampled output:

$$
x_t = [\hat{y}_{t-1}^{\text{sampled}},\, 0,\, \text{embeddings}]
$$

Teacher forcing dramatically stabilises training because the model never needs to
recover from its own mistakes. The downside is a train-test discrepancy: at training
time the model always sees perfect inputs, while at inference time it sees its own
(noisy) predictions. This is a known issue called **exposure bias**. In practice,
for short horizons (24–168 steps), the effect is manageable, and the benefits of
stable training outweigh the bias.

### 1.14 Ancestral Sampling

At inference time, the model generates $N$ independent sample paths, each of length
$H$ (the forecast horizon). For each sample path $n = 1, \ldots, N$:

1. Encode the context window $(y_1, \ldots, y_T)$ through the LSTM, producing the
   final hidden state $(h_T, c_T)$.
2. Set $\hat{y}_T = y_T$ (the last observed value).
3. For each future step $t = T+1, \ldots, T+H$:
   a. Construct input $x_t = [\hat{y}_{t-1},\, 0,\, \text{embeddings}]$.
   b. Run one LSTM step to get $h_t$.
   c. Predict $(\mu_t, \sigma_t, \nu_t) = \text{OutputHeads}(h_t)$.
   d. Draw $\hat{y}_t \sim \text{Student-t}(\mu_t, \sigma_t, \nu_t)$.
4. The sample path is $(\hat{y}_{T+1}^{(n)}, \ldots, \hat{y}_{T+H}^{(n)})$.

The $N$ sample paths form an empirical approximation to the joint forecast
distribution $p(y_{T+1}, \ldots, y_{T+H} \mid y_1, \ldots, y_T)$. Marginal quantiles
at each horizon step are obtained by sorting the samples at that step and reading off
the appropriate order statistics. For example, the 10th percentile forecast at step
$t$ is the 10th percentile of $\{\hat{y}_t^{(1)}, \ldots, \hat{y}_t^{(N)}\}$.

Crucially, because each sample path is generated autoregressively, the samples
capture the **temporal correlation** structure of the forecast. A sample path that
starts with a large upward move at $T+1$ will tend to continue with higher values at
$T+2, T+3, \ldots$, because the high sampled value at $T+1$ feeds into the LSTM and
influences subsequent predictions. This is a significant advantage over approaches
that produce independent marginal distributions at each step.

### 1.15 Gradient Flow Through the Full Pipeline

Understanding how gradients propagate from the loss function backward through
the entire model is essential for diagnosing training issues. Let us trace the
gradient path step by step.

The loss is $L = -\frac{1}{BT}\sum_{b,t} \log p(y_{b,t} \mid \mu_{b,t}, \sigma_{b,t}, \nu_{b,t})$.

**Step 1 — Loss to output parameters.** The partial derivatives
$\frac{\partial L}{\partial \mu_t}$, $\frac{\partial L}{\partial \sigma_t}$,
$\frac{\partial L}{\partial \nu_t}$ are computed analytically from the
log-likelihood formula. For the location parameter:

$$
\frac{\partial L}{\partial \mu_t} = \frac{(\nu+1)(y_t - \mu_t)}{\sigma_t^2(\nu + z_t^2)}
$$

where $z_t = (y_t - \mu_t)/\sigma_t$. This gradient is large when the residual
is large relative to the scale — exactly the behaviour we want. The factor
$(\nu+1)/(\nu + z_t^2)$ is the Student's t "downweighting" effect: when
$z_t^2$ is very large (an outlier), the denominator grows, reducing the gradient
magnitude. This makes the Student's t loss naturally **robust** to outliers,
unlike the Gaussian loss where $\frac{\partial L}{\partial \mu} \propto (y - \mu)/\sigma^2$
grows linearly without bound.

For the scale parameter, the gradient involves the balance between the
$-\log\sigma$ penalty and the kernel term. The equilibrium gradient is zero when
$\sigma$ correctly matches the conditional spread of the data.

**Step 2 — Output parameters to hidden state.** The parameters are linear
functions of $h_t$: $\mu_t = W_\mu h_t + b_\mu$, so
$\frac{\partial \mu_t}{\partial h_t} = W_\mu^T$ (the transpose of the weight
matrix). Similarly for $\sigma_t$ and $\nu_t$, but with the chain rule through
the softplus activation:

$$
\frac{\partial \sigma_t}{\partial h_t} = \text{sigmoid}(W_\sigma h_t + b_\sigma) \cdot W_\sigma^T
$$

since $\text{softplus}'(x) = \sigma(x)$ (the logistic sigmoid). The sigmoid
is always in $(0, 1)$, so the gradient is never exactly zero — this is another
advantage of softplus over ReLU, whose gradient is exactly zero for negative
inputs.

**Step 3 — Hidden state to LSTM parameters.** This is the BPTT component.
The gradient flows backward through the LSTM's unrolled computation graph.
Thanks to the cell state highway (Section 1.7.2), gradients can flow across
the full context window without multiplicative attenuation (though they may still
be modulated by the forget and input gates). Gradient clipping at this stage
(max norm 10.0 in the training loop) prevents individual batches from producing
excessively large parameter updates.

**Step 4 — LSTM to embeddings.** The embedding layers receive gradients through
the concatenated input vector. Only the embedding rows that were accessed in the
current batch receive gradient updates — all other rows remain unchanged. This is
the "sparse update" property of embedding layers.

### 1.16 The Softplus Function — Detailed Analysis

The softplus function appears twice in the output layer (for $\sigma$ and $\nu$)
and deserves a detailed treatment because it is central to the model's ability to
predict valid distribution parameters.

**Definition:**
$$
\text{softplus}(x) = \log(1 + e^x)
$$

**Key properties:**

1. **Strictly positive:** $\text{softplus}(x) > 0$ for all $x \in \mathbb{R}$.
   This follows from $1 + e^x > 1$, so $\log(1 + e^x) > 0$.

2. **Smooth approximation of ReLU:** As $x \to +\infty$,
   $\text{softplus}(x) \approx x$. As $x \to -\infty$,
   $\text{softplus}(x) \approx e^x \to 0^+$.

3. **Non-zero gradient everywhere:** $\text{softplus}'(x) = \sigma(x) = \frac{e^x}{1 + e^x} \in (0, 1)$.
   This ensures that the model can always adjust $\sigma$ and $\nu$ regardless of
   the current parameter values.

4. **Linear growth for large arguments:** Unlike $\exp(x)$, which grows
   exponentially and can overflow float32 ($\sim 3.4 \times 10^{38}$) for
   $x > 88$, softplus grows linearly. This prevents the model from accidentally
   predicting $\sigma \to \infty$ through a single large weight update.

5. **Numerical implementation:** PyTorch's `F.softplus` uses a piecewise
   approximation: for $x > 20$, it returns $x$ directly (since $\log(1 + e^{20}) \approx 20$ to float32 precision). For $x < -15$, it returns $e^x$ (since
   $\log(1 + e^{-15}) \approx e^{-15}$ to float32 precision). Only in the
   middle range does it compute the full formula. This avoids numerical overflow
   from $e^x$ for large positive $x$.

**Why not use $\exp$?** The exponential function $\exp(x)$ is another common
positivity-enforcing activation. Its advantage is mathematical elegance (the
log-likelihood simplifies when $\sigma = e^s$ because $\log\sigma = s$). Its
disadvantage is severe: a gradient update that increases $s$ by 10 multiplies
$\sigma$ by $e^{10} \approx 22{,}000$, causing catastrophic instability. The
softplus function is much better behaved: a gradient update that increases the
pre-activation by 10 increases the output by approximately 10 (in the linear
regime), not by a factor of 22,000.

### 1.17 Embedding Theory for Categorical Variables

The three embedding layers (symbol, timeframe, asset type) implement a learned
dense representation for categorical variables. This section explains why embeddings
are preferable to alternative encodings.

**One-hot encoding:** The simplest representation assigns a binary vector of length
$V$ (vocabulary size) to each category, with a 1 in the position corresponding to
the category and 0s elsewhere. For $V = 10$ symbols, this adds 10 dimensions to
the RNN input. The problems are:

1. **Dimensionality:** For large vocabularies, the input becomes very wide.
2. **No sharing:** Each category is equidistant from all others in Euclidean space.
   BTC and ETH, which share many statistical properties (both crypto, similar
   volatility dynamics), would be no more "similar" than BTC and SPY.
3. **Sparse gradients:** Only one dimension is active per sample, wasting
   computation.

**Embedding:** Each category is mapped to a dense vector of dimension $d_e$
(8 in this project). These vectors are learned jointly with the rest of the model.
During training, categories that produce similar hidden states will develop similar
embeddings. This is a form of transfer learning: information learned about BTC
can partially transfer to ETH through similar embedding vectors.

The total parameter cost of the three embeddings is:
$(10 + 5 + 3) \times 8 = 144$ parameters, which is negligible compared to the
LSTM's tens of thousands of parameters.

### 1.18 The Adam Optimiser — Internal Mechanics

Adam (Adaptive Moment Estimation, Kingma and Ba 2015) is the optimiser used for
training. It maintains two running averages per parameter:

**First moment (mean of gradients):**
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**Second moment (mean of squared gradients):**
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

where $g_t = \nabla_\theta L_t$ is the gradient at step $t$, $\beta_1 = 0.9$, and
$\beta_2 = 0.999$ (default values used in this project).

The bias-corrected estimates are:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The bias correction compensates for the initialisation at zero: without it, the
early estimates of $m_t$ and $v_t$ would be biased toward zero.

The parameter update is:
$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\alpha = 10^{-3}$ is the learning rate and $\epsilon = 10^{-8}$ is a small
constant preventing division by zero.

The adaptive nature of Adam means that parameters with consistently large gradients
have their effective learning rate reduced (because $v_t$ is large), while parameters
with small or noisy gradients maintain a higher effective learning rate. This is
particularly useful for RNNs, where different parameters (e.g. forget gate biases
vs. input gate weights) may have very different gradient magnitudes.

The weight decay of $10^{-5}$ adds a penalty $\lambda \|\theta\|^2$ to the loss,
which nudges all weights toward zero. In Adam's implementation (decoupled weight
decay, also called AdamW), this is applied directly to the parameters rather than
through the gradient, which is theoretically more principled.

### 1.19 ReduceLROnPlateau — Learning Rate Scheduling

The learning rate scheduler monitors the validation loss and reduces the learning
rate when progress stalls. The specific schedule used is:

- Monitor: validation loss (mode="min").
- Factor: 0.5 (halve the learning rate).
- Patience: `patience // 2` (typically 5 epochs).

When the validation loss has not improved for 5 consecutive epochs, the learning
rate is multiplied by 0.5. This can happen multiple times during training:
$10^{-3} \to 5 \times 10^{-4} \to 2.5 \times 10^{-4} \to \ldots$

The rationale is that a large learning rate helps the optimiser explore the loss
landscape broadly in early training, but prevents convergence to a precise minimum
in later training. Reducing the learning rate allows finer-grained updates near the
optimum. The ReduceLROnPlateau strategy adapts the timing of these reductions to
the actual training dynamics, rather than using a fixed schedule.

### 1.20 Early Stopping — Theory and Justification

Early stopping is a regularisation technique that halts training when the validation
loss begins to increase, indicating that the model has started to overfit. The
implementation maintains a counter:

1. After each epoch, compare the validation loss to the best seen so far.
2. If improved: reset the counter to 0 and save a checkpoint.
3. If not improved: increment the counter.
4. If the counter reaches the patience threshold (10): stop training.

The model returned to the user is the checkpoint with the best validation loss,
not the model from the final epoch. This means that even if training continues for
several epochs beyond the optimum (before patience is exhausted), the user receives
the best version.

Early stopping is closely related to L2 regularisation (weight decay) in its effect
on the model's complexity. In a simplified linear model, early stopping corresponds
to implicit L2 regularisation with a penalty strength that decreases as training
continues. In deep networks the relationship is less precise, but the intuition
holds: stopping early prevents the model from memorising training-set noise.

### 1.21 References

- Salinas D., Flunkert V., Gasthaus J., Januschowski T. (2020). *DeepAR:
  Probabilistic forecasting with autoregressive recurrent networks.* International
  Journal of Forecasting 36(3), 1181–1191. arXiv: **1704.04110v3**.
- Hochreiter S., Schmidhuber J. (1997). *Long Short-Term Memory.* Neural
  Computation 9(8), 1735–1780.
- Gosset W. S. (writing as "Student") (1908). *The probable error of a mean.*
  Biometrika 6(1), 1–25.
- Kingma D. P., Ba J. (2015). *Adam: A method for stochastic optimization.*
  ICLR 2015. arXiv: **1412.6980**.
- Engle R. F. (1982). *Autoregressive conditional heteroscedasticity with estimates
  of the variance of United Kingdom inflation.* Econometrica 50(4), 987–1007.
- Jegadeesh N., Titman S. (1993). *Returns to buying winners and selling losers:
  Implications for stock market efficiency.* Journal of Finance 48(1), 65–91.

---

## Chapter 2 — Code Walk-through (Line-by-Line)

This chapter provides a detailed examination of every line of machine-learning code
in the system. The reader should have the source files open alongside this manual.
For each function and logical block, we explain what it does, why it exists, what
mathematical concept it implements, and what would break if it were removed or changed.

### 2.1 `models/deepar.py` — Model Definition

This file contains 460 lines defining the DeepAR model architecture.

#### 2.1.1 Module Imports (Lines 1–16)

```python
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats
```

- `torch.nn` provides the base classes for neural network modules (`nn.Module`,
  `nn.Linear`, `nn.LSTM`, `nn.Embedding`). Every trainable component inherits from
  `nn.Module`.
- `torch.nn.functional` (aliased as `F`) provides stateless versions of common
  operations. Here it is used for `F.softplus`, which is a function rather than a
  module object.
- `scipy.stats` is imported specifically for `stats.t.rvs` — sampling from the
  Student's t distribution during inference. PyTorch does not include a built-in
  Student's t distribution, so sampling is delegated to SciPy and then converted back
  to tensors. This is computationally acceptable because sampling only occurs during
  inference (not during training, where gradients must flow through PyTorch operations).
- `loguru.logger` is the project's logging framework, used pervasively for diagnostic
  output.

#### 2.1.2 `StudentTOutput` Class (Lines 18–66)

```python
class StudentTOutput(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.nu_layer = nn.Linear(hidden_size, 1)
```

This class inherits from `nn.Module`, which is PyTorch's base class for all neural
network modules. Inheriting from `nn.Module` is mandatory for several reasons:

1. It enables automatic parameter registration — when you assign an `nn.Linear` as an
   attribute, PyTorch automatically discovers its weights and biases and includes them
   in `model.parameters()`, which the optimiser iterates over.
2. It enables `.to(device)` to move all parameters to GPU/CPU.
3. It enables `model.train()` / `model.eval()` mode switching, which affects dropout
   and batch normalisation behaviour.

Three independent `nn.Linear` layers are created, each projecting from
`hidden_size` (64 by default) to a single output dimension. Each linear layer
implements $o = Wx + b$ where $W$ has shape $(1, \text{hidden\_size})$ and $b$ has
shape $(1,)$. The three layers share no parameters — they are completely independent
projections of the same hidden state $h_t$. This is by design: $\mu$, $\sigma$, and
$\nu$ are conceptually different quantities with different constraints and should be
learned independently.

Why three separate layers rather than one layer with output size 3? Either design
would work mathematically, but three layers make the activation constraints cleaner to
apply: each output passes through a different activation pipeline.

```python
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_layer(h)
        sigma = F.softplus(self.sigma_layer(h)) + 1e-6
        nu = F.softplus(self.nu_layer(h)) + 2.0
        return mu, sigma, nu
```

The `forward` method is the standard PyTorch entry point for computation. When you
call `output_layer(h)`, PyTorch dispatches to `forward(h)`. This is the method that
the autograd engine traces to build the computation graph for backpropagation.

**`mu`**: No activation is applied; the location parameter is unrestricted and can take
any real value. The linear layer's output directly becomes $\mu_t$.

**`sigma`**: The softplus function is defined as $\text{softplus}(x) = \log(1 + e^x)$.
It is a smooth approximation to the ReLU function that is strictly positive for all
inputs. Unlike $\exp(x)$ (another common positivity-enforcing activation), softplus
grows linearly for large $x$ rather than exponentially, which makes it numerically
stable and prevents training instability when the pre-activation value is large.

The epsilon $10^{-6}$ is added after softplus to provide a hard floor. Without it,
softplus could produce values extremely close to zero (for very negative
pre-activation inputs), leading to division-by-zero errors in the log-likelihood
computation (which contains $-\log\sigma$ and $z^2 = (y-\mu)^2/\sigma^2$). The value
$10^{-6}$ is small enough to have no practical effect on the distribution but large
enough to prevent numerical overflow in $1/\sigma^2$.

**What would break if softplus were replaced with ReLU?** ReLU outputs exactly 0 for
negative inputs. If the pre-activation value happened to be negative, $\sigma$ would
be exactly 0 (plus epsilon), which is pathologically small. The gradient of ReLU is 0
for negative inputs, so the model could not learn to increase $\sigma$ from that state.
Softplus has a non-zero gradient everywhere, ensuring the model can always adjust
$\sigma$.

**What would break if the epsilon were removed?** For most training runs, the epsilon
never matters because softplus rarely produces values close to zero. However, during
the early stages of training (when weights are randomly initialised), the
pre-activation values may be highly negative, making softplus output values of order
$10^{-10}$. Computing $\log\sigma$ for such values produces very large negative
numbers, and $(y-\mu)^2/\sigma^2$ produces very large positive numbers. The two terms
partially cancel in the log-likelihood, but their individual magnitudes can exceed
float32's representable range ($\sim 10^{38}$), producing `inf` or `nan` values that
corrupt the loss and propagate through the gradient computation, breaking training
entirely. The epsilon prevents this.

**`nu`**: Softplus plus 2.0 ensures $\nu > 2$ always. As discussed in Section 1.9.3,
$\nu = 2$ is the boundary where the variance of the Student's t distribution becomes
infinite. Allowing the model to predict $\nu \leq 2$ would create a degenerate loss
landscape where the variance diverges. The floor of 2.0 means the minimum
degrees-of-freedom value immediately after initialisation is $\text{softplus}(0) + 2 =
\log 2 + 2 \approx 2.69$.

**Tensor shapes**: The input `h` has shape `(batch, seq_len, hidden_size)`. Each
`nn.Linear` operates on the last dimension, producing shape
`(batch, seq_len, 1)`. The shapes of `mu`, `sigma`, and `nu` match — all are
`(batch, seq_len, 1)`. This trailing dimension of 1 is maintained throughout the
model for consistency with the target tensor's shape `(batch, seq_len, 1)`.

#### 2.1.3 `student_t_log_likelihood` Function (Lines 68–104)

```python
def student_t_log_likelihood(y, mu, sigma, nu):
    z = (y - mu) / sigma
    log_gamma_term = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
    log_norm = -0.5 * torch.log(torch.tensor(np.pi, device=y.device) * nu)
    log_scale = -torch.log(sigma)
    log_kernel = -((nu + 1) / 2) * torch.log1p((z ** 2) / nu)
    log_prob = log_gamma_term + log_norm + log_scale + log_kernel
    return log_prob
```

This function implements the four-term decomposition derived in Section 1.11. Every
tensor operation here is differentiable via PyTorch's autograd, so gradients flow
backward through this function during training.

**`z = (y - mu) / sigma`** — the standardised residual. Shape: same as inputs,
`(batch, seq_len, 1)`. This is the normalised distance from the prediction to the
observation, measured in units of the predicted scale. Division by `sigma` is
element-wise because all tensors are broadcastable.

**`log_gamma_term`** — uses `torch.lgamma`, which computes $\log\Gamma(x)$. The gamma
function generalises the factorial: $\Gamma(n) = (n-1)!$ for positive integers, and
interpolates smoothly for real arguments. `lgamma` is numerically stable and
differentiable, returning gradients that PyTorch can backpropagate through.

**`log_norm`** — the constant $\pi$ is created as a tensor on the same device as `y`
via `torch.tensor(np.pi, device=y.device)`. If `y` is on GPU, $\pi$ must also be on
GPU for the multiplication to work. This is a common pitfall in PyTorch: if you create
a constant without specifying the device, operations between CPU and GPU tensors will
raise a runtime error.

**`log_kernel`** — uses `torch.log1p` as discussed in Section 1.11. The expression
`(z ** 2) / nu` can be very small when the prediction is accurate ($z \approx 0$),
and `log1p` avoids the floating-point precision loss inherent in computing
$\log(1 + \epsilon)$ for small $\epsilon$.

**Return value**: `log_prob` has shape `(batch, seq_len, 1)`. Each element is the
log-likelihood of the corresponding observation under the predicted distribution.
The training loop averages this across all elements and negates it to obtain the loss.

**What would break if this function were removed?** Without it, there is no loss
function. The model cannot be trained.

#### 2.1.4 `DeepARStudentT.__init__` (Lines 106–197)

```python
class DeepARStudentT(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1,
                 rnn_type="lstm", embedding_dim=8, num_symbols=10,
                 num_timeframes=5, num_asset_types=3):
        super().__init__()
```

The `super().__init__()` call is mandatory for `nn.Module` subclasses. It initialises
the internal machinery that tracks parameters, submodules, and the module's training
mode.

```python
        if input_size < 1:
            logger.warning(f"... clamping to 1")
            input_size = 1
```

**The input_size guard**. The LSTM requires a positive input dimension. If
`input_size` were 0, the RNN input tensor would have shape
`(batch, seq_len, 1 + 0 + 24) = (batch, seq_len, 25)`, which is valid, but the saved
`self.input_size = 0` would cause problems later when allocating zero-feature tensors
during sampling (the resulting tensor would have shape `(batch, seq_len, 0)`, which
is a degenerate 0-dimensional tensor that cannot participate in concatenation or
matrix multiplication). The clamp to 1 ensures dummy features of dimension 1 are
always created, which are filled with zeros during inference.

This guard was added to fix a production bug where old training runs stored
`input_size=0` in metadata, and the forecast endpoint would attempt to create tensors
with a 0-sized last dimension, causing reshape errors.

```python
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.timeframe_embedding = nn.Embedding(num_timeframes, embedding_dim)
        self.asset_type_embedding = nn.Embedding(num_asset_types, embedding_dim)
```

Three `nn.Embedding` layers provide learned dense representations for categorical
variables. Each embedding is a lookup table: given an integer index $i$, it returns
the $i$-th row of a learnable weight matrix of shape `(vocab_size, embedding_dim)`.
The gradients from the loss propagate back through the embedding lookup, updating the
rows that were accessed.

The embedding dimension of 8 was chosen as a pragmatic balance: large enough to
encode meaningful distinctions between symbols (e.g. BTC vs. ETH have different
volatility regimes) but small enough not to dominate the RNN input width. The
vocabulary sizes (10 symbols, 5 timeframes, 3 asset types) are upper bounds on the
number of distinct values expected in production. If a symbol index exceeds the
vocabulary size, PyTorch will raise an `IndexError`. In practice, the API assigns
index 0 to all symbols (since each training run typically targets a single symbol),
but the embedding infrastructure is in place for multi-series global models.

```python
        rnn_input_size = 1 + input_size + 3 * embedding_dim
```

The RNN input at each timestep is the concatenation of:

- 1 — the target value $y_{t-1}$ (a scalar, so dimension 1).
- `input_size` — the exogenous feature vector (e.g. volatility, RSI, etc.).
- $3 \times 8 = 24$ — three embedding vectors, each of dimension 8.

With 7 features (the default when all features are enabled), this gives
$1 + 7 + 24 = 32$. When no features are enabled and the dummy dimension is used,
it gives $1 + 1 + 24 = 26$.

```python
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
```

The LSTM is instantiated with `batch_first=True`, meaning input and output tensors
have shape `(batch, seq_len, features)` rather than `(seq_len, batch, features)`.
This is a matter of convention; the `batch_first` setting must be consistent
throughout the model.

Dropout is set to 0 when there is only one layer because PyTorch applies inter-layer
dropout (between stacked LSTM layers), and with only one layer there is no inter-layer
connection to drop out. Setting a non-zero dropout with one layer has no effect but
triggers a PyTorch warning.

```python
        self.output_layer = StudentTOutput(hidden_size)
```

The output layer is the `StudentTOutput` module described in Section 2.1.2. It is
registered as a submodule and its parameters are included in `model.parameters()`.

#### 2.1.5 `DeepARStudentT.forward` — Training Path (Lines 199–287)

The `forward` method is called during training (and during inference for context
encoding). Its signature accepts both past and future target tensors.

```python
    def forward(self, past_target, past_features, future_target=None,
                symbol_ids=None, timeframe_ids=None, asset_type_ids=None):
        batch_size = past_target.size(0)
        context_length = past_target.size(1)
        device = past_target.device
```

`past_target.size(0)` reads the first dimension of the tensor, which is the batch
size. `.device` returns whether the tensor is on CPU or GPU. All newly created tensors
must be placed on the same device.

```python
        if symbol_ids is None:
            symbol_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
```

Default embedding indices. When no categorical IDs are provided (the common case in
this project, since each run trains on a single symbol), all samples use index 0. The
dtype `torch.long` (int64) is required by `nn.Embedding`'s lookup operation.

```python
        symbol_emb = self.symbol_embedding(symbol_ids)        # (batch, emb_dim)
        symbol_emb = symbol_emb.unsqueeze(1).repeat(1, context_length, 1)
```

The embedding lookup produces a vector per sample: shape `(batch, 8)`. To concatenate
with the time-varying inputs, this must be expanded to `(batch, context_length, 8)`.
`unsqueeze(1)` adds a dummy time dimension: `(batch, 1, 8)`. `.repeat(1, context_length, 1)`
tiles this across the time dimension: `(batch, context_length, 8)`. The embedding is
constant across all timesteps — the symbol does not change within a sequence.

```python
        rnn_input = torch.cat([past_target, past_features, symbol_emb, timeframe_emb, asset_type_emb], dim=-1)
```

Concatenation along the feature dimension (dim=-1). The resulting shape is
`(batch, context_length, rnn_input_size)`.

```python
        rnn_output, hidden = self.rnn(rnn_input)
```

The LSTM processes the entire context window in a single call. `rnn_output` has shape
`(batch, context_length, hidden_size)` — the hidden state at every timestep.
`hidden` is a tuple `(h_n, c_n)` where each has shape
`(num_layers, batch, hidden_size)` — the final hidden and cell states for each layer.
This `hidden` state will be used to initialise the RNN for the future (forecast)
portion.

```python
        if future_target is not None:
            horizon = future_target.size(1)
            ...
            future_features = torch.zeros(batch_size, horizon, self.input_size, device=device)
```

During training, the future portion feeds the **observed** future target (teacher
forcing). Exogenous features for the future are unknown — they are set to zero. This
is a simplification: in a production system, some features might be known in advance
(e.g. calendar features), but in this project all features are derived from price data
that does not exist in the future.

```python
            future_output, _ = self.rnn(future_input, hidden)
            full_output = torch.cat([rnn_output, future_output], dim=1)
            mu, sigma, nu = self.output_layer(full_output)
```

The LSTM continues from the context's final hidden state, processing the future
inputs. The outputs from context and future are concatenated along the time axis to
form `full_output` of shape `(batch, context_length + horizon, hidden_size)`. The
output layer predicts $(\mu, \sigma, \nu)$ for **every** timestep in the
concatenated sequence. The loss is then computed over all timesteps, not just the
future ones. This means the model is trained to predict the distribution of observing
the context values as well as the future values, which provides additional training
signal and regularises the model.

#### 2.1.6 `DeepARStudentT.sample` — Inference Path (Lines 289–420)

The `sample` method implements ancestral sampling as described in Section 1.14.

```python
        self.eval()
        with torch.no_grad():
```

`self.eval()` switches the model to evaluation mode. This disables dropout (all
neurons are active) and changes batch normalisation behaviour (not used here, but good
practice). `torch.no_grad()` disables the autograd engine entirely: no computation
graph is built, no gradients are stored. This reduces memory consumption by roughly
50 % and speeds up computation. Since we are not training (no backward pass), gradients
would be wasted.

```python
            past_target = past_target.to(dtype=torch.float32)
            past_features = past_features.to(dtype=torch.float32)
```

Explicit dtype casts. The model's weights are `float32` (PyTorch's default). If the
input data arrives as `float64` (which happens when numpy arrays are created without
explicit dtype), the matrix multiplication would produce `float64` outputs, and the
model's weights would be silently upcast, consuming twice the memory and potentially
causing dtype mismatch errors in downstream operations. This guard was added after a
production bug where `float64` numpy arrays from pandas caused a cascade of dtype
errors.

```python
            safe_input = max(self.input_size, 1)
            if past_features.size(-1) < 1:
                ...
                past_features = torch.zeros(batch_size, context_length, safe_input, ...)
```

Defensive replacement of 0-dimensional features. If the feature tensor has a last
dimension of 0 (which was a recurring production bug caused by old training runs with
empty feature lists), concatenation with other tensors would produce a tensor of the
correct total width minus the feature width, causing a shape mismatch when the LSTM
tries to process it. Replacing with a zeros tensor of the correct dimension ensures
the LSTM receives the input width it was trained with.

```python
            past_target_rep = past_target.repeat(n_samples, 1, 1)
```

Replication for parallel sampling. Instead of running `n_samples` sequential
sample paths, the batch dimension is inflated from `batch_size` to
`batch_size * n_samples`. This exploits GPU parallelism: all 100 sample paths are
processed in a single batched LSTM call, which is orders of magnitude faster than 100
sequential calls. The shape goes from `(1, context_length, 1)` to
`(n_samples, context_length, 1)`.

```python
            _, hidden = self.rnn(rnn_input)
```

Context encoding. The full output is discarded (`_`); only the final hidden state
`hidden` is kept. This hidden state encapsulates the LSTM's understanding of the
context window and serves as the starting point for autoregressive generation.

```python
            for t in range(horizon):
                ...
                output_t, hidden = self.rnn(input_t, hidden)
                mu_t, sigma_t, nu_t = self.output_layer(output_t)
```

The autoregressive loop. At each step, the LSTM processes a single timestep
(sequence length 1) and updates its hidden state. The output heads predict the
distribution parameters.

```python
                mu_np = mu_t.squeeze(-1).cpu().numpy()
                sigma_np = sigma_t.squeeze(-1).cpu().numpy()
                nu_np = nu_t.squeeze(-1).cpu().numpy()

                samples_t = []
                for i in range(mu_np.shape[0]):
                    s = stats.t.rvs(df=nu_np[i, 0], loc=mu_np[i, 0], scale=sigma_np[i, 0], size=1)
                    samples_t.append(s[0])
```

Sampling is performed per element using SciPy's `stats.t.rvs`. The tensors are moved
to CPU and converted to numpy because SciPy operates on numpy arrays. The loop
iterates over all `batch_size * n_samples` elements — this is the computational
bottleneck of inference. A more efficient implementation would use PyTorch's built-in
`torch.distributions.StudentT`, but the SciPy approach was chosen for numerical
fidelity (SciPy's implementation is battle-tested for edge cases with very small or
very large $\nu$).

```python
                current_target = sample_t
```

The sampled value becomes the input for the next timestep — this is the
autoregressive feedback. The randomness in this sample propagates forward through all
subsequent timesteps, which is what makes different sample paths diverge and capture
the uncertainty of the forecast.

```python
            samples = torch.cat(samples, dim=1)
            samples = samples.view(batch_size, n_samples, horizon, 1)
```

The concatenated tensor has shape `(batch_size * n_samples, horizon, 1)`. The `view`
reshapes it to `(batch_size, n_samples, horizon, 1)` without copying data. This
separates the sample paths back into the sample dimension, enabling downstream
quantile computation.

#### 2.1.7 `compute_quantiles` (Lines 422–440)

```python
    def compute_quantiles(self, samples, quantiles=[0.1, 0.5, 0.9]):
        quantile_dict = {}
        for q in quantiles:
            quantile_dict[q] = torch.quantile(samples, q, dim=1)
        return quantile_dict
```

`torch.quantile(samples, q, dim=1)` computes the $q$-th quantile along the sample
dimension (dim=1). For the 0.5 quantile, this is the median. For 0.1 and 0.9, these
form the 80% prediction interval. The output shape for each quantile is
`(batch, horizon, 1)` — eliminating the sample dimension.

The function returns a dictionary rather than a tensor because the set of requested
quantiles may vary between callers. The forecast endpoint requests
`[0.025, 0.1, 0.5, 0.9, 0.975]` (five quantiles forming 80% and 95% bands), while
other callers might request a different set.

Internally, `torch.quantile` sorts the samples along the specified dimension and
interpolates between order statistics to find the requested percentile. For
$N = 100$ samples, the 10th percentile corresponds to (roughly) the 10th-smallest
sample value, with linear interpolation between the 10th and 11th for exact
positioning.

#### 2.1.8 Complete Data Flow Summary for `deepar.py`

To consolidate understanding, here is the complete data flow through the model during
one training step, with tensor shapes annotated at every stage. Assume
`batch_size=32`, `context_length=168`, `horizon=24`, `input_size=7`, `hidden_size=64`,
`num_layers=2`, `embedding_dim=8`:

```
Input:
  past_target:   (32, 168, 1)        — 168 historical log-returns
  past_features: (32, 168, 7)        — 7 exogenous features over context
  future_target: (32, 24, 1)         — 24 future log-returns (teacher forcing)

Embedding (each):
  symbol_ids:    (32,)               — integer indices
  → embed:       (32, 8)             — dense vector
  → unsqueeze:   (32, 1, 8)          — add time dim
  → repeat:      (32, 168, 8)        — tile to context length

Concatenation (context):
  rnn_input:     (32, 168, 1+7+8+8+8) = (32, 168, 32)

LSTM (context):
  rnn_output:    (32, 168, 64)        — hidden states for all context steps
  hidden:        ((2, 32, 64), (2, 32, 64))  — (h_n, c_n) for 2 layers

Future input construction:
  future_target:  (32, 24, 1)
  zero_features:  (32, 24, 7)         — zeros (features unknown)
  embeddings:     (32, 24, 24)        — tiled embeddings
  future_input:   (32, 24, 32)

LSTM (future, continued from hidden):
  future_output:  (32, 24, 64)

Concatenation (full):
  full_output:    (32, 192, 64)       — 168 + 24 steps

Output heads:
  mu:             (32, 192, 1)
  sigma:          (32, 192, 1)
  nu:             (32, 192, 1)

Loss computation:
  full_target:    (32, 192, 1)        — concatenation of past and future targets
  log_likelihood: (32, 192, 1)        — per-element log-likelihood
  loss:           scalar              — mean NLL = -mean(log_likelihood)
```

This complete trace demonstrates that every dimension is accounted for, every
concatenation preserves the batch and time dimensions, and the final loss is a
scalar that can be backpropagated.

#### 2.1.9 PyTorch Concepts Demonstrated in `deepar.py`

**`nn.Module` lifecycle.** The model class inherits from `nn.Module`, which requires:
(a) calling `super().__init__()` to initialise the module registry,
(b) assigning submodules as attributes so they are discoverable by
`parameters()`, `to()`, `state_dict()`, etc.,
(c) implementing `forward()` as the computation entry point.

**Autograd.** Every tensor operation in the `forward` method (concatenation, LSTM
evaluation, softplus, log-likelihood computation) is recorded in a directed acyclic
graph (DAG). Calling `loss.backward()` traverses this graph in reverse topological
order, applying the chain rule at each node to compute
$\partial L / \partial \theta$ for every parameter $\theta$. The user never writes
explicit gradient formulae — PyTorch derives them automatically.

**`torch.no_grad()` context.** In the `sample()` method, `torch.no_grad()` disables
the DAG construction. This is a performance optimisation: without it, PyTorch would
allocate memory for every intermediate tensor's gradient, which is wasteful during
inference.

**`model.train()` vs `model.eval()`.** Calling `model.train()` enables dropout (random
neuron zeroing) and enables any stochastic behaviour in the model. Calling
`model.eval()` disables these, ensuring deterministic outputs for the same input.
In this model, the LSTM's inter-layer dropout is the only component affected by
this switch.

**`state_dict()`.** The model's learnable parameters are serialised as an
`OrderedDict` of name → tensor pairs. The keys follow a hierarchical naming
convention: `rnn.weight_ih_l0` refers to the input-hidden weight matrix of layer 0
of the module named `rnn`. This naming is what enables the reverse engineering of
architecture from checkpoints in the forecast endpoint.

### 2.2 `models/training.py` — Training Loop

#### 2.2.1 `DeepARTrainer.__init__` (Lines 17–55)

```python
class DeepARTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", seed=42):
        self.model = model.to(device)
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
```

`model.to(device)` moves all model parameters to the specified device (GPU or CPU).
This is a no-op if the model is already on the correct device.

Seed setting ensures reproducibility. `torch.manual_seed(seed)` seeds the CPU random
number generator. `np.random.seed(seed)` seeds numpy's generator (used in SciPy
sampling during inference). `torch.cuda.manual_seed_all(seed)` seeds all GPU
generators (relevant if training on GPU). Without reproducible seeds, training runs
produce different results each time, making debugging extremely difficult.

#### 2.2.2 `DeepARTrainer.train` (Lines 57–165)

```python
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

The Adam optimiser (Kingma and Ba, 2015) is an adaptive learning rate optimiser that
maintains per-parameter first and second moment estimates. It is the default choice
for deep learning because it converges faster than plain SGD for most architectures
and is relatively insensitive to hyperparameter tuning.

`weight_decay=1e-5` adds L2 regularisation to the weight updates:
$\theta_{t+1} = \theta_t - \alpha (\nabla L + \lambda \theta_t)$, where
$\lambda = 10^{-5}$. This weakly penalises large weights, reducing overfitting.

```python
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)
```

This scheduler monitors the validation loss and halves the learning rate if the loss
has not improved for `patience // 2` epochs (default: 5). Reducing the learning rate
allows the optimiser to make finer-grained updates as it approaches a minimum, often
recovering from plateaus where a larger learning rate would oscillate.

```python
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            ...
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, optimizer, val_loss)
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
```

**Early stopping**: If the validation loss has not improved for `patience` consecutive
epochs (default: 10), training halts. This is the primary regularisation mechanism:
the model is saved at the point of best validation performance, and subsequent
overfitting epochs are discarded. Without early stopping, the model would overfit to
the training set, producing excellent training loss but poor generalisation.

#### 2.2.3 `DeepARTrainer._train_epoch` (Lines 167–213)

```python
        self.model.train()
```

Sets the model to training mode. This enables dropout (random neurons are zeroed with
probability $p$ during each forward pass to prevent co-adaptation).

```python
            if past_target.dim() == 2:
                past_target = past_target.unsqueeze(-1)
```

The sequence creation function produces `(batch, context_length)` arrays (2D). The
model expects `(batch, context_length, 1)` (3D with explicit feature dimension).
`unsqueeze(-1)` adds the trailing dimension without copying data.

```python
            mu, sigma, nu = self.model(past_target=past_target, past_features=past_features, future_target=future_target)
            full_target = torch.cat([past_target, future_target], dim=1)
            log_likelihood = student_t_log_likelihood(full_target, mu, sigma, nu)
            loss = -log_likelihood.mean()
```

The forward pass produces parameters for all timesteps (context + horizon). The target
tensor is the concatenation of context and future values. The log-likelihood is
computed for every timestep, and its negation (NLL) is the loss. `.mean()` averages
over all elements: batch, time, and the trailing dimension.

```python
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
```

This is the standard PyTorch training step:

1. `zero_grad()` — clears accumulated gradients from the previous iteration. Without
   this, gradients would accumulate across batches, producing incorrect updates.
2. `loss.backward()` — runs backpropagation, computing $\partial L / \partial \theta$
   for every parameter $\theta$. This traverses the computation graph from the loss
   backward through every tensor operation (log_likelihood, softplus, LSTM, etc.).
3. `clip_grad_norm_` — gradient clipping. If the total gradient norm exceeds 10.0,
   all gradients are scaled down proportionally. This prevents **exploding gradients**,
   which can occur in RNNs when the loss landscape has steep cliffs. Without clipping,
   a single pathological batch could produce gradients of magnitude $10^{10}$, sending
   the parameters to extreme values and destroying the model.
4. `optimizer.step()` — applies the Adam update rule to all parameters using the
   (clipped) gradients.

#### 2.2.4 `_validate_epoch` (Lines 215–252)

```python
        self.model.eval()
        ...
        with torch.no_grad():
```

Evaluation mode + no gradient computation. Dropout is disabled (all neurons active),
giving a deterministic output. `torch.no_grad()` saves memory by not recording
operations.

The validation loss computation mirrors the training loss exactly (same forward pass,
same NLL), except no backward pass or parameter update occurs.

#### 2.2.5 `_create_dataloader` (Lines 254–270)

```python
        past_target = torch.FloatTensor(data["past_target"])
        past_features = torch.FloatTensor(data["past_features"])
        future_target = torch.FloatTensor(data["future_target"])
        dataset = TensorDataset(past_target, past_features, future_target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

`torch.FloatTensor` converts numpy arrays to float32 tensors. This is where the
dtype boundary is crossed: all upstream data preparation uses numpy float32 arrays,
and this function converts them to PyTorch tensors.

`TensorDataset` is a simple container that groups corresponding elements from multiple
tensors. Indexing `dataset[i]` returns `(past_target[i], past_features[i],
future_target[i])`.

`DataLoader` wraps the dataset with batching and shuffling. On each epoch, if
`shuffle=True`, the samples are randomly permuted before batching. This prevents the
model from learning spurious patterns from sample ordering. For validation data,
`shuffle=False` ensures consistent loss values across epochs. The loader also handles
splitting the dataset into mini-batches of the specified size.

#### 2.2.6 Checkpoint Operations (Lines 272–300)

```python
    def save_checkpoint(self, path, epoch, optimizer, val_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, path)
```

`model.state_dict()` returns an `OrderedDict` mapping parameter names to their
tensor values. This includes every learnable parameter in the model: LSTM weights,
embedding tables, output layer weights and biases. The optimizer state dict captures
Adam's running moment estimates ($m_t$ and $v_t$), enabling training to resume from
exactly where it left off.

`torch.save` serialises the dictionary using Python's `pickle` protocol to a file.
The file format is PyTorch-specific (`.pt` extension). The saved file size depends on
the model's parameter count; for the default configuration (64 hidden, 2 layers), the
checkpoint is approximately 300 KB.

#### 2.2.7 Model Metadata — `save_model_metadata` and `load_model_metadata`

```python
def save_model_metadata(metadata: dict, path: str):
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
```

This function serialises the training run's metadata to a JSON file alongside
the checkpoint. The metadata dictionary includes:

- `run_schema_version`: Integer (currently 2) indicating which fields to expect.
  Older runs (v1) may lack fields like `input_size` or `feature_cols`, and the
  forecast endpoint uses the schema version to decide whether to infer architecture
  from the checkpoint or trust the metadata.
- `input_size`: The number of exogenous features the model was trained with.
- `hidden_size`, `num_layers`, `dropout`: Architecture hyperparameters.
- `context_length`, `horizon`: The sequence lengths used during training.
- `feature_cols`: The exact list of feature column names (e.g.
  `["volatility_20", "rsi_14", ...]`). This is critical: the forecast endpoint
  must reproduce the same features in the same order.
- `uses_dummy_features`: Boolean flag indicating whether the model was trained with
  a zero-filled dummy feature (input_size=1, no real features).
- `feature_config`: The complete feature configuration dictionary that was passed to
  `engineer_features()`.

The `default=str` argument in `json.dump` ensures that non-serialisable types (e.g.
numpy integers, datetime objects) are converted to strings rather than raising an
exception.

`load_model_metadata` reads this JSON file and returns the dictionary. If the file
does not exist, it returns an empty dictionary, triggering the fallback architecture
inference path in the forecast endpoint.

#### 2.2.8 Parameter Count Analysis

For the default configuration (hidden_size=64, num_layers=2, input_size=7,
embedding_dim=8), the total number of trainable parameters is:

**LSTM layer 0:**
- `weight_ih_l0`: $4 \times 64 \times 32 = 8{,}192$ (32 = 1 + 7 + 24)
- `weight_hh_l0`: $4 \times 64 \times 64 = 16{,}384$
- `bias_ih_l0`: $4 \times 64 = 256$
- `bias_hh_l0`: $4 \times 64 = 256$

**LSTM layer 1:**
- `weight_ih_l1`: $4 \times 64 \times 64 = 16{,}384$
- `weight_hh_l1`: $4 \times 64 \times 64 = 16{,}384$
- `bias_ih_l1`: $256$
- `bias_hh_l1`: $256$

**Embeddings:**
- Symbol: $10 \times 8 = 80$
- Timeframe: $5 \times 8 = 40$
- Asset type: $3 \times 8 = 24$

**Output heads:**
- `mu_layer`: $64 \times 1 + 1 = 65$
- `sigma_layer`: $64 \times 1 + 1 = 65$
- `nu_layer`: $64 \times 1 + 1 = 65$

**Total:** approximately 58,700 parameters. This is a small model by modern
deep learning standards (GPT-3 has 175 billion parameters). The small size is
intentional: with only a few thousand training sequences, a larger model would
overfit catastrophically. The effective capacity is further limited by dropout
and early stopping.

#### 2.2.9 The Complete Training Pipeline in Pseudocode

```
INITIALISE model with architecture from feature count
INITIALISE Adam optimiser (lr=1e-3, weight_decay=1e-5)
INITIALISE LR scheduler (reduce on plateau, factor=0.5)

FOR epoch = 1 to 50:
    SET model to train mode
    FOR each mini-batch (past_target, past_features, future_target):
        UNSQUEEZE targets to add trailing dim if needed
        MOVE all tensors to device (CPU or GPU)
        
        FORWARD PASS:
            Embed categorical IDs → tile to sequence length
            Concatenate [target, features, embeddings]
            Run LSTM over context → get hidden states
            Continue LSTM over future (teacher forcing) → get future hidden states
            Concatenate context + future hidden states
            Apply three output heads → (μ, σ, ν) for every timestep
        
        COMPUTE LOSS:
            Concatenate context + future targets
            Compute Student's t log-likelihood for every (target, μ, σ, ν) pair
            Loss = -mean(log-likelihood)
        
        BACKWARD PASS:
            Zero all parameter gradients
            Backpropagate loss through computation graph
            Clip gradient norm to max 10.0
            Apply Adam update to all parameters
        
        ACCUMULATE epoch loss
    
    SET model to eval mode
    COMPUTE validation loss (same as training but no backward pass)
    
    STEP the LR scheduler with validation loss
    
    IF validation loss improved:
        SAVE checkpoint (model weights, optimiser state, epoch, loss)
        RESET patience counter
    ELSE:
        INCREMENT patience counter
        IF patience counter >= 10:
            STOP training early
    
    LOG epoch number, train loss, val loss, learning rate

RETURN best checkpoint path
```

### 2.3 API Integration — `app_api/main.py`

#### 2.3.1 `/train` Endpoint — Sequence Building and Model Instantiation

The training endpoint orchestrates the full pipeline. Key ML-relevant sections:

```python
        model = DeepARStudentT(
            input_size=train_past_features.shape[-1],
            hidden_size=request.hidden_size or hyperparams.get("hidden_size", 64),
            ...
        )
```

The `input_size` is derived dynamically from the feature tensor's last dimension.
This ensures the model's architecture matches the actual data. If the model were
instantiated with a different `input_size` than the data provides, the LSTM would
expect a different input width, producing a shape mismatch error during the forward
pass.

```python
        stored_hyperparams = {
            "run_schema_version": 2,
            "input_size": actual_input_size,
            "hidden_size": actual_hidden,
            ...
            "feature_cols": feature_cols if feature_cols else [],
            "uses_dummy_features": uses_dummy,
        }
```

The `run_schema_version` distinguishes between old runs (v1, missing many fields) and
new runs (v2, complete metadata). This metadata is critical for the forecast endpoint:
it must reconstruct the exact same model architecture and feature pipeline. Without
it, the forecast would fail with architecture mismatches or produce incorrect results
due to feature pipeline differences.

#### 2.3.2 `/forecast` Endpoint — Architecture Inference from Checkpoint

When old runs lack complete metadata, the forecast endpoint reverse-engineers the
architecture from the saved weights:

```python
        ih_shape = state_dict["rnn.weight_ih_l0"].shape   # [4*H, rnn_input]
        hh_shape = state_dict["rnn.weight_hh_l0"].shape   # [4*H, H]
        inferred_hidden = int(hh_shape[1])
        n_layers = sum(1 for k in state_dict if k.startswith("rnn.weight_ih_l"))
        rnn_input_total = int(ih_shape[1])
        emb_total = 3 * 8
        inferred_input_size = max(rnn_input_total - 1 - emb_total, 1)
```

`weight_ih_l0` has shape `(4 * hidden_size, rnn_input_size)` for an LSTM (the 4 comes
from the four gates). `weight_hh_l0` has shape `(4 * hidden_size, hidden_size)`. From
these shapes, `hidden_size` and `rnn_input_size` can be recovered. The number of
layers is the count of `weight_ih_l{N}` keys in the state dict. The `input_size` is
then `rnn_input_size - 1 - 24` (subtracting the target dimension and three embeddings
of dimension 8 each), clamped to at least 1.

This inference mechanism is essential for backward compatibility. When the project was
first deployed, metadata JSON files did not record `input_size` or `hidden_size`.
Without this fallback, all old training runs would be unusable. The inference is exact
(no approximation) because PyTorch's weight tensor shapes fully determine the
architecture.

#### 2.3.3 Feature Pipeline Reproduction at Forecast Time

The forecast endpoint must reproduce the exact same feature pipeline that was used
during training. Any difference — a missing feature column, a different window size,
a different normalisation — would cause feature misalignment, meaning the model
receives inputs drawn from a different distribution than it was trained on.

The relevant code path:

1. Load `feature_config` from metadata.
2. Fetch the most recent `context_length` OHLCV bars from SQLite.
3. Call `engineer_features(df, feature_config)` with the stored config.
4. Extract the feature columns in the same order as during training.
5. Build the context tensor with `create_sequences` (using context_length from metadata).

Several safety guards validate the pipeline:

- If the number of feature columns in the regenerated data does not match
  `metadata["input_size"]`, a warning is logged and the system falls back to
  zero-filled dummy features.
- If `uses_dummy_features` is True in metadata, the system skips feature regeneration
  entirely and creates a zeros tensor of the correct size.
- If no feature config is stored (old runs), all features are enabled by default.

#### 2.3.4 Tensor Preparation and Model Loading

```python
past_features_tensor = torch.FloatTensor(past_features).unsqueeze(0)
past_target_tensor = torch.FloatTensor(past_target).unsqueeze(0).unsqueeze(-1)
```

The `.unsqueeze(0)` adds a batch dimension (the forecast always produces a single
forecast for a single symbol). The `.unsqueeze(-1)` adds the trailing feature
dimension to the target tensor. The resulting shapes are:

- `past_target_tensor`: `(1, context_length, 1)`
- `past_features_tensor`: `(1, context_length, input_size)`

The model is then instantiated with the inferred or stored architecture, its weights
loaded from the checkpoint, and `model.sample()` is called to generate forecast
samples.

#### 2.3.5 Response Formatting

The sample tensor `(1, n_samples, horizon, 1)` is reduced to quantiles, and each
quantile is converted to a Python list for JSON serialisation. The API response
includes:

- `forecast_values`: The median (50th percentile) log-returns.
- `quantiles`: A dictionary mapping quantile levels to log-return sequences.
- `timestamps`: The future timestamps (computed by adding `horizon` intervals to
  the last observation's timestamp).
- `metadata`: A dictionary containing the model configuration, for debugging.

---

## Chapter 3 — Feature Engineering Theory

This chapter expands on the feature engineering pipeline, providing the mathematical
foundations for each derived feature and a formal treatment of data leakage.

### 3.1 Data Leakage — Formal Definition

**Data leakage** occurs when the model's training, validation, or inference process
has access to information that would not be available at prediction time in production.
Formally, if the model is making a prediction at time $t$, any input feature $x_t^{(j)}$
must satisfy:

$$
x_t^{(j)} = f(y_{t-1}, y_{t-2}, \ldots, y_1)
$$

That is, $x_t^{(j)}$ may depend only on observations **strictly before** time $t$.
Using $y_t$ (the current observation) or $y_{t+k}$ (future observations) in the
computation of $x_t^{(j)}$ constitutes leakage.

Leakage is insidious because it produces artificially good training and validation
metrics — the model learns to "cheat" by extracting information from the future — but
fails catastrophically in production where future data is not available. In the worst
case, the model may achieve near-perfect accuracy on historical backtests while
providing no predictive value on live data.

### 3.2 The Shift-by-One Rule

In this codebase, every rolling feature follows the same pattern:

```python
feature_value = rolling_statistic.shift(1).fillna(default)
```

The `.shift(1)` operation moves each value forward by one position: the value at index
$t$ is replaced by the value at index $t-1$. This ensures that the feature at time $t$
uses data only up to time $t-1$.

Without the shift, a rolling window of size $w$ centered at index $t$ would include
the observation $y_t$ itself. A rolling standard deviation, for example, would use
$y_t$ to compute the volatility that is supposed to help **predict** $y_t$. The model
would learn that high volatility features are associated with high absolute returns
at the same timestep — but this is not a useful prediction, merely a tautology.

The `.fillna(default)` call handles the boundary: after shifting, the first row has no
predecessor and produces NaN. Each feature uses a domain-appropriate default:
- Volatility: 0.0 (no volatility information)
- Mean return: 0.0 (neutral)
- RSI: 50.0 (neutral, neither overbought nor oversold)
- MACD: 0.0 (no signal)
- Volume z-score: 0.0 (average volume)

### 3.3 Log-Returns — `compute_log_returns()`

```python
df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
df["log_return"] = df["log_return"].fillna(0.0)
```

The log-return $r_t = \ln(P_t / P_{t-1})$ is computed using numpy's vectorised `log`
function. The `shift(1)` in the denominator is intrinsic to the definition: $P_{t-1}$
is the previous price. This is not a leakage-prevention shift — it is the natural
structure of the return computation. The first row has no predecessor ($P_0$ is not
defined), so its return is NaN, filled with 0.0 (zero return).

### 3.4 Rolling Volatility — `compute_rolling_volatility()`

**Mathematical definition**: The rolling realised volatility over a window of size $w$
at time $t$ is the sample standard deviation of the $w$ most recent returns:

$$
\hat{\sigma}_t = \sqrt{\frac{1}{w-1} \sum_{i=1}^{w} (r_{t-i+1} - \bar{r})^2}
$$

where $\bar{r}$ is the mean of the returns in the window. In code:

```python
vol = df[return_col].rolling(window=window, min_periods=1).std()
df[f"volatility_{window}"] = vol.shift(1).fillna(0.0)
```

The `min_periods=1` parameter allows the rolling window to produce results even when
fewer than $w$ observations are available (at the start of the series). The result is
then shifted by 1 to prevent leakage.

**Why volatility might help**: Financial returns exhibit **volatility clustering** —
periods of high volatility tend to be followed by more high volatility, and calm
periods tend to persist. This is the ARCH effect (Engle, 1982). Providing the model
with a recent volatility estimate gives it direct information about the current
regime, potentially improving both the location ($\mu$) and scale ($\sigma$)
predictions.

**Why it might fail**: Rolling volatility is a backward-looking statistic. If the
regime changes abruptly (e.g. a flash crash), the rolling volatility will take $w$
periods to catch up. The model may also learn to simply copy the volatility feature
into its $\sigma$ prediction without learning any additional structure.

### 3.5 Rolling Mean Return — `compute_rolling_mean_return()`

**Mathematical definition**:

$$
\bar{r}_t = \frac{1}{w} \sum_{i=1}^{w} r_{t-i+1}
$$

This is a simple short-term momentum indicator. If recent returns have been positive,
the rolling mean is positive, suggesting (under a momentum hypothesis) that near-future
returns may also be positive.

**Why it might help**: Momentum is one of the most robust empirical regularities in
financial markets (Jegadeesh and Titman, 1993). Assets that have risen recently tend
to continue rising in the short term.

**Why it might fail**: Momentum reverses over longer horizons and is highly vulnerable
to regime changes. The signal-to-noise ratio of momentum in hourly data is extremely
low.

### 3.6 RSI — Relative Strength Index

The Relative Strength Index (Wilder, 1978) is a bounded oscillator in $[0, 100]$
that measures the magnitude of recent gains versus recent losses.

**Formula**:

$$
\text{RSI} = 100 - \frac{100}{1 + RS}
$$

where:

$$
RS = \frac{\text{Average Gain over } w \text{ periods}}{\text{Average Loss over } w \text{ periods}}
$$

The average gain is computed using an exponentially weighted moving average (EWMA) of
positive returns, and the average loss is the EWMA of absolute negative returns. The
`ta` library's `RSIIndicator` class handles this computation.

When RSI is above 70, the asset is considered "overbought" (potentially due for a
pullback). Below 30, it is "oversold" (potentially due for a bounce). The neutral
value is 50, which is used as the fill default.

**Leakage prevention**: The RSI at time $t$ uses prices up to and including $P_t$.
The `.shift(1)` ensures the model at forecast time $t$ only sees the RSI computed from
prices up to $P_{t-1}$.

### 3.7 MACD — Moving Average Convergence Divergence

MACD (Appel, 1979) is a trend-following indicator based on the difference between two
exponential moving averages (EMAs) of different periods.

**Exponential Moving Average (EMA)**: The EMA of a series $x_t$ with smoothing factor
$\alpha = 2/(w+1)$ is:

$$
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

This is a first-order infinite impulse response (IIR) filter. By recursively unrolling
the recurrence, we can express EMA as a weighted sum of all past observations:

$$
\text{EMA}_t = \alpha \sum_{k=0}^{t-1} (1-\alpha)^k x_{t-k}
$$

The weights $(1-\alpha)^k$ decay exponentially, hence the name. Recent observations
have more influence than distant ones, with the effective "half-life" of the weights
equal to $\ln(2)/\ln(1/(1-\alpha)) \approx (w-1)/2$ periods.

For the standard MACD parameters:

- Fast EMA ($w = 12$): $\alpha_{\text{fast}} = 2/13 \approx 0.154$, half-life $\approx 5.5$ periods.
- Slow EMA ($w = 26$): $\alpha_{\text{slow}} = 2/27 \approx 0.074$, half-life $\approx 12.5$ periods.
- Signal EMA ($w = 9$): $\alpha_{\text{signal}} = 2/10 = 0.2$, half-life $\approx 4$ periods.

**MACD components**:

1. **MACD line** $= \text{EMA}_{12}(\text{close}) - \text{EMA}_{26}(\text{close})$
   — the difference between a fast (12-period) and slow (26-period) EMA. When the
   MACD line is positive, the fast EMA is above the slow EMA, indicating upward
   momentum. The magnitude reflects the strength of the momentum.

2. **Signal line** $= \text{EMA}_9(\text{MACD line})$ — a smoothed version of the
   MACD line. The signal line lags the MACD line because it applies an additional
   layer of smoothing.

3. **MACD histogram** $= \text{MACD line} - \text{Signal line}$ — the difference
   between the MACD line and the signal line. This is the "acceleration" of the
   momentum: when the histogram is positive and increasing, momentum is accelerating
   upward.

When the MACD line crosses above the signal line, it is a bullish signal (the fast EMA
is gaining momentum relative to the slow EMA). A cross below is bearish.

**Why three MACD columns?** The model receives all three (macd, macd_signal,
macd_diff) as separate features because they carry different information content.
The MACD line measures the current momentum state. The signal line is a filtered
version that reduces noise. The histogram (macd_diff) measures the rate of change of
momentum, which may predict reversals.

**Critical implementation detail — all three are shifted by 1.** The MACD at time $t$
uses the close price at time $t$ in its EMA computation. Without the shift, the model
at forecast time $t$ would have access to the current close price through the MACD
features, which is the very quantity it is trying to predict (since the target is
$\ln(P_t/P_{t-1})$, and knowing $P_t$ is equivalent to knowing the return). The
shift ensures the MACD at time $t$ uses data only up to time $t-1$.

### 3.8 Volume Z-Score

**Definition**:

$$
z_t = \frac{V_t - \bar{V}_w}{\hat{\sigma}_{V,w} + \epsilon}
$$

where $V_t$ is the volume at time $t$, $\bar{V}_w$ is the rolling mean volume over
$w$ periods, $\hat{\sigma}_{V,w}$ is the rolling standard deviation, and
$\epsilon = 10^{-8}$ prevents division by zero.

The z-score normalises volume to a dimensionless quantity with (approximately) zero mean
and unit variance. Large positive values indicate unusually high volume, which often
accompanies significant price moves (breakouts, panics). Large negative values
indicate unusually low volume, which may signal consolidation or low liquidity.

#### 3.8.1 Deriving the Z-Score Step by Step

The rolling mean volume over a window of $w$ periods ending at time $t$ is:

$$
\bar{V}_w(t) = \frac{1}{w}\sum_{i=0}^{w-1} V_{t-i}
$$

The rolling standard deviation is:

$$
\hat{\sigma}_{V,w}(t) = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1}\left(V_{t-i} - \bar{V}_w(t)\right)^2}
$$

The z-score then measures how many standard deviations the current volume is away
from the rolling mean:

$$
z_t = \frac{V_t - \bar{V}_w(t)}{\hat{\sigma}_{V,w}(t) + \epsilon}
$$

If volume were normally distributed (it is not, but the approximation is useful for
interpretation), $|z_t| > 2$ would indicate an event occurring less than 5% of the
time, and $|z_t| > 3$ would indicate less than 0.3%. In practice, volume
distributions have heavy right tails (occasional bursts of very high volume), so
large positive z-scores are more common than the Gaussian approximation predicts.

#### 3.8.2 The Epsilon Guard

The epsilon value of $10^{-8}$ is necessary because the rolling standard deviation
can be exactly zero. This occurs when all $w$ volume values in the window are
identical — rare in practice but possible during periods of extremely low trading
activity or when the data source reports zero volume for multiple consecutive periods
(which can happen for newly listed assets or during exchange outages). Without epsilon,
the division would produce $\pm\infty$, which would propagate through all subsequent
operations and corrupt the model's gradients.

The choice of $10^{-8}$ is small enough to have no measurable effect on the z-score
when $\hat{\sigma}_{V,w}$ is non-zero (any reasonable standard deviation of volume is
at least an order of magnitude larger) but large enough to prevent `inf` values.

#### 3.8.3 Why Volume Might Help the Model

Volume carries information that price alone does not. Specifically:

- **Breakouts with volume confirmation** are more likely to persist than breakouts
  on low volume. A large price move on high volume suggests broad market participation,
  while the same move on low volume may be due to thin liquidity and is more likely
  to reverse.

- **Volume-volatility correlation**: High volume is empirically associated with high
  volatility. Providing the volume z-score alongside the volatility feature gives the
  model two partially independent signals about the current market state.

- **Accumulation and distribution patterns**: Sustained high volume during a price
  advance suggests institutional buying (accumulation), which may predict continued
  upward pressure. Conversely, high volume during a decline suggests selling pressure
  (distribution).

#### 3.8.4 Why Volume Might Fail as a Feature

- Volume patterns vary dramatically across asset types. Cryptocurrency markets trade
  24/7 with no official close, while equity markets have distinct intraday volume
  patterns (U-shaped, with high volume at open and close). The model receives no
  explicit information about market hours.

- Volume data quality varies. Some exchanges report inflated volumes (wash trading
  is a known issue in cryptocurrency markets). The z-score normalisation partially
  mitigates this by comparing current volume to recent history, but if the reporting
  changes discontinuously, the z-score will spike incorrectly.

### 3.9 Numerical Stability Considerations

Several choices in the feature engineering pipeline are motivated by numerical
stability:

1. **`pd.to_numeric(df[col], errors="coerce")`**: Forces columns to numeric dtype.
   Data from external APIs sometimes arrives with string values or mixed types. Without
   this coercion, numpy operations would fail with "cannot convert object to float"
   errors. The `errors="coerce"` parameter converts unparseable values to NaN rather
   than raising an exception, which is then handled by the downstream `fillna` logic.

2. **The epsilon in volume z-score**: Detailed in Section 3.8.2.

3. **`fillna(0.0)` after shift**: The shift creates NaN at the boundary, and NaN
   propagates through all arithmetic operations (any computation involving NaN
   produces NaN). Filling with zero ensures clean input to the model. The choice of 0
   is not arbitrary: for most features, zero represents "no information" or "neutral".
   For RSI, the fill value is 50 (the midpoint of the 0–100 range), which represents
   neutral momentum.

4. **`ffill().bfill().fillna(0)`**: The triple fill strategy in `engineer_features`
   addresses three scenarios: (a) `ffill` carries the last valid value forward through
   gaps; (b) `bfill` fills remaining NaN at the start of the series using the first
   valid value; (c) `fillna(0)` catches any columns that are entirely NaN (e.g. a
   feature that could not be computed because the prerequisite column was missing).

5. **`min_periods=1` in rolling operations**: Without this parameter, the rolling
   window returns NaN for the first $w-1$ rows (where the full window is not yet
   available). With `min_periods=1`, partial windows are allowed: the first row uses
   only one observation, the second uses two, etc. This produces less accurate
   statistics at the start of the series but avoids NaN propagation.

### 3.10 `create_sequences` — Sliding Window Construction

The `create_sequences` function converts a time-aligned DataFrame into the
$(N, \text{context\_length}, \text{features})$ tensor format required by the model.

For each position $i$ in the DataFrame:

- **Context window**: rows $[i, i + \text{context\_length})$ — the model's input.
- **Prediction window**: rows $[i + \text{context\_length}, i + \text{context\_length} + \text{horizon})$ — the model's target.

There is zero overlap between context and prediction windows for the same sample.
Consecutive samples overlap by $\text{context\_length} + \text{horizon} - \text{stride}$
rows, with stride defaulting to 1. This maximum overlap produces the largest possible
training set from a fixed dataset, which is important when data is limited (e.g.
6 months of hourly data = ~4,300 rows, yielding ~4,100 samples with context 168 and
horizon 24).

#### 3.10.1 Mathematical Formulation of the Sliding Window

Given a time series of length $L$, context length $C$, horizon $H$, and stride $S$,
the number of valid samples is:

$$
N = \left\lfloor\frac{L - C - H}{S}\right\rfloor + 1
$$

For the default configuration ($C = 168$, $H = 24$, $S = 1$) with $L = 4{,}300$:

$$
N = \left\lfloor\frac{4300 - 168 - 24}{1}\right\rfloor + 1 = 4{,}109
$$

Each sample consists of a context window and a corresponding prediction window. The
$i$-th sample (0-indexed) covers:

- Context: rows $[iS, iS + C)$
- Prediction: rows $[iS + C, iS + C + H)$

When $S = 1$, consecutive samples share $C + H - 1$ rows. This creates strong
correlation between adjacent samples, which is acceptable for training (the
mini-batch SGD procedure and shuffling mitigate the effects) but must be considered
when estimating the effective sample size for statistical tests.

#### 3.10.2 Implementation Details

The resulting arrays have the following dimensions:

- `past_target`: shape $(N, \text{context\_length})$, dtype `float32`
- `past_features`: shape $(N, \text{context\_length}, n_{\text{features}})$, dtype `float32`
- `future_target`: shape $(N, \text{horizon})$, dtype `float32`

When no feature columns are provided, a placeholder of shape
$(N, \text{context\_length}, 1)$ filled with zeros is created. This ensures
`input_size` is never 0, which would break the LSTM (see Section 2.1.4).

The dtype enforcement is explicit and critical. The code casts all arrays to `float32`
using `.astype(np.float32)`. Without this cast, pandas DataFrames may produce `float64`
arrays (the numpy default), which would cause a dtype mismatch when PyTorch creates
`FloatTensor` (which expects `float32`). The mismatch does not cause a Python error
(PyTorch silently converts), but it creates `float64` tensors that consume twice the
memory and are incompatible with models initialised in `float32`.

#### 3.10.3 The Dummy Feature Guard

When the feature column list is empty (the user disabled all optional features), the
target is the only signal. However, the LSTM expects an input tensor of width
$1 + \text{input\_size} + 24$, and the model stores `self.input_size`. If `input_size`
were 0, several downstream operations would fail:

1. Concatenation with a 0-wide tensor is valid in numpy but produces a tensor where
   the feature dimension vanishes, making indexing ambiguous.
2. The saved metadata would record `input_size=0`, and the forecast endpoint would
   attempt to create a `(1, context, 0)` tensor, which has 0 elements.
3. The LSTM's `weight_ih_l0` would have `rnn_input_size = 1 + 0 + 24 = 25`, but
   at inference time the concatenation would produce width 25 only if the zero-dim
   feature tensor is correctly handled. This is fragile.

The dummy feature guard creates a `(N, context, 1)` zeros tensor and sets
`input_size=1`, providing a clean, uniform code path regardless of feature selection.

### 3.11 `split_by_time` — Temporal Train/Validation/Test Split

Traditional machine learning uses random splits. For time series, random splits
create future data leaking into the training set: a model trained on data from Monday
and Wednesday would be evaluated on Tuesday, which it has implicitly seen context
around. The temporal split enforces strict chronological ordering:

- **Train**: first 70 % of the data (by time).
- **Validation**: next 15 %.
- **Test**: final 15 %.

The assertion guards verify that the maximum timestamp in the training set is strictly
less than the minimum timestamp of the validation set, and similarly for
validation/test. If data were accidentally shuffled before splitting, these assertions
would catch it immediately.

#### 3.11.1 Why These Specific Proportions?

The 70/15/15 split is a common default, but the rationale deserves explanation:

- **70% training**: Enough data for the model to learn temporal patterns. Reducing this
  below ~60% risks underfitting because the model sees too few examples.
- **15% validation**: Used for early stopping and hyperparameter selection. It must be
  large enough to produce a stable loss estimate (at least several hundred sequences)
  but not so large that it steals from the training set.
- **15% test**: The gold-standard evaluation set that is never used during training or
  hyperparameter selection. The test set provides an unbiased estimate of the model's
  true out-of-sample performance.

In the training endpoint code, when the dataset is small (fewer than ~500 sequences),
the split logic may adjust to ensure each set has a minimum number of samples. The
validation and training sets are required to have at least enough samples to form one
full mini-batch.

### 3.12 `get_feature_columns` — Feature Column Discovery

```python
def get_feature_columns(df, target_col="log_return"):
    exclude = {"open", "high", "low", "close", "volume", "timestamp", target_col, ...}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
```

This function discovers which columns in the DataFrame are exogenous features by
excluding known non-feature columns (OHLCV, timestamp, the target) and non-numeric
columns (e.g. string symbol identifiers). The resulting list determines the
`input_size` parameter and the order in which features are stacked into the feature
tensor.

The order of columns in this list is deterministic (matching the DataFrame column
order), which is critical: if the order changed between training and inference, the
model would receive volatility where it expects RSI, producing garbage predictions
with no error message.

### 3.13 Feature Scaling and Normalisation

This codebase does NOT apply explicit feature scaling (standardisation, min-max
normalisation, etc.) to the exogenous features. Each feature is used in its raw
computed form:

- Log-returns: typically in $[-0.1, +0.1]$ for hourly data.
- Volatility: typically in $[0, 0.05]$.
- Rolling mean return: typically in $[-0.01, +0.01]$.
- RSI: in $[0, 100]$ (much larger scale than other features).
- MACD components: typically in $[-1, +1]$ for crypto, but can vary.
- Volume z-score: typically in $[-3, +3]$.

The lack of scaling means that features with larger absolute values (RSI, ranging
0–100) will have larger influence on the gradient updates than features with smaller
values (volatility, ranging 0–0.05). This is partially compensated by Adam's
adaptive learning rates (see Section 1.18), which normalise gradients per-parameter.
However, explicit feature standardisation (subtracting the mean and dividing by the
standard deviation) would provide more uniform gradient magnitudes and potentially
faster convergence.

Standardisation was not implemented because it introduces an additional complexity:
the mean and standard deviation must be computed on the training set only (to prevent
leakage from validation/test), stored alongside the model, and applied at inference
time. This creates another point of failure in the metadata pipeline.

---

## Chapter 4 — Financial Forecasting Interpretation

### 4.1 From Log-Return Forecasts to Price Forecasts

The model outputs log-return forecasts $(\hat{r}_{T+1}, \ldots, \hat{r}_{T+H})$.
To convert these into price forecasts, we use the exponential cumulative sum:

$$
\hat{P}_{T+k} = P_T \cdot \exp\!\left(\sum_{i=1}^{k} \hat{r}_{T+i}\right)
$$

In the code (from `plot_helpers.py`):

```python
prices = last_close * np.exp(np.cumsum(log_returns))
```

`np.cumsum` computes the running sum $S_k = \sum_{i=1}^{k} r_i$. `np.exp(S_k)` gives
the cumulative return factor. Multiplying by the last observed close price $P_T$
produces the price trajectory.

For a single sample path, this produces a deterministic price trajectory. The
uncertainty comes from having $N$ different sample paths, each with different sampled
log-returns. The fan chart on the dashboard shows the median trajectory (50th
percentile across sample paths) and various prediction intervals (80%, 95%) obtained
from percentiles across sample paths at each horizon step.

### 4.2 Why Uncertainty Explodes in Price Space

Consider the variance of the cumulative log-return at horizon $k$:

$$
\text{Var}\!\left(\sum_{i=1}^{k} r_{T+i}\right)
$$

If the log-returns were independent with common variance $\sigma_r^2$, this would be
$k \sigma_r^2$ — variance grows linearly with horizon. But the relationship between
log-return and price is exponential, so the price forecast variance grows roughly as:

$$
\text{Var}(\hat{P}_{T+k}) \approx P_T^2 \exp(2k\mu_r)\left[\exp(k\sigma_r^2) - 1\right]
$$

(under a Gaussian log-return assumption). The term $\exp(k\sigma_r^2) - 1$ grows
exponentially with $k$. This means:

- At short horizons (1–5 steps), prediction intervals are tight.
- At medium horizons (10–50 steps), intervals widen noticeably.
- At long horizons (100+ steps), intervals can span an enormous range.

This is not a defect of the model — it is the mathematical reality of forecasting a
nonlinear transformation of an uncertain process. Wide prediction intervals at long
horizons are **expected and correct**. A model that shows narrow intervals at long
horizons is likely miscalibrated (overconfident).

### 4.3 Types of Uncertainty

There are three distinct sources of uncertainty in the system:

**1. Aleatoric uncertainty (irreducible noise)**: This is the inherent randomness of
financial markets. Even with a perfect model, future returns are genuinely random. This
uncertainty is captured by the $\sigma$ and $\nu$ parameters of the Student's t
distribution.

**2. Epistemic uncertainty (model uncertainty)**: This arises from the model's limited
capacity and finite training data. A different random seed, a different training set,
or a different architecture would produce a different model with different predictions.
Epistemic uncertainty is not explicitly quantified by a single DeepAR model (which
produces a fixed set of parameters for a given input), but it is implicitly reflected
in the width of prediction intervals when the model is uncertain.

**3. Estimation error**: The parameters of the model ($\theta$) are estimated from a
finite training set. The optimiser may not have found the global optimum, and the
training set may not be representative of future market conditions. This uncertainty
is not captured by the model at all.

### 4.4 Interpreting Fan Charts

The forecast visualisation displays:

- **Steel-blue line**: Historical observed prices.
- **Teal dashed line**: Median forecast (50th percentile of sample paths).
- **Darker shaded area**: 80 % prediction interval (10th to 90th percentile).
- **Lighter shaded area**: 95 % prediction interval (2.5th to 97.5th percentile).

The correct interpretation is: if the model is well calibrated, approximately 80 % of
future realisations should fall within the darker band, and approximately 95 % within
the lighter band. Calibration can be checked empirically by running backtests and
measuring coverage — the fraction of actual values that fall within the predicted
intervals.

If the coverage is significantly less than the nominal level (e.g. only 60 % of
actuals fall within the 80 % band), the model is **overconfident** and the intervals
are too narrow. If the coverage is significantly more (e.g. 95 % of actuals fall
within the 80 % band), the model is **underconfident** and the intervals are too wide.

### 4.5 The Difference Between a Forecast and a Trading Signal

A probabilistic forecast answers the question: "What is the distribution of future
returns?" A trading signal answers the question: "Should I buy or sell?" These are
fundamentally different.

A forecast may predict a 55 % probability of positive returns with a median of +0.2 %.
Whether this constitutes a profitable trade depends on:

- Transaction costs (commissions, spreads)
- Slippage (the difference between the expected and actual execution price)
- Position sizing and risk management
- The asymmetry of the distribution (even with a positive median, the expected return
  can be negative if the left tail is fat)
- Correlation with existing portfolio positions

The DeepAR model is a **forecasting** tool, not a trading strategy. Treating its
output as a direct buy/sell signal without accounting for these factors will likely
result in losses.

### 4.6 Prediction Interval vs. Confidence Interval

A common source of confusion is the distinction between **prediction intervals** and
**confidence intervals**. The DeepAR model produces prediction intervals.

A **confidence interval** quantifies uncertainty about a population parameter (e.g.
the true mean return). A 95% confidence interval for $\mu$ means that if we repeated
the estimation procedure many times, 95% of the resulting intervals would contain the
true $\mu$. The width of the confidence interval shrinks as the sample size increases.

A **prediction interval** quantifies uncertainty about a future individual
observation. A 95% prediction interval for $y_{T+1}$ means that the next observation
has a 95% probability of falling within the interval (under the model's assumptions).
The width of the prediction interval does NOT shrink to zero with more data — even
with infinite training data and a perfect model, the future observation is still
random.

The prediction intervals produced by DeepAR are always wider than confidence intervals
would be because they must account for the inherent randomness of future returns
(aleatoric uncertainty) in addition to any model uncertainty. The irreducible
randomness sets a floor on the width of the prediction interval.

### 4.7 Calibration — What It Means and How to Check It

A probabilistic forecast is **calibrated** if its predicted probabilities match
observed frequencies. Concretely, if the model's 80% prediction interval captures
exactly 80% of future observations, the model is well calibrated at that level.

**Reliability diagram**: The standard tool for assessing calibration is a reliability
diagram (also called a calibration plot). For each nominal coverage level
$\alpha \in \{10\%, 20\%, \ldots, 90\%\}$:

1. Count the fraction of test observations that fall within the $\alpha$%
   prediction interval.
2. Plot observed coverage vs. nominal coverage.

A perfectly calibrated model lies on the diagonal (observed = nominal). If the curve
is below the diagonal, the model is **overconfident** (intervals too narrow). If
above, it is **underconfident** (intervals too wide).

**PIT histogram**: The Probability Integral Transform (PIT) provides a more detailed
calibration diagnostic. For each observation $y_t$, compute $u_t = F(y_t \mid \hat{\theta}_t)$,
where $F$ is the CDF of the predicted distribution. If the model is correctly
specified and well calibrated, $u_t \sim \text{Uniform}(0, 1)$. A histogram of the
PIT values should be approximately flat. Departures from flatness indicate specific
types of miscalibration:

- A U-shape (excess mass at 0 and 1) indicates underdispersion (too-narrow intervals).
- An inverted U (excess mass near 0.5) indicates overdispersion.
- A slope indicates bias in the location parameter.

### 4.8 The Log-Return to Price Conversion in Detail

The conversion from log-return forecasts to price forecasts involves an important
subtlety related to Jensen's inequality. If $r_{T+1} \sim \text{Student-t}(\mu, \sigma, \nu)$,
then $\hat{P}_{T+1} = P_T e^{r_{T+1}}$ has a **log-Student-t** distribution, not a
Student's t distribution in price space.

The median of $e^r$ is $e^{\text{median}(r)}$, so the median price forecast is
correctly computed as $P_T e^{\hat{r}_{50\%}}$. However, the **mean** of an
exponentiated random variable is not the exponential of the mean:

$$
E[e^r] = e^{E[r]} \cdot E[e^{r - E[r]}] \geq e^{E[r]}
$$

by Jensen's inequality (since $e^x$ is convex). This means the expected price is
**higher** than $P_T e^{\mu}$. The bias equals $e^{\sigma^2/2}$ for a Gaussian $r$,
and is even larger for a Student's t $r$ (heavier tails increase the expectation of
the exponential).

In practice, this bias is small for typical log-return magnitudes ($\sigma \approx 0.01$
gives a bias of $e^{0.00005} \approx 1.00005$, or 0.005%). At longer horizons where
cumulative $\sigma$ is larger, the bias becomes relevant. The code uses the median
(50th percentile) as the central forecast, which avoids this bias entirely.

### 4.9 Multi-Step Forecasting and Error Accumulation

The autoregressive sampling procedure generates multi-step forecasts by feeding each
sampled value back as input to the next step. This creates a fundamental tension
between two desirable properties:

**Temporal coherence**: The sample paths are internally consistent — each path evolves
according to the model's learned dynamics. If the model captures mean-reversion, the
sample paths will exhibit mean-reversion. If it captures momentum, the paths will
show momentum.

**Error accumulation**: Each sampled value introduces noise that propagates to all
subsequent steps. After $k$ steps of autoregressive generation, the effective noise
level is (very roughly) $\sigma\sqrt{k}$ for independent errors, but potentially
worse if errors are correlated.

The practical consequence is that the model's forecast quality degrades with horizon
length. For the first few steps, the forecast is driven primarily by the context
information encoded in the LSTM's hidden state. For later steps, the forecast is
increasingly driven by the random samples rather than the context. Eventually, the
forecast converges to the unconditional distribution of returns — the model has
"forgotten" the context and is effectively sampling from the marginal distribution.

The speed of this convergence depends on the LSTM's "memory horizon" — how many
timesteps of useful context the hidden state can retain. For a well-trained 2-layer
LSTM with hidden dimension 64, this is typically on the order of the context length
used during training (168 timesteps in the default configuration). Beyond this
horizon, the model has limited ability to distinguish different starting conditions.

### 4.10 Comparison with Naive Baselines

Any forecast model should be evaluated against simple baselines. The relevant baselines
for this system are:

**Random walk (no-change forecast)**: $\hat{r}_t = 0$ for all future $t$. This
predicts that the price will remain at its current level. In efficient markets, this
is a surprisingly strong baseline.

**Historical mean**: $\hat{r}_t = \bar{r}$ (the sample mean of training returns). For
short horizons, this is nearly identical to the random walk because $\bar{r}$ is very
close to zero.

**Historical distribution**: $\hat{r}_t \sim \text{Student-t}(\hat{\mu}, \hat{\sigma}, \hat{\nu})$
where $(\hat{\mu}, \hat{\sigma}, \hat{\nu})$ are estimated from the training data.
This captures the unconditional distribution shape but not the conditional dynamics.

A DeepAR model that cannot beat the historical distribution baseline has learned
nothing useful about the temporal dynamics of returns. A model that beats the random
walk on the median forecast has learned something about the conditional mean. A model
that produces better-calibrated prediction intervals than the historical distribution
has learned something about the conditional variance and/or tail behaviour.

---

## Chapter 5 — Diagnostics, Failure Modes, and Limitations

### 5.1 Training Loss Appears as Zero

After training, the logs may show:

```
Train Loss: 0.000000, Val Loss: 0.000000
```

This can have two very different causes:

**Cause 1 — Display rounding (benign)**: The NLL of a well-fitting Student's t
distribution for log-returns near zero is a small negative number (close to zero). When
displayed with `:.6f` formatting, values like $-0.0000004$ appear as `0.000000`.
This does not indicate a problem — it means the model's predicted distribution is
a reasonable fit to the data.

**Cause 2 — Degenerate solution (pathological)**: If the model learns to predict
$\sigma \to \infty$ (by driving sigma pre-activation weights to large positive
values), the NLL approaches $-\log\sigma \to -\infty$ (penalised) but the kernel
term approaches 0 (because $z = (y-\mu)/\sigma \to 0$). The net loss can be very
close to zero if these terms approximately cancel. This is a pathological local
minimum where the model has learned to be infinitely uncertain about everything,
which satisfies the loss but provides no predictive value.

**Diagnosis**: Check the predicted $\sigma$ values. If they are of order $10^{-2}$
to $10^{0}$ (comparable to the actual standard deviation of log-returns), the model is
healthy. If they are of order $10^{2}$ or larger, the model has collapsed to the
degenerate solution.

### 5.2 How Student's t Can Hide Poor Mean Predictions

The Student's t distribution has a peculiar property: with low $\nu$ (heavy tails),
even large deviations from $\mu$ incur a relatively small NLL penalty because the
tails are fat enough to accommodate them. A model that predicts $\mu = 0$ for every
timestep but learns the correct $\sigma$ and $\nu$ will achieve a loss that appears
reasonable, because log-returns are distributed approximately symmetrically around zero
anyway.

This means the NLL loss alone does not tell you whether the model has learned useful
temporal patterns in $\mu$. To diagnose this, examine:

1. **Directional accuracy**: What fraction of the time does the sign of $\mu$
   correctly predict the sign of the actual return?
2. **RMSE of the median forecast**: How close is the median forecast to reality?
3. **Comparison to a naive baseline**: Does the model outperform predicting
   $\mu = 0, \sigma = \text{historical\_std}, \nu = 5$ at every step?

### 5.3 Known Failure Modes of DeepAR in Finance

**1. Regime changes**: The LSTM learns patterns from the training data's regime. If
the market enters a fundamentally different regime (e.g. from low-volatility trending
to high-volatility mean-reverting), the model's learned parameters are poorly
suited.

**2. Insufficient data**: With fewer than ~2,000 training sequences, the model may
overfit to idiosyncratic patterns. The early stopping mechanism mitigates this but
cannot compensate for a fundamentally too-small dataset.

**3. Exogenous feature overfitting**: The model has access to RSI, MACD, volatility,
and volume z-score. These features are derived from the same price series that
generates the target. If the model overfits to quirks in these features (e.g. RSI
crossing 70 in the training set always preceded a drop), it will fail on new data
where that pattern does not hold.

**4. Horizon-dependent degradation**: The model is trained with teacher forcing but
evaluated with autoregressive sampling. Error accumulates with each sampled step
because the model feeds its own (noisy) output back as input. At long horizons, the
model's predictions may be no better than the unconditional distribution.

**5. Non-stationarity of the target**: While log-returns are approximately stationary
over short windows, their distribution does change over time (volatility regimes,
structural breaks). The model assumes the patterns learned during training will
persist during the forecast window, which is not guaranteed.

### 5.4 What the Model Is Not Learning

The model is NOT learning:

- **Fundamental value**: The model has no concept of earnings, revenue, or fair value.
  It learns statistical patterns in return sequences, not economic fundamentals.
- **News or events**: There is no natural language input. The model cannot anticipate
  or react to news announcements, regulatory changes, or geopolitical events.
- **Market microstructure**: The model does not see order books, bid-ask spreads,
  or trade flow. It operates on aggregated OHLCV candles.
- **Cross-asset dependencies**: BTC and ETH are modelled independently. The model
  does not know that they are highly correlated.
- **Calendar effects**: There are no day-of-week or hour-of-day features. Intraday
  seasonality (e.g. higher volume during US market hours) is not explicitly modelled.

### 5.5 Why This Is Not a Trading Strategy

A forecast model produces probability distributions over future returns. Converting
these into profitable trades requires:

1. **A decision rule**: When to enter and exit positions, and in what size.
2. **Risk management**: Position sizing, stop losses, portfolio constraints.
3. **Transaction cost modelling**: Commissions, spreads, slippage, funding costs.
4. **Robustness testing**: Out-of-sample testing, walk-forward validation,
   sensitivity analysis to parameter choices.

The backtest module in this project provides a basic framework for testing decision
rules (threshold-based strategies with transaction costs), but it is a starting point,
not a production-ready trading system. The model's forecast accuracy, even if
statistically significant, may not survive the friction of real-world trading.

### 5.6 Hyperparameter Sensitivity

The model's behaviour is sensitive to several hyperparameters. Understanding these
sensitivities is important for diagnosing unexpected results.

**Hidden size (default: 64).** Increasing hidden size increases the model's capacity
to represent complex temporal patterns. However, with limited training data (a few
thousand sequences), increasing beyond 64–128 typically leads to overfitting: the
training loss decreases but the validation loss increases. The symptom is that the
model produces excellent in-sample predictions but poor out-of-sample forecasts. A
hidden size of 32 is suitable for very small datasets (< 1,000 sequences); 64 is
appropriate for moderate datasets (1,000–10,000); 128+ may help for large datasets
(> 10,000 sequences) or very complex patterns.

**Number of layers (default: 2).** Two layers provide a hierarchical representation
that one layer cannot. Three or more layers are rarely beneficial for this type of
time series data; the additional depth provides diminishing returns while increasing
the risk of vanishing gradients (despite the LSTM's mitigation).

**Context length (default: 168).** This determines how far back the model can look.
For hourly data, 168 hours = 7 days. Increasing the context beyond the effective
memory horizon of the LSTM wastes computation without improving predictions. Decreasing
it below the characteristic time scale of the patterns (e.g. daily seasonality
requires at least 24 hours of context for hourly data) prevents the model from
detecting those patterns.

**Learning rate (default: 1e-3).** Too high causes oscillation and training
instability (the loss jumps around rather than decreasing steadily). Too low causes
extremely slow convergence and may trigger early stopping before the model has
learned anything. The ReduceLROnPlateau scheduler partially mitigates mis-specification
by adapting the rate during training.

**Dropout (default: 0.1).** Dropout is applied between LSTM layers. Values of 0.1–0.3
are typical. Higher dropout (0.5+) can severely impair the LSTM's ability to propagate
information between layers. Zero dropout removes regularisation, increasing the risk
of overfitting.

**Gradient clipping (default: max_norm=10.0).** This prevents catastrophic gradient
explosions. A value of 10.0 is conservative — it allows fairly large gradients while
preventing extreme ones. Reducing this to 1.0 would limit the model's ability to
make large parameter adjustments in a single step, which could slow convergence.
Increasing it to 100 or removing it entirely risks occasional training collapses
when a pathological mini-batch produces extreme gradients.

### 5.7 Training Diagnostics Checklist

When a training run produces unexpected results, the following diagnostic checklist
can help identify the cause:

**1. Loss curve shape:**
- Smooth, monotonically decreasing → normal training.
- Jagged with large spikes → learning rate too high or gradient clipping too lax.
- Flat from the start → data may be constant (all zeros), learning rate too low, or
  bug in data pipeline.
- Decreasing then suddenly increasing → possible learning rate schedule issue or
  data corruption.

**2. Train vs. validation loss gap:**
- Small gap (train ≈ val) → good generalisation, model may be underfit.
- Large gap (train ≪ val) → overfitting. Remedies: increase dropout, reduce hidden
  size, add weight decay, collect more training data.
- Val < train → possible bug (validation data leaked into training, or validation
  set is easier than training set).

**3. Predicted parameter ranges:**
- $\mu$ should be of order $10^{-3}$ to $10^{-2}$ (typical log-return magnitudes).
  If $\mu$ is consistently zero, the model may have collapsed to predicting the
  unconditional mean.
- $\sigma$ should be of order $10^{-2}$ to $10^{-1}$ (matching the conditional
  standard deviation of log-returns). If $\sigma$ is very large ($> 1$), the model
  may be in a degenerate solution. If very small ($< 10^{-4}$), the model is
  overconfident.
- $\nu$ should be in the range 3–10 for financial data. If $\nu > 30$, the model is
  effectively Gaussian. If $\nu$ is very close to 2 (the floor), the model is
  predicting maximally heavy tails, which may indicate poor fit.

**4. Data quality checks:**
- Are there NaN or inf values in the features?
- Is the target constant (e.g. all zeros due to a missing data column)?
- Is the time ordering correct (chronological, no duplicates)?
- Is the volume data plausible (not all zeros, not all identical)?

### 5.8 Common Error Messages and Their Causes

**`RuntimeError: mat1 and mat2 shapes cannot be multiplied`:**
This occurs when the LSTM's input width does not match the weight matrix dimensions.
The most common cause is a mismatch between `input_size` at training time and
inference time. Check that the metadata `input_size` matches the actual feature count
in the input tensor.

**`RuntimeError: Expected all tensors to be on the same device`:**
Some tensors are on CPU while others are on GPU (or vice versa). This happens when
constants (like $\pi$) are created without `device=` specification. The model code
explicitly passes `device=y.device` when creating the $\pi$ tensor.

**`ValueError: cannot convert float NaN to integer`:**
NaN values have propagated through the feature pipeline into the model. This typically
indicates missing data that was not caught by the `fillna` guards. Check the raw data
for unexpected NaN values.

**`IndexError: index out of range in self (embedding)`:**
A categorical ID exceeds the embedding vocabulary size. If `num_symbols=10` and a
symbol is assigned ID 10, the embedding lookup fails because valid indices are 0–9.

### 5.9 Limitations of the Student's t Assumption

The Student's t distribution is symmetric around its location parameter $\mu$. This
means the model predicts equal probability of upward and downward deviations of the
same magnitude from $\mu$. In reality, financial return distributions often exhibit
**asymmetric tails** (negative skewness): large negative returns are more probable
than large positive returns of the same magnitude. The Student's t distribution cannot
capture this asymmetry.

Additionally, the Student's t assumes a single value of $\nu$ (tail thickness) at each
timestep. This means the left and right tails are equally heavy. In practice, the left
tail (large losses) may be thicker than the right tail (large gains). Models that
could capture this asymmetry include the skewed Student's t distribution, the
generalised hyperbolic distribution, or non-parametric approaches like normalising
flows.

The Student's t is nonetheless a significant improvement over the Gaussian for
financial data because it captures the **excess kurtosis** (both tails being heavier
than Gaussian) even though it cannot capture the **asymmetry**.

### 5.10 Cold Start and Cross-Series Transfer

The model is trained separately for each symbol. When a new symbol is added to the
system (e.g. a newly listed cryptocurrency), there is no trained model available. The
user must collect enough data to form at least a few hundred training sequences before
the model can be trained.

In the DeepAR paper, Salinas et al. advocate training a single global model across
all series, which would enable zero-shot forecasting for new series through the
embedding mechanism. This project's architecture supports this (the embedding layers
are in place), but the current training pipeline trains per-symbol. Extending to
global training would require:

1. A data loader that batches sequences from different symbols together, assigning the
   correct symbol ID to each sequence so the embedding layer can differentiate them.
2. Correct assignment of symbol IDs. Currently, all sequences use ID 0. A global model
   would need a consistent mapping from symbol strings (e.g. "BTC/USDT") to integer
   indices that persists across training runs.
3. Shared feature normalization across symbols (or per-symbol normalization stored in
   the metadata). Different assets have different volatility scales; without
   normalization, the model would be dominated by the most volatile asset.
4. Handling of different timeframes in the same batch. A global model could either
   restrict to a single timeframe (e.g. train only on hourly data from all symbols)
   or use the timeframe embedding to distinguish.

The cold start problem is distinct from the cross-series transfer problem. Even with
a global model, a brand-new asset with no historical data whatsoever cannot be
forecasted because there is no context window to encode. The minimum viable context
is one full context window (168 periods = 7 days of hourly data), plus enough
additional data to form at least a few training sequences for fine-tuning.

### 5.11 Overfitting Detection in Practice

Overfitting is the single most common failure mode for deep learning models trained
on small financial datasets. The following practical signs indicate overfitting:

**1. Validation loss divergence.** The most direct indicator. If the training loss
continues to decrease while the validation loss increases for multiple consecutive
epochs, the model is memorising training-set patterns that do not generalise. The
early stopping mechanism is designed to catch this, but if the patience is set too
high, the model may overfit severely before stopping.

**2. Overconfident predictions.** An overfit model may produce very narrow prediction
intervals (high certainty) because it has learned to fit the training data with
very small $\sigma$. On new data, these narrow intervals fail to capture actual
outcomes, leading to extremely poor coverage.

**3. Unstable forecasts.** Retraining the model with a different random seed produces
radically different forecasts. A robust model should be relatively stable across
seeds; large seed sensitivity indicates that the model is fitting noise rather than
signal.

**4. Feature importance collapse.** If the model assigns all predictive weight to a
single feature (identifiable by checking the magnitude of LSTM weights connected to
that feature's input dimension) while ignoring others, it may be overfitting to a
spurious correlation in that feature.

**Remediation strategies:**
- Reduce model capacity (hidden_size from 64 to 32, num_layers from 2 to 1).
- Increase dropout (from 0.1 to 0.2 or 0.3).
- Increase weight decay (from 1e-5 to 1e-4 or 1e-3).
- Collect more training data (longer historical window).
- Disable exogenous features (train on log-returns only).
- Reduce the number of training epochs (more aggressive early stopping patience).

### 5.12 Reproducibility Guarantees and Their Limits

The training pipeline sets random seeds for Python's random module, numpy, and
PyTorch. With the same seed, data, and hardware, training should produce identical
results. However, full reproducibility is not always achievable:

**Hardware non-determinism.** GPU floating-point operations may produce slightly
different results depending on the GPU architecture, driver version, and CUDA version.
The cuDNN library, which PyTorch uses for LSTM acceleration, has non-deterministic
kernel selection by default. Setting `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` enforces determinism at a performance cost.

**Library version differences.** Upgrading PyTorch, numpy, or scipy can change the
implementation of operations (e.g. a different sorting algorithm in `torch.quantile`),
producing floating-point differences that accumulate across thousands of operations.

**Platform differences.** Training on Windows vs. Linux may produce different results
due to differences in floating-point rounding modes or library implementations.

For all these reasons, the exact numerical values of a training run should not be
treated as ground truth. Reproducibility is a useful debugging tool (if two runs with
the same seed produce different results, something has changed in the environment),
but slight numerical differences across platforms are expected and acceptable.

### 5.13 Potential Improvements and Extensions

**Multi-horizon direct training:** Instead of training with teacher forcing and
evaluating with autoregressive sampling (creating a train-test discrepancy), train
the model to directly predict distributions at all horizon steps simultaneously.
This eliminates the exposure bias but sacrifices the temporal coherence of sample
paths.

**Attention mechanisms:** Replacing or augmenting the LSTM with self-attention
(Transformer-style) could improve the model's ability to capture long-range
dependencies. The Transformer architecture uses a self-attention mechanism where each
timestep attends to all other timesteps in the context window, with attention weights
learned from the data. The trade-off is computational cost ($O(T^2)$ vs. $O(T)$ for
the LSTM) and the requirement for positional encoding (since attention is permutation-
invariant, explicit position information must be injected).

**Cross-asset features:** Including correlated assets as features (e.g. BTC returns
as a feature when predicting ETH) could help capture cross-market dynamics. The
correlation structure of cryptocurrencies is well-documented: during risk-off events,
all cryptocurrencies tend to sell off together, and during risk-on events, they tend
to rally together. A model that sees only one asset at a time cannot capture this
correlated structure.

**Online learning:** Updating the model incrementally as new data arrives, rather
than retraining from scratch. This would allow the model to adapt to regime changes
more quickly. The challenge is catastrophic forgetting: incrementally trained models
may lose performance on older patterns as they adapt to new ones.

**Ensemble methods:** Training multiple models with different random seeds or
hyperparameters and combining their forecasts could improve both accuracy and
calibration by averaging out individual model biases. A simple ensemble of 5 models
with different seeds typically produces better-calibrated intervals than any
individual model.

**Normalising flow output layer:** Replace the Student's t distribution with a
normalising flow — a flexible learned distribution constructed by composing a sequence
of invertible transformations. This would allow the model to capture arbitrary
distribution shapes (asymmetric, multimodal, etc.) without being constrained to
the symmetric, unimodal Student's t. The cost is additional model complexity and
potential overfitting of the distribution shape to training data.

**Disclaimer**: This software is for research and educational purposes only.
Past performance does not guarantee future results. Trading involves substantial
risk of loss.

---

*End of manual.*
