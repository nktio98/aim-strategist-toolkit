# AIM Strategist Toolkit

Quant toolkit mirroring the core processes of an insurance investment
strategist role (Allianz Investment Management style): yield curve
modeling, regime detection, and ALM-aware stress testing.

## Modules

### 1. `yield_curve.py` — Dynamic Nelson-Siegel engine
- Cross-sectional NS fit per date (lambda by grid search), VAR(1) factor
  dynamics, h-step curve forecasts. In-sample fit ~2-3bp RMSE.
- Upgrade path: AFNS yield-adjustment term (Christensen-Diebold-Rudebusch),
  ACM term-premium decomposition on the same factor panel, shadow-rate
  extension for near-ZLB markets (JPY, TWD).

### 2. `regimes.py` — Regime detection
- `GaussianMS`: 2-state Markov-switching (Hamilton filter + EM), from scratch.
- `JumpModel`: statistical jump model (k-means + switch penalty, exact DP
  state assignment) — the modern buy-side alternative; produces more
  persistent, more tradable regimes and takes arbitrary feature vectors
  (vol, momentum, spread changes, ...).

### 3. `stress.py` — ALM stress engine
- Portfolio via key-rate durations, spread duration, equity beta, FX delta.
- Stylized life liability book -> duration gap and economic surplus
  sensitivity (the insurance lens: falling rates HURT when liab dur > asset dur).
- Named scenarios (taper tantrum, credit blowout, Asia FX crisis) with P&L
  decomposition by risk factor.
- Illustrative Solvency-style market-risk capital aggregation (correlation
  matrix square-root rule). NOT a regulatory calculation.
- Upgrade path: BVAR / GARCH-DCC Monte Carlo feeding the same revaluation
  function; entropy pooling (Meucci) for view-conditioned distributions.

### `data.py`
- `load_yield_csv(path)`: drop in real data (FRED, MAS, Bloomberg export;
  first column date, remaining columns maturities in years, yields in %).
- Simulators used by the demo so everything runs offline.

## Run

```bash
python3 run_demo.py       # console report + charts in outputs/
```

### 4. `fx.py` — FX analytics (insurance investor lens)
- Hedge-cost engine: covered interest parity + cross-currency basis;
  hedged yield pickup decision table (hedged USD credit vs local bonds
  per investor currency) -- the core Asian insurance allocation question.
- Fair-value engine: Engle-Granger cointegration (from-scratch ADF test)
  + error-correction model -> misvaluation, half-life, +/-2sd signal bands.
- Rolling minimum-variance hedge ratio (upgrade path: DCC-GARCH betas).
- Demo: `python3 run_fx_demo.py`

### 5. `taa.py` — TAA research with anti-overfitting machinery
- Signal library (momentum, value z-score, carry) + z-score positioning.
- PurgedKFold cross-validation (purging + embargo, Lopez de Prado).
- Backtester net of transaction costs.
- Probabilistic & Deflated Sharpe ratios: strategies only pass if the
  Sharpe survives correction for non-normality AND number of trials.
- Demo: `python3 run_taa_demo.py`

### 6. `managers.py` — Asset manager oversight
- Factor regressions with Newey-West (HAC) alpha t-stats.
- Benjamini-Hochberg FDR control across the manager panel (the fix for
  "1-in-20 managers looks skilled by luck").
- Rolling-beta style-drift / mandate-compliance monitor; appraisal
  metrics (IR, tracking error, hit rate).
- Demo: `python3 run_manager_demo.py`

### 7. `allocation.py` — View-conditioned allocation
- Black-Litterman (reverse-optimized equilibrium + views).
- Entropy pooling (Meucci): impose views on a full scenario distribution
  by minimum relative entropy — handles non-normal stress-engine
  scenarios and views on any moment. Effective-scenario diagnostic.
- Constrained long-only mean-variance optimizer (SLSQP).
- Demo: `python3 run_allocation_demo.py`

### 8. `dashboard.py` — Self-contained HTML dashboard
- Single shareable .html embedding every chart and table; no server.
- Build everything end-to-end: `python3 build_dashboard.py`

## Roadmap (extensions)
- Real data feeds (FRED/MAS/Bloomberg CSV drop-in via data.load_yield_csv).
- AFNS/ACM term premium; DCC-GARCH hedge ratios; BVAR scenario generator
  feeding entropy pooling; Streamlit/Power BI front end; LLM-drafted
  daily commentary layer.
