# Initial Margin — Filtered Historical Simulation

Calcul de la **Marge Initiale (IM)** pour des portefeuilles de produits de taux (obligations à taux fixe, repos, forward repos) selon une approche **Filtered Historical Simulation (FHS)** hybride avec composante de stress historique non rescalée. Conforme EMIR Art. 28, EMIR Art. 49, RTS 153/2013, BCBS-IOSCO et FRTB.

## Pipeline

1. **Prétraitement** — déduplication, tri, dropna sur la matrice de taux ZC.
2. **Facteurs de risque** — prix ZC `P(t,T) = N / (1+r(t,T))^T` et returns `R = P(t)/P(t-HP) - 1`.
3. **Volatilité EWMA** — `σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}`.
4. **Scénarios FHS** — rescaling symétrisé `R̃ = φ·R` avec `φ = (σ_{t0} + σ_t) / (2σ_t)`.
5. **Scénarios de stress** — returns bruts sur la fenêtre 2022-01-01 → 2023-12-31.
6. **Valorisation** — full revaluation du portefeuille sous chaque scénario.
7. **Expected Shortfall** — ES empirique au seuil α=99 % sur les `k = ⌈(1-α)·LP⌉` pires pertes.
8. **Marge Initiale** — `IM = max(ES_FHS, 0.75·ES_FHS + 0.25·ES_stress)`.

Modules complémentaires : `apc.py` (anti-procyclicité EMIR Art. 28 — buffer option (a), plancher option (c), indicateurs APC) et `backtesting.py` (validation EMIR Art. 49 — Kupiec POF, Christoffersen, Bâle Traffic Light, backtest IM rolling).

## Structure

```
im_ccp_project/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── config.py                  # ModelConfig (validation fail-fast)
│   ├── constants.py               # Paramètres par défaut + MODEL_VERSION
│   ├── market_data/
│   │   ├── loader.py              # Chargement CSV
│   │   ├── cleaner.py             # Déduplication, dropna, tri
│   │   └── curve.py               # Extraction date + interpolation
│   ├── pricing/
│   │   ├── discounting.py         # Interpolation, facteurs d'actualisation
│   │   └── bonds.py               # Pricing taux fixe (depuis taux ou prix ZC)
│   └── risk/
│       ├── risk_factors.py        # Prix ZC, returns historiques HP-jours
│       ├── ewma.py                # EWMA vectorisée + impl. boucle de référence
│       ├── scenarios.py           # FHS scalés + stress bruts
│       ├── lookback.py            # Fenêtrage temporel
│       ├── pnl.py                 # Full revaluation
│       ├── es.py                  # Expected Shortfall (avec _safe_tail_size)
│       ├── im.py                  # Marge Initiale hybride
│       ├── apc.py                 # Anti-procyclicité (EMIR Art. 28)
│       └── backtesting.py         # Kupiec, Christoffersen, Traffic Light
├── data/raw/ZeroCouponCurve.csv
├── notebooks/run_im_pipeline_patched.ipynb
└── tests/test_pipeline.py
```

## Installation

Pré-requis : Python ≥ 3.10.

```bash
# Runtime moteur uniquement
pip install -e .

# Avec tests + lint + types
pip install -e ".[dev]"

# Avec notebook (jupyter, matplotlib)
pip install -e ".[all]"
```

## Utilisation

```python
from src.config import ModelConfig
from src.market_data.loader import load_zero_coupon_curve
from src.market_data.cleaner import clean_zero_coupon_curve
from src.risk.risk_factors import build_zero_coupon_price_matrix, compute_historical_returns
from src.risk.ewma import get_ewma_window, compute_ewma_volatility
from src.risk.scenarios import build_scaled_scenarios, build_unscaled_scenarios
from src.risk.pnl import compute_portfolio_pnl_under_scenarios
from src.risk.es import compute_es_from_pnl
from src.risk.im import compute_initial_margin

config = ModelConfig()

zc = clean_zero_coupon_curve(load_zero_coupon_curve("data/raw/ZeroCouponCurve.csv"))
prices = build_zero_coupon_price_matrix(zc, nominal=config.nominal)
returns = compute_historical_returns(prices, HP=config.HP)

vol = compute_ewma_volatility(
    get_ewma_window(returns, t0=config.t0, LP=config.LP, SW=config.SW),
    lambda_=config.lambda_ewma,
)

scaled   = build_scaled_scenarios(returns, vol, t0=config.t0, LP=config.LP)
unscaled = build_unscaled_scenarios(returns, stress_start=config.stress_start, stress_end=config.stress_end)

current = prices.loc[config.t0]
portfolio = [
    {"maturity": 5.0, "coupon_rate": 0.03, "nominal": 100.0, "frequency": 1, "quantity": 10},
]

pnl_fhs    = compute_portfolio_pnl_under_scenarios(current, scaled,   portfolio)
pnl_stress = compute_portfolio_pnl_under_scenarios(current, unscaled, portfolio)

es_fhs    = compute_es_from_pnl(pnl_fhs,    alpha=config.alpha)
es_stress = compute_es_from_pnl(pnl_stress, alpha=config.alpha)

im = compute_initial_margin(es_fhs, es_stress, fhs_w=config.FHS_w, stress_w=config.Stress_w)
print(f"Initial Margin = {im:,.2f}")
```

### Anti-procyclicité

```python
from src.risk.apc import apply_buffer_option_a, apply_floor_option_c

out = apply_buffer_option_a(im_raw=im, buffer_pct=0.25)            # Option (a)
floored = apply_floor_option_c(im_history, floor_pct=0.25)         # Option (c)
```

### Backtesting

```python
from src.risk.backtesting import run_im_backtest, summarize_backtest

bt = run_im_backtest(zc_curve_df=zc, portfolio=portfolio, config=config,
                     start="2024-01-01", end="2025-05-30")
report = summarize_backtest(bt, alpha=0.99)
```

## Paramètres par défaut

| Paramètre | Symbole | Valeur |
|---|---|---|
| Lookback Period | LP | 2500 jours (~10 ans) |
| Holding Period | HP | 5 jours |
| Smoothing Window | SW | 60 jours |
| Decay EWMA | λ | 0.97 |
| Quantile ES | α | 99 % |
| Poids FHS / stress | w_FHS / w_stress | 0.75 / 0.25 |
| Fenêtre de stress | — | 2022-01-01 → 2023-12-31 |
| Buffer / plancher APC | — | 25 % / 25 % |


## Références

https://www.euronext.com/en/clearing/risk-management/methodologies
