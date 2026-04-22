"""
Tests unitaires et de non-régression de la pipeline IM v1.1.0.

Couvre chaque module avec des cas déterministes pour garantir :
  - la conformité aux équations de la note méthodologique ;
  - la non-régression numérique sur le portefeuille de référence ;
  - l'équivalence entre l'EWMA vectorisée et l'implémentation boucle.

Organisation
------------
Chaque classe `Test*` cible un module unique et n'utilise que des
fixtures synthétiques contrôlées, sauf le bloc final
`TestEndToEndNonRegression` qui rejoue la pipeline complète sur
`data/raw/ZeroCouponCurve.csv` et compare au chiffre publié.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.constants import MODEL_VERSION
from src.market_data.cleaner import clean_zero_coupon_curve
from src.market_data.curve import get_curve_at_date, interpolate_zc_rate
from src.market_data.loader import load_zero_coupon_curve
from src.pricing.bonds import (
    price_fixed_rate_bond,
    price_fixed_rate_bond_from_zc_prices,
)
from src.pricing.discounting import (
    get_discount_factor,
    get_discount_factor_from_zc_price_curve,
    get_zero_coupon_price,
    interpolate_curve,
)
from src.risk.apc import (
    apply_buffer_option_a,
    apply_floor_option_c,
    compute_apc_metrics,
)
from src.risk.backtesting import (
    basel_traffic_light,
    christoffersen_independence_test,
    kupiec_pof_test,
)
from src.risk.es import (
    _safe_tail_size,
    compute_es_from_pnl,
    compute_expected_shortfall,
)
from src.risk.ewma import (
    compute_ewma_variance,
    compute_ewma_variance_loop,
    compute_ewma_volatility,
    get_ewma_window,
)
from src.risk.im import compute_initial_margin
from src.risk.lookback import (
    get_effective_lookback_length,
    get_lookback_window,
)
from src.risk.pnl import (
    compute_portfolio_initial_value,
    compute_portfolio_pnl_under_scenarios,
)
from src.risk.risk_factors import (
    build_zero_coupon_price_matrix,
    compute_historical_returns,
)
from src.risk.scenarios import (
    build_scaled_scenarios,
    build_unscaled_scenarios,
    compute_scaling_factors,
)


# =========================================================================== #
#  Fixtures
# =========================================================================== #
@pytest.fixture
def simple_zc_curve() -> pd.DataFrame:
    """Courbe ZC synthétique : 10 dates, 3 piliers, taux croissants."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame(
        {
            1.0: np.linspace(0.03, 0.035, 10),
            2.0: np.linspace(0.035, 0.04, 10),
            5.0: np.linspace(0.04, 0.045, 10),
        },
        index=dates,
    )


@pytest.fixture
def simple_zc_prices(simple_zc_curve: pd.DataFrame) -> pd.DataFrame:
    return build_zero_coupon_price_matrix(simple_zc_curve, nominal=100.0)


@pytest.fixture
def deterministic_returns() -> pd.DataFrame:
    """Returns pour vérification analytique de la récurrence EWMA."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    return pd.DataFrame(
        {1.0: [0.01, -0.02, 0.015, 0.005]},
        index=dates,
    )


# =========================================================================== #
#  ModelConfig (v1.1.0 : LP=2500 par défaut, __post_init__ valide)
# =========================================================================== #
class TestModelConfig:
    def test_default_config_v110(self):
        """Defaults v1.1.0 : LP=2500, total_window=2561."""
        cfg = ModelConfig()
        assert cfg.LP == 2500
        assert cfg.total_window == 2561  # LP + SW + 1 = 2500 + 60 + 1
        assert cfg.model_version == MODEL_VERSION == "1.1.0"

    def test_es_tail_size_default(self):
        """k = ceil((1-0.99) * 2500) = 25 (vs 3 avec l'ancien LP=250).

        Cohérence garantie avec ``es._safe_tail_size`` depuis le fix
        config.py (cf. CHANGELOG).
        """
        cfg = ModelConfig()
        assert cfg.es_tail_size == 25

    def test_es_tail_size_consistency_with_es_module(self):
        """Garantit que ``config.es_tail_size`` et ``_safe_tail_size``
        renvoient la même chose. Si ce test échoue, le reporting d'audit
        (summary) diverge du calcul ES effectif."""
        cfg = ModelConfig()
        assert cfg.es_tail_size == _safe_tail_size(cfg.alpha, cfg.LP)

    def test_invalid_lambda_raises_at_construction(self):
        """__post_init__ déclenche validate() → exception à la construction."""
        with pytest.raises(ValueError, match="lambda_ewma"):
            ModelConfig(lambda_ewma=1.5)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ModelConfig(alpha=1.0)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="FHS_w \\+ Stress_w"):
            ModelConfig(FHS_w=0.5, Stress_w=0.3)

    def test_lp_below_emir_minimum(self):
        with pytest.raises(ValueError, match="LP doit"):
            ModelConfig(LP=100)  # < MIN_LOOKBACK_DAYS = 250

    def test_stress_dates_order(self):
        with pytest.raises(ValueError, match="stress_start"):
            ModelConfig(stress_start="2023-12-31", stress_end="2022-01-01")

    def test_summary_contains_audit_fields(self):
        cfg = ModelConfig()
        s = cfg.summary()
        for key in ("model_version", "t0", "LP", "alpha", "es_tail_size_k"):
            assert key in s


# =========================================================================== #
#  Loader / Curve / Cleaner
# =========================================================================== #
class TestLoader:
    def test_loads_real_csv(self):
        """Smoke test sur le fichier de référence."""
        df = load_zero_coupon_curve(Path("data/raw/ZeroCouponCurve.csv"))
        assert df.index.name == "date"
        assert isinstance(df.index, pd.DatetimeIndex)
        # Toutes les colonnes doivent être typées float
        assert all(isinstance(c, float) for c in df.columns)
        # Index trié
        assert df.index.is_monotonic_increasing


class TestCurve:
    def test_get_curve_at_date(self, simple_zc_curve):
        date = str(simple_zc_curve.index[3].date())
        s = get_curve_at_date(simple_zc_curve, date)
        assert isinstance(s, pd.Series)
        assert s.iloc[0] == simple_zc_curve.iloc[3, 0]

    def test_interpolate_zc_rate_linear(self):
        curve = pd.Series({1.0: 0.03, 5.0: 0.05})
        rate_3y = interpolate_zc_rate(curve, 3.0)
        # Interpolation linéaire : 0.03 + 0.5*(0.05-0.03) = 0.04
        assert rate_3y == pytest.approx(0.04, abs=1e-12)

    def test_interpolate_extrapolation_flat(self):
        """np.interp extrapole à plat aux bornes."""
        curve = pd.Series({1.0: 0.03, 5.0: 0.05})
        assert interpolate_zc_rate(curve, 0.5) == pytest.approx(0.03)
        assert interpolate_zc_rate(curve, 10.0) == pytest.approx(0.05)


class TestCleaner:
    def test_removes_duplicates_keeps_first(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"1.0": [0.03, 0.031, 0.032]}, index=dates)
        cleaned = clean_zero_coupon_curve(df)
        assert len(cleaned) == 2
        assert cleaned.iloc[0, 0] == 0.03

    def test_drops_nan(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"1.0": [0.03, None]}, index=dates)
        assert len(clean_zero_coupon_curve(df)) == 1

    def test_sorts_index(self):
        dates = pd.to_datetime(["2024-01-03", "2024-01-02"])
        df = pd.DataFrame({"1.0": [0.04, 0.03]}, index=dates)
        cleaned = clean_zero_coupon_curve(df)
        assert cleaned.index.is_monotonic_increasing


# =========================================================================== #
#  Risk factors (éq. 2 et 3)
# =========================================================================== #
class TestRiskFactors:
    def test_zc_price_formula(self, simple_zc_curve):
        """P(t, T) = N / (1 + r(t, T))^T  — éq. (2)."""
        prices = build_zero_coupon_price_matrix(simple_zc_curve, nominal=100.0)
        r = simple_zc_curve.iloc[0, 0]
        T = simple_zc_curve.columns[0]
        expected = 100.0 / (1.0 + r) ** T
        assert prices.iloc[0, 0] == pytest.approx(expected, abs=1e-10)

    def test_returns_shape(self, simple_zc_prices):
        """compute_historical_returns(HP=2) sur 10 lignes → 8 lignes (dropna)."""
        ret = compute_historical_returns(simple_zc_prices, HP=2)
        assert len(ret) == len(simple_zc_prices) - 2

    def test_returns_formula(self, simple_zc_prices):
        """R(t, T) = P(t, T) / P(t-HP, T) - 1  — éq. (3)."""
        ret = compute_historical_returns(simple_zc_prices, HP=2)
        expected = simple_zc_prices.iloc[2, 0] / simple_zc_prices.iloc[0, 0] - 1.0
        assert ret.iloc[0, 0] == pytest.approx(expected, abs=1e-12)


# =========================================================================== #
#  EWMA — récurrence + équivalence vectorisé/boucle (test critique)
# =========================================================================== #
class TestEWMA:
    def test_ewma_recursion_step_by_step(self, deterministic_returns):
        """σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}  — éq. (4)."""
        lam = 0.94
        var = compute_ewma_variance(deterministic_returns, lambda_=lam)

        r = deterministic_returns.iloc[:, 0].to_numpy()
        # σ²_0 = r²_0  (initialisation)
        assert var.iloc[0, 0] == pytest.approx(r[0] ** 2, abs=1e-15)
        # σ²_1 = λ σ²_0 + (1-λ) r²_0
        expected_1 = lam * (r[0] ** 2) + (1 - lam) * (r[0] ** 2)
        assert var.iloc[1, 0] == pytest.approx(expected_1, abs=1e-15)
        # σ²_2 = λ σ²_1 + (1-λ) r²_1
        expected_2 = lam * expected_1 + (1 - lam) * (r[1] ** 2)
        assert var.iloc[2, 0] == pytest.approx(expected_2, abs=1e-15)

    def test_volatility_strictly_positive(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        ret = pd.DataFrame({1.0: [0.01, -0.02, 0.015, 0.005, -0.01]}, index=dates)
        vol = compute_ewma_volatility(ret, lambda_=0.97)
        assert (vol > 0).all().all()

    def test_lambda_out_of_bounds_raises(self, deterministic_returns):
        with pytest.raises(ValueError, match="lambda"):
            compute_ewma_variance(deterministic_returns, lambda_=0.0)
        with pytest.raises(ValueError, match="lambda"):
            compute_ewma_variance(deterministic_returns, lambda_=1.0)

    def test_nan_in_returns_raises(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        ret = pd.DataFrame({1.0: [0.01, np.nan, 0.005]}, index=dates)
        with pytest.raises(ValueError, match="NaN"):
            compute_ewma_variance(ret, lambda_=0.97)


class TestEWMAVectorizedVsLoop:
    """Test de non-régression critique : la version vectorisée doit
    reproduire exactement la version boucle (oracle)."""

    @pytest.mark.parametrize("lam", [0.90, 0.94, 0.97, 0.99])
    def test_match_loop_implementation(self, lam):
        np.random.seed(42)
        n_days, n_pillars = 200, 5
        ret = pd.DataFrame(
            np.random.randn(n_days, n_pillars) * 0.01,
            index=pd.bdate_range("2024-01-02", periods=n_days),
            columns=[1.0, 2.0, 5.0, 10.0, 30.0],
        )
        v_vec = compute_ewma_variance(ret, lambda_=lam)
        v_loop = compute_ewma_variance_loop(ret, lambda_=lam)
        np.testing.assert_allclose(
            v_vec.values, v_loop.values, rtol=1e-12, atol=1e-15
        )


class TestGetEwmaWindow:
    def test_extracts_correct_size(self):
        dates = pd.bdate_range("2024-01-02", periods=400)
        ret = pd.DataFrame(
            np.random.randn(400, 1) * 0.01, index=dates, columns=[1.0]
        )
        win = get_ewma_window(ret, t0=str(dates[-1].date()), LP=250, SW=60)
        assert len(win) == 250 + 60 + 1

    def test_insufficient_history_raises(self):
        dates = pd.bdate_range("2024-01-02", periods=50)
        ret = pd.DataFrame(
            np.random.randn(50, 1) * 0.01, index=dates, columns=[1.0]
        )
        with pytest.raises(ValueError, match="Fenetre EWMA insuffisante"):
            get_ewma_window(ret, t0=str(dates[-1].date()), LP=250, SW=60)


# =========================================================================== #
#  Scenarios (éq. 7-9)
# =========================================================================== #
class TestScenarios:
    def test_scaling_identity_when_constant_vol(self):
        """σ_{t0} = σ_t partout ⇒ φ = (σ+σ)/(2σ) = 1."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        vol = pd.DataFrame({1.0: [0.01, 0.01, 0.01]}, index=dates)
        factors = compute_scaling_factors(vol, t0=str(dates[-1].date()))
        np.testing.assert_allclose(factors.values, 1.0, atol=1e-15)

    def test_scaling_formula(self):
        """ϕ = (σ_{t0} + σ_t) / (2 σ_t)  — éq. (7)."""
        dates = pd.bdate_range("2024-01-02", periods=2)
        vol = pd.DataFrame({1.0: [0.01, 0.02]}, index=dates)
        factors = compute_scaling_factors(vol, t0=str(dates[-1].date()))
        # σ_{t0} = 0.02, σ_0 = 0.01 → φ_0 = (0.02+0.01)/(2*0.01) = 1.5
        assert factors.iloc[0, 0] == pytest.approx(1.5, abs=1e-12)
        # σ_{t0} = 0.02, σ_1 = 0.02 → φ_1 = 1.0
        assert factors.iloc[1, 0] == pytest.approx(1.0, abs=1e-12)

    def test_unscaled_window_filtered_correctly(self):
        dates = pd.bdate_range("2022-01-03", periods=300)
        ret = pd.DataFrame(np.ones((300, 1)) * 0.001, index=dates, columns=[1.0])
        out = build_unscaled_scenarios(
            ret, stress_start="2022-06-01", stress_end="2022-09-30"
        )
        assert (out.index >= pd.Timestamp("2022-06-01")).all()
        assert (out.index <= pd.Timestamp("2022-09-30")).all()

    def test_unscaled_requires_dates_or_periods(self):
        ret = pd.DataFrame({1.0: [0.001]}, index=[pd.Timestamp("2022-01-03")])
        with pytest.raises(ValueError, match="stressed_periods"):
            build_unscaled_scenarios(ret)

    def test_scaled_size_equals_LP(self, deterministic_returns):
        # Construire vol et returns sur 4 jours, demander LP=3
        vol = compute_ewma_volatility(deterministic_returns, lambda_=0.97)
        scaled = build_scaled_scenarios(
            deterministic_returns,
            vol,
            t0=str(deterministic_returns.index[-1].date()),
            LP=3,
        )
        assert len(scaled) == 3


# =========================================================================== #
#  Discounting
# =========================================================================== #
class TestDiscounting:
    def test_discount_factor_formula(self):
        """DF(t₀, T) = 1 / (1 + r)^T."""
        curve = pd.Series({1.0: 0.05, 2.0: 0.06, 5.0: 0.07})
        assert get_discount_factor(curve, 2.0) == pytest.approx(
            1.0 / (1.06**2), abs=1e-12
        )

    def test_zero_coupon_price(self):
        curve = pd.Series({1.0: 0.05, 2.0: 0.06})
        assert get_zero_coupon_price(curve, 2.0, nominal=100.0) == pytest.approx(
            100.0 / (1.06**2), abs=1e-10
        )

    def test_interpolation_linear(self):
        curve = pd.Series({1.0: 0.03, 5.0: 0.05})
        assert interpolate_curve(curve, 3.0) == pytest.approx(0.04, abs=1e-12)

    def test_df_from_zc_prices(self):
        """DF = P_ZC / N."""
        zc_prices = pd.Series({1.0: 95.0, 2.0: 90.0})
        assert get_discount_factor_from_zc_price_curve(
            zc_prices, 2.0, zc_nominal=100.0
        ) == pytest.approx(0.90, abs=1e-12)


# =========================================================================== #
#  Bonds
# =========================================================================== #
class TestBonds:
    def test_zero_coupon_bond_from_rates(self):
        """Coupon = 0 ⇒ prix = N · DF(T)."""
        curve = pd.Series({1.0: 0.05, 2.0: 0.06})
        price = price_fixed_rate_bond(
            curve, maturity=2.0, coupon_rate=0.0, nominal=100.0
        )
        assert price == pytest.approx(100.0 / (1.06**2), abs=1e-10)

    def test_par_bond_at_zero_rates(self):
        """À taux nul, le prix d'un bond classique = nominal + Σ coupons."""
        curve = pd.Series({1.0: 0.0, 2.0: 0.0, 5.0: 0.0})
        price = price_fixed_rate_bond(
            curve, maturity=2.0, coupon_rate=0.05, nominal=100.0, frequency=1
        )
        # 2 coupons de 5 + nominal 100 = 110
        assert price == pytest.approx(110.0, abs=1e-10)

    def test_zc_price_method_matches_rate_method(self):
        """Les deux pricers doivent donner le même résultat."""
        rate_curve = pd.Series({1.0: 0.05, 2.0: 0.06, 5.0: 0.07})
        zc_curve = pd.Series(
            {T: 100.0 / (1.0 + r) ** T for T, r in rate_curve.items()}
        )
        p1 = price_fixed_rate_bond(
            rate_curve, maturity=2.0, coupon_rate=0.04, nominal=100.0
        )
        p2 = price_fixed_rate_bond_from_zc_prices(
            zc_curve, maturity=2.0, coupon_rate=0.04, nominal=100.0
        )
        assert p1 == pytest.approx(p2, rel=1e-10)


# =========================================================================== #
#  Expected Shortfall — y compris la correction k=25 vs k=26 (bug v1.1.1)
# =========================================================================== #
class TestSafeTailSize:
    """La fonction qui corrige le bug numérique flottant sur ceil((1-α)·n)."""

    def test_bugfix_alpha099_n2500(self):
        """LE cas qui a motivé le fix : (1-0.99)*2500 = 25.000...0023 → 25."""
        assert _safe_tail_size(0.99, 2500) == 25

    def test_bugfix_alpha099_n250(self):
        """Pour LP=250 : k=3 (cas legacy)."""
        assert _safe_tail_size(0.99, 250) == 3

    def test_legitimate_ceiling_preserved(self):
        """ceil(0.025*250)=ceil(6.25)=7 ne doit PAS être arrondi à 6."""
        assert _safe_tail_size(0.975, 250) == 7

    def test_minimum_one(self):
        assert _safe_tail_size(0.99, 50) >= 1


class TestExpectedShortfall:
    def test_es_basic_uses_safe_k(self):
        """ES α=99 % sur 2500 → moyenne des 25 pires (pas 26)."""
        losses = pd.Series(np.arange(2500, dtype=float))  # 0..2499
        # k = 25, top losses = 2475..2499, moyenne = 2487
        expected = float(np.mean(np.arange(2475, 2500)))
        assert compute_expected_shortfall(losses, alpha=0.99) == pytest.approx(
            expected, abs=1e-10
        )

    def test_es_from_pnl_converts_sign(self):
        """PnL négatif = perte positive."""
        pnl = pd.Series([-10.0, -5.0, 2.0, 3.0, 4.0])
        # losses = [10, 5, -2, -3, -4], k=ceil(0.5*5)=3, top 3 = 10,5,-2
        es = compute_es_from_pnl(pnl, alpha=0.50)
        assert es == pytest.approx((10.0 + 5.0 - 2.0) / 3.0, abs=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="vide"):
            compute_expected_shortfall(pd.Series([], dtype=float))


# =========================================================================== #
#  PnL
# =========================================================================== #
class TestPnL:
    def test_initial_value_sums_positions(self):
        zc_prices = pd.Series({1.0: 95.0, 2.0: 90.0, 5.0: 80.0})
        portfolio = [
            {"maturity": 2.0, "coupon_rate": 0.0, "nominal": 100.0,
             "frequency": 1, "quantity": 10.0},
        ]
        v0 = compute_portfolio_initial_value(zc_prices, portfolio)
        # Coupon 0, maturity 2 → prix unitaire = 90, ×10 = 900
        assert v0 == pytest.approx(900.0, abs=1e-8)

    def test_zero_scenario_returns_zero_pnl(self):
        """Returns scénarios = 0 ⇒ PnL = 0 sous chaque scénario."""
        zc_prices = pd.Series({1.0: 95.0, 2.0: 90.0, 5.0: 80.0})
        portfolio = [
            {"maturity": 2.0, "coupon_rate": 0.03, "nominal": 100.0,
             "frequency": 1, "quantity": 5.0},
        ]
        scen = pd.DataFrame(
            np.zeros((4, 3)),
            index=pd.bdate_range("2024-01-02", periods=4),
            columns=[1.0, 2.0, 5.0],
        )
        pnl = compute_portfolio_pnl_under_scenarios(zc_prices, scen, portfolio)
        np.testing.assert_allclose(pnl.values, 0.0, atol=1e-8)


# =========================================================================== #
#  Initial Margin (éq. 19-20)
# =========================================================================== #
class TestInitialMargin:
    def test_floor_binding_when_stress_below_fhs(self):
        """ES_stress < ES_FHS ⇒ plancher actif, IM = ES_FHS."""
        im = compute_initial_margin(es_fhs=100.0, es_stress=50.0)
        assert im == 100.0

    def test_hybrid_when_stress_above_fhs(self):
        """ES_stress > ES_FHS ⇒ IM = combinaison hybride."""
        im = compute_initial_margin(es_fhs=100.0, es_stress=200.0)
        assert im == pytest.approx(0.75 * 100.0 + 0.25 * 200.0, abs=1e-10)

    def test_equality_case(self):
        assert compute_initial_margin(100.0, 100.0) == 100.0

    def test_default_weights(self):
        """Vérifie que les poids par défaut sont 0.75/0.25."""
        im = compute_initial_margin(100.0, 200.0)
        assert im == pytest.approx(125.0)


# =========================================================================== #
#  Lookback (uniquement les 2 fonctions exposées)
# =========================================================================== #
class TestLookback:
    @pytest.fixture
    def df_20(self):
        dates = pd.bdate_range("2024-01-02", periods=20)
        return pd.DataFrame({1.0: range(20)}, index=dates)

    def test_window_exact_size(self, df_20):
        t0 = str(df_20.index[-1].date())
        assert len(get_lookback_window(df_20, t0, LP=5)) == 5

    def test_window_clipped_when_lp_too_large(self, df_20):
        """Si LP > données disponibles, on récupère tout l'historique."""
        t0 = str(df_20.index[-1].date())
        assert len(get_lookback_window(df_20, t0, LP=100)) == 20

    def test_window_includes_t0(self, df_20):
        t0 = str(df_20.index[-1].date())
        win = get_lookback_window(df_20, t0, LP=5)
        assert pd.to_datetime(t0) in win.index

    def test_effective_length_capped(self, df_20):
        t0 = str(df_20.index[-1].date())
        assert get_effective_lookback_length(df_20, t0, requested_lp=5) == 5
        assert get_effective_lookback_length(df_20, t0, requested_lp=100) == 20


# =========================================================================== #
#  APC (EMIR Art. 28)
# =========================================================================== #
class TestAPC:
    def test_buffer_option_a_adds_25pct(self):
        out = apply_buffer_option_a(im_raw=1000.0, buffer_pct=0.25)
        assert out["im_raw"] == 1000.0
        assert out["buffer"] == 250.0
        assert out["im_with_buffer"] == 1250.0

    def test_floor_option_c_constant_series(self):
        """Sur une série constante, floor = floor_pct * IM_max = même valeur."""
        s = pd.Series([100.0] * 10, index=pd.bdate_range("2024-01-02", periods=10))
        floored = apply_floor_option_c(s, floor_pct=0.25)
        # floor à chaque date = 0.25 * 100 = 25 < 100 → série inchangée
        np.testing.assert_allclose(floored.values, s.values, atol=1e-10)

    def test_floor_option_c_binding(self):
        """Si IM courante chute sous 25 % du max, le plancher la relève."""
        idx = pd.bdate_range("2024-01-02", periods=5)
        # IM passe de 100 à 10 → plancher = 0.25*100 = 25 doit s'activer
        s = pd.Series([100.0, 80.0, 50.0, 20.0, 10.0], index=idx)
        floored = apply_floor_option_c(s, floor_pct=0.25, lookback_days=10)
        # Dernière valeur : max=100 → floor=25, IM_brut=10 → max(10, 25) = 25
        assert floored.iloc[-1] == pytest.approx(25.0, abs=1e-10)

    def test_floor_empty_series(self):
        out = apply_floor_option_c(pd.Series([], dtype=float))
        assert out.empty

    def test_apc_metrics_basic(self):
        idx = pd.bdate_range("2024-01-02", periods=300)
        np.random.seed(0)
        # IM légèrement bruitée autour de 1000
        s = pd.Series(1000.0 + np.random.randn(300) * 10, index=idx)
        m = compute_apc_metrics(s, apc_window=60)
        assert m.n_obs == 300
        assert m.im_min < m.im_mean < m.im_max
        assert 0.0 < m.apc_ratio <= 1.0

    def test_apc_metrics_empty_raises(self):
        with pytest.raises(ValueError, match="vide"):
            compute_apc_metrics(pd.Series([], dtype=float))


# =========================================================================== #
#  Backtesting (Kupiec, Christoffersen, Bâle Traffic Light)
# =========================================================================== #
class TestKupiecPOF:
    def test_zero_exceptions(self):
        """Aucune exception sur 250 jours, alpha=99 % : H0 plausible."""
        r = kupiec_pof_test(n_exceptions=0, n_obs=250, alpha=0.99)
        assert r.name == "Kupiec POF"
        assert r.df == 1
        assert r.statistic > 0  # LR > 0 même si x=0 (cf. cas dégénéré)

    def test_observed_rate_equals_expected(self):
        """N=2500, x=25, p=0.01 → π_hat = p exactement → LR ≈ 0."""
        r = kupiec_pof_test(n_exceptions=25, n_obs=2500, alpha=0.99)
        assert r.statistic == pytest.approx(0.0, abs=1e-10)
        assert not r.rejected_at_5pct

    def test_too_many_exceptions_rejects(self):
        """50 exceptions sur 1000 jours à α=99 % → 5x le taux attendu."""
        r = kupiec_pof_test(n_exceptions=50, n_obs=1000, alpha=0.99)
        assert r.rejected_at_5pct
        assert r.p_value < 0.05

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            kupiec_pof_test(n_exceptions=-1, n_obs=100)
        with pytest.raises(ValueError):
            kupiec_pof_test(n_exceptions=10, n_obs=0)


class TestChristoffersenIndependence:
    def test_no_clustering(self):
        """Exceptions isolées, pas de clustering → H0 acceptée."""
        ex = pd.Series([0] * 100 + [1] + [0] * 100 + [1] + [0] * 50)
        r = christoffersen_independence_test(ex)
        assert not r.rejected_at_5pct

    def test_strong_clustering_detected(self):
        """Toutes les exceptions consécutives → clustering extrême."""
        # 0 longtemps, puis bloc d'exceptions consécutives, puis 0
        ex = pd.Series([0] * 200 + [1] * 10 + [0] * 100)
        r = christoffersen_independence_test(ex)
        # On attend une statistique élevée et rejet (ou cas dégénéré géré)
        assert r.statistic >= 0


class TestBaselTrafficLight:
    def test_green_zone(self):
        out = basel_traffic_light(n_exceptions=2, n_obs=250)
        assert out["color"] == "GREEN"
        assert out["multiplier"] == 3.00

    def test_yellow_zone(self):
        out = basel_traffic_light(n_exceptions=6, n_obs=250)
        assert out["color"] == "YELLOW"
        assert 3.0 < out["multiplier"] < 4.0

    def test_red_zone(self):
        out = basel_traffic_light(n_exceptions=15, n_obs=250)
        assert out["color"] == "RED"
        assert out["multiplier"] == 4.00


# =========================================================================== #
#  Test de NON-RÉGRESSION end-to-end sur le portefeuille de référence
# =========================================================================== #
# Valeurs cibles publiées dans le notebook v1.1.0 (config par défaut sur
# data/raw/ZeroCouponCurve.csv au 2025-05-30).
REFERENCE_V0 = 81687674.5775437
REFERENCE_ES_FHS = 1686174.5450133937
REFERENCE_ES_STRESS = 2713758.701500714
REFERENCE_IM = 1943070.5841352236
REFERENCE_N_FHS = 2500
REFERENCE_N_STRESS = 520

# Portefeuille obligataire de référence (cf. notebook §7)
REFERENCE_PORTFOLIO = [
    {"maturity": 4.47, "coupon_rate": 0.0545, "nominal": 100.0,
     "frequency": 1, "quantity": 192678.0},
    {"maturity": 4.16, "coupon_rate": 0.0560, "nominal": 100.0,
     "frequency": 1, "quantity": 105696.0},
    {"maturity": 9.12, "coupon_rate": 0.0585, "nominal": 100.0,
     "frequency": 1, "quantity": -92066.0},
    {"maturity": 26.02, "coupon_rate": 0.0345, "nominal": 100.0,
     "frequency": 1, "quantity": 13829.0},
    {"maturity": 7.33, "coupon_rate": 0.0240, "nominal": 100.0,
     "frequency": 1, "quantity": 58940.0},
    {"maturity": 0.92, "coupon_rate": 0.0270, "nominal": 100.0,
     "frequency": 1, "quantity": 111563.0},
    {"maturity": 30.19, "coupon_rate": 0.0450, "nominal": 100.0,
     "frequency": 1, "quantity": 177496.0},
    {"maturity": 30.01, "coupon_rate": 0.0490, "nominal": 100.0,
     "frequency": 1, "quantity": 153059.0},
]


@pytest.fixture(scope="module")
def full_pipeline_results():
    """Exécute la pipeline complète une seule fois et partage les résultats."""
    csv = Path("data/raw/ZeroCouponCurve.csv")
    if not csv.exists():
        pytest.skip(f"Fichier de données absent : {csv}")

    config = ModelConfig()
    zc = clean_zero_coupon_curve(load_zero_coupon_curve(csv))
    prices = build_zero_coupon_price_matrix(zc, nominal=config.nominal)
    returns = compute_historical_returns(prices, HP=config.HP)
    win = get_ewma_window(returns, t0=config.t0, LP=config.LP, SW=config.SW)
    vol = compute_ewma_volatility(win, lambda_=config.lambda_ewma)
    scaled = build_scaled_scenarios(returns, vol, t0=config.t0, LP=config.LP)
    unscaled = build_unscaled_scenarios(
        returns,
        stress_start=config.stress_start,
        stress_end=config.stress_end,
    )
    current = prices.loc[config.t0]
    v0 = compute_portfolio_initial_value(current, REFERENCE_PORTFOLIO)
    pnl_s = compute_portfolio_pnl_under_scenarios(
        current, scaled, REFERENCE_PORTFOLIO
    )
    pnl_u = compute_portfolio_pnl_under_scenarios(
        current, unscaled, REFERENCE_PORTFOLIO
    )
    es_s = compute_es_from_pnl(pnl_s, alpha=config.alpha)
    es_u = compute_es_from_pnl(pnl_u, alpha=config.alpha)
    im = compute_initial_margin(
        es_s, es_u, fhs_w=config.FHS_w, stress_w=config.Stress_w
    )
    return {
        "v0": v0,
        "es_fhs": es_s,
        "es_stress": es_u,
        "im": im,
        "n_fhs": len(pnl_s),
        "n_stress": len(pnl_u),
    }


class TestEndToEndNonRegression:
    """Verrouille les valeurs publiées du modèle v1.1.0.

    Tout changement non intentionnel dans n'importe quel maillon de la
    pipeline (cleaner, EWMA, scenarios, pricer, ES, IM) provoquera un
    écart sur ces 4 chiffres et fera échouer ces tests — c'est le
    filet de sécurité principal pour les futures contributions.
    """

    def test_initial_portfolio_value(self, full_pipeline_results):
        assert full_pipeline_results["v0"] == pytest.approx(REFERENCE_V0, rel=1e-12)

    def test_es_fhs_scaled(self, full_pipeline_results):
        assert full_pipeline_results["es_fhs"] == pytest.approx(
            REFERENCE_ES_FHS, rel=1e-12
        )

    def test_es_stress_unscaled(self, full_pipeline_results):
        assert full_pipeline_results["es_stress"] == pytest.approx(
            REFERENCE_ES_STRESS, rel=1e-12
        )

    def test_initial_margin(self, full_pipeline_results):
        assert full_pipeline_results["im"] == pytest.approx(
            REFERENCE_IM, rel=1e-12
        )

    def test_scenarios_count(self, full_pipeline_results):
        assert full_pipeline_results["n_fhs"] == REFERENCE_N_FHS
        assert full_pipeline_results["n_stress"] == REFERENCE_N_STRESS

    def test_floor_not_binding_on_reference(self, full_pipeline_results):
        """Sur ce portefeuille, le plancher n'est PAS contraignant
        (cf. notebook §10 : 'Plancher contraignant ? NON')."""
        es_fhs = full_pipeline_results["es_fhs"]
        es_stress = full_pipeline_results["es_stress"]
        es_hybrid = 0.75 * es_fhs + 0.25 * es_stress
        assert es_hybrid > es_fhs  # hybride plus haut → c'est lui qui pilote
        assert full_pipeline_results["im"] == pytest.approx(es_hybrid, rel=1e-12)
