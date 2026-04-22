"""
Backtesting du modèle de Marge Initiale.

Module **NOUVEAU** (v1.1.0) — livrable obligatoire pour validation
modèle CCP au sens de :
    - EMIR Art. 49 (validation independante)
    - RTS 153/2013 Annexe II (parametres de modele)
    - CPMI-IOSCO PFMI Principe 6 (margin)

Tests implementes
-----------------
1. Couverture inconditionnelle (Kupiec POF, 1995)
   H0 : la frequence d'exceptions = 1 - alpha
   Statistique : LR_POF ~ chi2(1) sous H0

2. Independance (Christoffersen, 1998)
   H0 : les exceptions sont independantes (pas de clustering)
   Statistique : LR_IND ~ chi2(1) sous H0

3. Couverture conditionnelle = POF + IND
   Statistique : LR_CC ~ chi2(2) sous H0

4. Traffic Light (Bale, 1996)
   Classification reglementaire des modeles :
     - Vert  : 0-4 exceptions / 250 jours  -> modele acceptable
     - Jaune : 5-9                          -> investigation requise
     - Rouge : >= 10                        -> modele rejete
   Adapte ici pour seuil ES alpha=99%.

5. Test du depassement moyen (magnitude)
   Pour ES (vs VaR) : la moyenne des pertes en queue doit etre
   coherente avec l'ES estime ex-ante.

Usage type
----------
>>> bt = run_im_backtest(
...     zc_curve_df, portfolio, config,
...     start='2024-01-01', end='2025-05-30',
... )
>>> bt['exception_rate']  # % de jours ou loss > IM
>>> kupiec_pof_test(bt['n_exceptions'], bt['n_obs'], alpha=0.99)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.config import ModelConfig
from src.risk.risk_factors import (
    build_zero_coupon_price_matrix,
    compute_historical_returns,
)
from src.risk.ewma import compute_ewma_volatility
from src.risk.scenarios import build_scaled_scenarios, build_unscaled_scenarios
from src.risk.pnl import (
    compute_portfolio_initial_value,
    compute_portfolio_pnl_under_scenarios,
)
from src.risk.es import compute_es_from_pnl
from src.risk.im import compute_initial_margin


# =========================================================================== #
#  Tests statistiques
# =========================================================================== #
@dataclass
class TestResult:
    """Resultat d'un test statistique."""
    name: str
    statistic: float
    p_value: float
    df: int
    rejected_at_5pct: bool
    interpretation: str


def kupiec_pof_test(
    n_exceptions: int,
    n_obs: int,
    alpha: float = 0.99,
) -> TestResult:
    """Test de couverture inconditionnelle de Kupiec (POF).

    Statistique : LR = -2 * ln[L(p) / L(pi_hat)]
    avec p = 1 - alpha (taux d'exception attendu)
    et pi_hat = N / T (taux observe).

    Parameters
    ----------
    n_exceptions : int
        Nombre d'exceptions observees (loss > IM).
    n_obs : int
        Nombre total de jours backtests.
    alpha : float
        Niveau de confiance ES.

    Returns
    -------
    TestResult
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs doit etre > 0, recu {n_obs}")
    if not (0 <= n_exceptions <= n_obs):
        raise ValueError(f"n_exceptions doit etre dans [0, {n_obs}]")

    p = 1.0 - alpha
    n = n_obs
    x = n_exceptions

    # Cas degeneres
    if x == 0:
        lr = -2.0 * n * math.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * math.log(p)
    else:
        pi_hat = x / n
        log_l_h0 = (n - x) * math.log(1.0 - p) + x * math.log(p)
        log_l_h1 = (n - x) * math.log(1.0 - pi_hat) + x * math.log(pi_hat)
        lr = -2.0 * (log_l_h0 - log_l_h1)

    p_value = 1.0 - chi2.cdf(lr, df=1)
    rejected = p_value < 0.05

    interp = (
        f"Taux observe = {x}/{n} = {x/n:.4f} vs attendu {p:.4f}. "
        f"{'REJET de H0 a 5%' if rejected else 'Non rejet de H0 a 5%'} "
        f"(p-value={p_value:.4f})."
    )
    return TestResult(
        name="Kupiec POF",
        statistic=lr,
        p_value=p_value,
        df=1,
        rejected_at_5pct=rejected,
        interpretation=interp,
    )


def christoffersen_independence_test(
    exceptions: pd.Series,
) -> TestResult:
    """Test d'independance de Christoffersen.

    Construit la matrice de transition entre etats {0=ok, 1=exception}.
    H0 : pi_01 = pi_11 (Markov d'ordre 0 = independance).

    Parameters
    ----------
    exceptions : pd.Series
        Serie binaire (0/1) indiquant les exceptions.

    Returns
    -------
    TestResult
    """
    arr = np.asarray(exceptions, dtype=int)
    if len(arr) < 2:
        raise ValueError("Serie d'exceptions trop courte (besoin >= 2 obs).")

    # Comptage des transitions
    n00 = int(np.sum((arr[:-1] == 0) & (arr[1:] == 0)))
    n01 = int(np.sum((arr[:-1] == 0) & (arr[1:] == 1)))
    n10 = int(np.sum((arr[:-1] == 1) & (arr[1:] == 0)))
    n11 = int(np.sum((arr[:-1] == 1) & (arr[1:] == 1)))

    # Probabilites conditionnelles
    n0 = n00 + n01
    n1 = n10 + n11
    n_total = n0 + n1
    if n_total == 0 or (n01 + n11) == 0:
        # Pas d'exceptions du tout -> test non pertinent
        return TestResult(
            name="Christoffersen IND",
            statistic=0.0,
            p_value=1.0,
            df=1,
            rejected_at_5pct=False,
            interpretation="Aucune exception observee, test non applicable.",
        )

    pi = (n01 + n11) / n_total  # probabilite inconditionnelle
    pi_01 = n01 / n0 if n0 > 0 else 0.0
    pi_11 = n11 / n1 if n1 > 0 else 0.0

    def _safe_log(x: float) -> float:
        return math.log(x) if x > 0 else 0.0

    log_l_h0 = (
        (n00 + n10) * _safe_log(1.0 - pi)
        + (n01 + n11) * _safe_log(pi)
    )
    log_l_h1 = (
        n00 * _safe_log(1.0 - pi_01)
        + n01 * _safe_log(pi_01)
        + n10 * _safe_log(1.0 - pi_11)
        + n11 * _safe_log(pi_11)
    )
    lr = -2.0 * (log_l_h0 - log_l_h1)
    p_value = 1.0 - chi2.cdf(lr, df=1)
    rejected = p_value < 0.05

    interp = (
        f"pi_01={pi_01:.4f}, pi_11={pi_11:.4f}. "
        f"{'REJET (clustering detecte)' if rejected else 'Pas de clustering'} "
        f"(p-value={p_value:.4f})."
    )
    return TestResult(
        name="Christoffersen IND",
        statistic=lr,
        p_value=p_value,
        df=1,
        rejected_at_5pct=rejected,
        interpretation=interp,
    )


def christoffersen_conditional_coverage_test(
    exceptions: pd.Series,
    alpha: float = 0.99,
) -> TestResult:
    """Test de couverture conditionnelle = POF + IND.

    LR_CC = LR_POF + LR_IND ~ chi2(2) sous H0.
    """
    n_exc = int(exceptions.sum())
    n_obs = len(exceptions)
    pof = kupiec_pof_test(n_exc, n_obs, alpha)
    ind = christoffersen_independence_test(exceptions)
    lr_cc = pof.statistic + ind.statistic
    p_value = 1.0 - chi2.cdf(lr_cc, df=2)
    rejected = p_value < 0.05

    interp = (
        f"LR_CC = LR_POF + LR_IND = {pof.statistic:.3f} + "
        f"{ind.statistic:.3f} = {lr_cc:.3f}. "
        f"{'REJET' if rejected else 'Non rejet'} a 5% (p-value={p_value:.4f})."
    )
    return TestResult(
        name="Christoffersen CC",
        statistic=lr_cc,
        p_value=p_value,
        df=2,
        rejected_at_5pct=rejected,
        interpretation=interp,
    )


def basel_traffic_light(
    n_exceptions: int,
    n_obs: int = 250,
) -> dict[str, Any]:
    """Classification Bale du modele.

    Seuils originaux pour VaR 99% / 250 jours :
        - Vert  : 0-4
        - Jaune : 5-9
        - Rouge : >= 10
    Pour un usage ES, ces seuils restent indicatifs.

    Returns
    -------
    dict
        Couleur, multiplicateur de scaling Bale, et probabilite cumulative.
    """
    if n_obs != 250:
        # Ajustement proportionnel des seuils
        scale = n_obs / 250
        green_max = int(round(4 * scale))
        yellow_max = int(round(9 * scale))
    else:
        green_max = 4
        yellow_max = 9

    if n_exceptions <= green_max:
        color = "GREEN"
        multiplier = 3.00
        verdict = "Modele acceptable."
    elif n_exceptions <= yellow_max:
        color = "YELLOW"
        # Multiplicateurs Bale pour la zone jaune
        ladder = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
        multiplier = ladder.get(n_exceptions, 3.85)
        verdict = "Investigation requise par la fonction de validation."
    else:
        color = "RED"
        multiplier = 4.00
        verdict = "Modele a recalibrer ou rejeter."

    return {
        "color": color,
        "multiplier": multiplier,
        "n_exceptions": n_exceptions,
        "n_obs": n_obs,
        "exception_rate": n_exceptions / n_obs,
        "verdict": verdict,
    }


# =========================================================================== #
#  Backtest IM rolling
# =========================================================================== #
def _compute_im_at_date(
    zc_curve_df: pd.DataFrame,
    zc_price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    portfolio: list[dict],
    config: ModelConfig,
    t: pd.Timestamp,
) -> tuple[float, float]:
    """Calcule l'IM et la valeur du portefeuille a la date t."""
    # Re-instancier la config pour cette date (immutabilite logique)
    cfg = ModelConfig(
        LP=config.LP,
        HP=config.HP,
        SW=config.SW,
        lambda_ewma=config.lambda_ewma,
        t0=t.strftime("%Y-%m-%d"),
        stress_start=config.stress_start,
        stress_end=config.stress_end,
        alpha=config.alpha,
        FHS_w=config.FHS_w,
        Stress_w=config.Stress_w,
        nominal=config.nominal,
    )

    # Volatilite EWMA sur fenetre [t - LP - SW, t]
    from src.risk.ewma import get_ewma_window
    ewma_window = get_ewma_window(returns_df, cfg.t0, cfg.LP, cfg.SW)
    vol_df = compute_ewma_volatility(ewma_window, cfg.lambda_ewma)

    # Scenarios FHS scales + stress bruts
    scaled = build_scaled_scenarios(returns_df, vol_df, cfg.t0, cfg.LP)
    unscaled = build_unscaled_scenarios(
        returns_df,
        stress_start=cfg.stress_start,
        stress_end=cfg.stress_end,
    )

    # Pricing
    current_curve = zc_price_df.loc[t]
    v0 = compute_portfolio_initial_value(current_curve, portfolio)
    pnl_scaled = compute_portfolio_pnl_under_scenarios(
        current_curve, scaled, portfolio
    )
    pnl_unscaled = compute_portfolio_pnl_under_scenarios(
        current_curve, unscaled, portfolio
    )

    # ES + IM
    es_fhs = compute_es_from_pnl(pnl_scaled, cfg.alpha)
    es_stress = compute_es_from_pnl(pnl_unscaled, cfg.alpha)
    im = compute_initial_margin(es_fhs, es_stress, cfg.FHS_w, cfg.Stress_w)
    return im, v0


def run_im_backtest(
    zc_curve_df: pd.DataFrame,
    portfolio: list[dict],
    config: ModelConfig,
    start: str,
    end: str,
    step: int = 1,
) -> pd.DataFrame:
    """Execute un backtest rolling de l'IM.

    Pour chaque date t dans [start, end], on calcule :
      - IM(t) avec l'historique disponible avant t,
      - V(t)  : valeur du portefeuille a t,
      - V(t + HP) : valeur a la fin de la periode de detention,
      - Loss = V(t) - V(t + HP),
      - Exception = (Loss > IM(t)).

    Parameters
    ----------
    zc_curve_df : pd.DataFrame
        Courbe ZC (taux), nettoyee.
    portfolio : list[dict]
        Positions obligataires.
    config : ModelConfig
        Configuration de reference.
    start, end : str
        Periode de backtest (format YYYY-MM-DD).
    step : int
        Pas en jours ouvres (1 = quotidien, 5 = hebdomadaire).

    Returns
    -------
    pd.DataFrame
        Colonnes : IM, V0, V_HP, loss, exception.
    """
    zc_price_df = build_zero_coupon_price_matrix(zc_curve_df, config.nominal)
    returns_df = compute_historical_returns(zc_price_df, config.HP)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Filtre des dates eligibles : doivent avoir LP+SW+1 obs en amont
    # ET HP obs en aval pour le calcul de la perte realisee.
    eligible = returns_df.index[
        (returns_df.index >= start_dt) & (returns_df.index <= end_dt)
    ]
    eligible = eligible[::step]

    records = []
    for t in eligible:
        try:
            im, v0 = _compute_im_at_date(
                zc_curve_df, zc_price_df, returns_df,
                portfolio, config, t,
            )
        except (ValueError, KeyError):
            # Historique insuffisant -> skip
            continue

        # Valeur realisee a t + HP
        future_dates = zc_price_df.index[zc_price_df.index > t]
        if len(future_dates) < config.HP:
            continue
        t_hp = future_dates[config.HP - 1]
        future_curve = zc_price_df.loc[t_hp]
        v_hp = compute_portfolio_initial_value(future_curve, portfolio)

        loss = v0 - v_hp  # convention : loss > 0 si portefeuille perd
        exception = bool(loss > im)

        records.append({
            "date": t,
            "IM": im,
            "V0": v0,
            "V_HP": v_hp,
            "loss": loss,
            "loss_pct_v0": loss / v0 if v0 != 0 else np.nan,
            "exception": exception,
        })

    df = pd.DataFrame(records).set_index("date")
    return df


def summarize_backtest(
    backtest_df: pd.DataFrame,
    alpha: float = 0.99,
) -> dict[str, Any]:
    """Aggrege tous les indicateurs de validation."""
    if backtest_df.empty:
        return {"status": "EMPTY", "n_obs": 0}

    n_obs = len(backtest_df)
    exceptions = backtest_df["exception"].astype(int)
    n_exc = int(exceptions.sum())

    pof = kupiec_pof_test(n_exc, n_obs, alpha)
    ind = christoffersen_independence_test(exceptions)
    cc = christoffersen_conditional_coverage_test(exceptions, alpha)
    tl = basel_traffic_light(n_exc, n_obs)

    # Magnitude moyenne des exceptions (utile pour ES vs VaR)
    if n_exc > 0:
        avg_exc_loss = backtest_df.loc[exceptions == 1, "loss"].mean()
        avg_exc_im = backtest_df.loc[exceptions == 1, "IM"].mean()
        magnitude_ratio = avg_exc_loss / avg_exc_im
    else:
        avg_exc_loss = np.nan
        magnitude_ratio = np.nan

    return {
        "n_obs": n_obs,
        "n_exceptions": n_exc,
        "exception_rate": n_exc / n_obs,
        "expected_rate": 1.0 - alpha,
        "kupiec_pof": pof,
        "christoffersen_ind": ind,
        "christoffersen_cc": cc,
        "basel_traffic_light": tl,
        "avg_exception_loss": avg_exc_loss,
        "magnitude_ratio": magnitude_ratio,
        "im_mean": backtest_df["IM"].mean(),
        "im_std": backtest_df["IM"].std(),
    }
