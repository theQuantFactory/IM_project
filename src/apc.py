"""
Mesures d'anti-procyclicite (APC) pour la Marge Initiale.

Module **NOUVEAU** (v1.1.0) — exigence reglementaire EMIR Art. 28 et
RTS 153/2013 Article 28 (CCP doivent appliquer au moins une des trois
options anti-procyclicite suivantes) :

    Option (a) : Buffer de marge constitue en periode calme (>= 25%),
                 librable en periode de stress.

    Option (b) : Lookback period d'au moins 10 ans (>= 2500 jours
                 ouvres) couvrant une periode de stress significative.
                 -> RESOLU par le passage a DEFAULT_LP=2500 (cf.
                    constants.py et CORRECTIONS.md §1).

    Option (c) : Plancher = max(IM_courant, 25% * max(IM) sur 10 ans).
                 -> Implemente ici (apply_floor_option_c).

Indicateurs APC produits ici (a publier dans le reporting modele) :
    - delta_J_J             : variation jour-sur-jour de l'IM
    - peak_to_trough        : amplitude max sur fenetre roulante
    - apc_ratio             : IM_min / IM_max sur 250 jours
    - apc_floor_binding_pct : % de jours ou le plancher option (c) est actif
    - max_increase_horizon  : pire hausse cumulee sur N jours (call de marge)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_APC_BUFFER_PCT,
    DEFAULT_APC_FLOOR_PCT,
    MAX_DAILY_IM_CHANGE_WARN,
)


# =========================================================================== #
#  Application des options EMIR Art. 28
# =========================================================================== #
def apply_buffer_option_a(
    im_raw: float,
    buffer_pct: float = DEFAULT_APC_BUFFER_PCT,
) -> dict[str, float]:
    """Option (a) EMIR : majoration de l'IM par un buffer en periode calme.

    L'IM publiee = IM_raw * (1 + buffer_pct), avec le buffer librable
    progressivement en periode de stress.

    Parameters
    ----------
    im_raw : float
        IM brute (sortie de compute_initial_margin).
    buffer_pct : float
        Pourcentage de buffer (defaut 25%).

    Returns
    -------
    dict
        IM avec buffer + composantes pour reporting.
    """
    buffer = im_raw * buffer_pct
    return {
        "im_raw": im_raw,
        "buffer": buffer,
        "im_with_buffer": im_raw + buffer,
        "buffer_pct": buffer_pct,
    }


def apply_floor_option_c(
    im_history: pd.Series,
    floor_pct: float = DEFAULT_APC_FLOOR_PCT,
    lookback_days: int = 2500,
) -> pd.Series:
    """Option (c) EMIR : plancher base sur l'IM max historique.

    IM_floored(t) = max(IM(t), floor_pct * max(IM[t-lookback : t]))

    Parameters
    ----------
    im_history : pd.Series
        Serie historique de l'IM brute, indexee par date.
    floor_pct : float
        Coefficient appliquee au max historique (defaut 25%).
    lookback_days : int
        Fenetre de calcul du max (defaut 10 ans).

    Returns
    -------
    pd.Series
        IM avec plancher applique.
    """
    if im_history.empty:
        return im_history.copy()

    rolling_max = im_history.rolling(
        window=lookback_days, min_periods=1
    ).max()
    floor = floor_pct * rolling_max
    return im_history.combine(floor, max)


# =========================================================================== #
#  Indicateurs APC pour reporting
# =========================================================================== #
@dataclass
class APCMetrics:
    """Indicateurs anti-procyclicite consolides."""
    n_obs: int
    im_mean: float
    im_std: float
    im_min: float
    im_max: float
    apc_ratio: float                # min / max sur fenetre
    daily_change_mean: float
    daily_change_std: float
    daily_change_p95: float
    daily_change_p99: float
    max_daily_increase: float
    max_daily_decrease: float
    n_warnings: int                 # jours ou |delta| > seuil
    max_n_day_increase: dict        # pire hausse cumulee sur N jours
    peak_to_trough_max: float       # amplitude max
    floor_binding_pct: float | None  # si option (c) appliquee


def compute_apc_metrics(
    im_history: pd.Series,
    apc_window: int = 250,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60),
    warn_threshold: float = MAX_DAILY_IM_CHANGE_WARN,
    floored_history: pd.Series | None = None,
) -> APCMetrics:
    """Calcule les indicateurs APC sur une serie historique d'IM.

    Parameters
    ----------
    im_history : pd.Series
        Serie historique de l'IM brute (sans plancher option c).
    apc_window : int
        Fenetre roulante pour le calcul du peak-to-trough (defaut 250 = 1 an).
    horizons : tuple[int, ...]
        Horizons (jours) pour le pire call de marge cumule.
    warn_threshold : float
        Seuil de variation J/J au-dessus duquel on compte une alerte.
    floored_history : pd.Series | None
        IM avec plancher option (c) applique (pour calculer le binding).

    Returns
    -------
    APCMetrics
    """
    if im_history.empty:
        raise ValueError("im_history est vide")

    s = im_history.sort_index()
    daily_change = s.pct_change().dropna()

    # Pire hausse cumulee sur N jours (call de marge maximal)
    max_n_day = {}
    for h in horizons:
        if len(s) > h:
            cumret = s / s.shift(h) - 1.0
            max_n_day[h] = float(cumret.max())
        else:
            max_n_day[h] = float("nan")

    # Peak-to-trough sur fenetre roulante
    rolling_max = s.rolling(apc_window, min_periods=1).max()
    rolling_min = s.rolling(apc_window, min_periods=1).min()
    peak_to_trough = (rolling_max - rolling_min) / rolling_max
    p2t_max = float(peak_to_trough.max())

    # Floor binding si l'on dispose de la version floored
    if floored_history is not None and not floored_history.empty:
        aligned = pd.DataFrame({
            "raw": s, "floored": floored_history.sort_index(),
        }).dropna()
        binding = (aligned["floored"] > aligned["raw"] + 1e-9).sum()
        floor_binding_pct = binding / len(aligned)
    else:
        floor_binding_pct = None

    return APCMetrics(
        n_obs=len(s),
        im_mean=float(s.mean()),
        im_std=float(s.std()),
        im_min=float(s.min()),
        im_max=float(s.max()),
        apc_ratio=float(s.min() / s.max()) if s.max() > 0 else float("nan"),
        daily_change_mean=float(daily_change.mean()),
        daily_change_std=float(daily_change.std()),
        daily_change_p95=float(daily_change.quantile(0.95)),
        daily_change_p99=float(daily_change.quantile(0.99)),
        max_daily_increase=float(daily_change.max()),
        max_daily_decrease=float(daily_change.min()),
        n_warnings=int((daily_change.abs() > warn_threshold).sum()),
        max_n_day_increase=max_n_day,
        peak_to_trough_max=p2t_max,
        floor_binding_pct=floor_binding_pct,
    )


def apc_report(metrics: APCMetrics) -> str:
    """Format texte d'un rapport APC, pret a inclure dans la documentation."""
    lines = [
        "=" * 65,
        " RAPPORT ANTI-PROCYCLICITE (APC) - EMIR Art. 28 / RTS 153/2013",
        "=" * 65,
        f"  Nombre d'observations          : {metrics.n_obs}",
        f"  IM moyenne                     : {metrics.im_mean:>12.4f}",
        f"  IM ecart-type                  : {metrics.im_std:>12.4f}",
        f"  IM min                         : {metrics.im_min:>12.4f}",
        f"  IM max                         : {metrics.im_max:>12.4f}",
        f"  Ratio APC (min/max)            : {metrics.apc_ratio:>12.4f}",
        "  ---------------------------------------------------------------",
        f"  Variation J/J moyenne          : {metrics.daily_change_mean:>+12.4%}",
        f"  Variation J/J ecart-type       : {metrics.daily_change_std:>12.4%}",
        f"  Variation J/J 95e percentile   : {metrics.daily_change_p95:>+12.4%}",
        f"  Variation J/J 99e percentile   : {metrics.daily_change_p99:>+12.4%}",
        f"  Pire hausse J/J                : {metrics.max_daily_increase:>+12.4%}",
        f"  Pire baisse J/J                : {metrics.max_daily_decrease:>+12.4%}",
        f"  Nb d'alertes (|delta|>seuil)   : {metrics.n_warnings}",
        "  ---------------------------------------------------------------",
        "  Pire call de marge cumule sur N jours :",
    ]
    for h, v in metrics.max_n_day_increase.items():
        lines.append(f"    {h:>3} jours : {v:>+12.4%}")
    lines.extend([
        "  ---------------------------------------------------------------",
        f"  Peak-to-trough max sur fenetre : {metrics.peak_to_trough_max:>12.4%}",
    ])
    if metrics.floor_binding_pct is not None:
        lines.append(
            f"  Plancher option (c) actif      : "
            f"{metrics.floor_binding_pct:>12.4%} des jours"
        )
    lines.append("=" * 65)
    return "\n".join(lines)
