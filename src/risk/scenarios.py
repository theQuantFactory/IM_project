"""
Construction des scénarios FHS scalés et des scénarios de stress bruts.

Cf. Note méthodologique, §7 :
    - §7.1, éq. (7-8) : Scaling symétrisé ϕ = (σ_{t0} + σ_t) / (2·σ_t)
    - §7.2, éq. (9) : Stress bruts (returns non rescalés)
"""

from __future__ import annotations

import pandas as pd


def compute_scaling_factors(
    vol_df: pd.DataFrame,
    t0: str,
) -> pd.DataFrame:
    """Calcule les facteurs de scaling FHS (formulation symétrisée).

    ϕ(t, T_k) = (σ_{t0}(T_k) + σ_t(T_k)) / (2 · σ_t(T_k))

    Parameters
    ----------
    vol_df : pd.DataFrame
        Matrice de volatilités EWMA.
    t0 : str
        Date d'évaluation.

    Returns
    -------
    pd.DataFrame
        Matrice de facteurs de scaling, mêmes dimensions que *vol_df*.
    """
    t0_dt = pd.to_datetime(t0)
    sigma_t0 = vol_df.loc[t0_dt]
    return vol_df.add(sigma_t0, axis=1) / (2.0 * vol_df)


def build_scaled_scenarios(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    t0: str,
    LP: int,
) -> pd.DataFrame:
    """Construit les LP scénarios FHS scalés.

    R̃(t, T_k) = ϕ(t, T_k) · R(t, T_k)

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns historiques.
    vol_df : pd.DataFrame
        Volatilités EWMA (même index que la fenêtre de returns).
    t0 : str
        Date d'évaluation.
    LP : int
        Nombre de scénarios à extraire.

    Returns
    -------
    pd.DataFrame
        Returns scalés (LP lignes × nb piliers colonnes).
    """
    t0_dt = pd.to_datetime(t0)
    unscaled_window = returns_df.loc[:t0_dt].tail(LP)
    vol_window = vol_df.loc[unscaled_window.index]
    scaling_factors = compute_scaling_factors(vol_window, t0)
    return scaling_factors * unscaled_window


def build_unscaled_scenarios(
    returns_df: pd.DataFrame,
    stress_start: str | None = None,
    stress_end: str | None = None,
    stressed_periods: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Construit les scénarios de stress bruts (non rescalés).

    R^stress(t, T_k) = R(t, T_k),  t ∈ [stress_start, stress_end]

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns historiques complets.
    stress_start, stress_end : str | None
        Bornes de la fenêtre de stress.
    stressed_periods : pd.DatetimeIndex | None
        Index de dates stressées (alternatif à start/end).

    Returns
    -------
    pd.DataFrame
        Returns bruts sur la période de stress.
    """
    returns_df = returns_df.sort_index()

    if stressed_periods is not None:
        idx = pd.DatetimeIndex(stressed_periods)
        return returns_df.loc[returns_df.index.intersection(idx)].dropna()

    if stress_start is None or stress_end is None:
        raise ValueError(
            "Fournir stressed_periods ou stress_start + stress_end."
        )

    start = pd.to_datetime(stress_start)
    end = pd.to_datetime(stress_end)
    return returns_df.loc[start:end].dropna()
