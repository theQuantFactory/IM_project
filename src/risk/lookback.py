"""
Utilitaires de fenêtrage (lookback) pour l'extraction de sous-ensembles
temporels de données.
"""

from __future__ import annotations

import pandas as pd


def get_effective_lookback_length(
    df: pd.DataFrame,
    t0: str,
    requested_lp: int,
) -> int:
    """Renvoie le nombre effectif de scénarios disponibles avant t0.

    Borne supérieure : min(requested_lp, nb observations disponibles).

    Parameters
    ----------
    df : pd.DataFrame
        Données indexées par date.
    t0 : str
        Date d'évaluation.
    requested_lp : int
        Lookback Period demandé.

    Returns
    -------
    int
        Nombre effectif d'observations utilisables.
    """
    t0_dt = pd.to_datetime(t0)
    available = len(df.sort_index().loc[:t0_dt])
    return min(int(requested_lp), int(available))


def get_lookback_window(
    df: pd.DataFrame,
    t0: str,
    LP: int,
) -> pd.DataFrame:
    """Extrait les LP dernières observations avant t0 (inclus).

    Parameters
    ----------
    df : pd.DataFrame
        Données indexées par date.
    t0 : str
        Date d'évaluation.
    LP : int
        Nombre d'observations à extraire.

    Returns
    -------
    pd.DataFrame
        Sous-DataFrame de LP lignes.
    """
    t0_dt = pd.to_datetime(t0)
    return df.sort_index().loc[:t0_dt].tail(LP)
