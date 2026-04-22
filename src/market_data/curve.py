"""
Extraction d'une courbe zéro-coupon à une date donnée et interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_curve_at_date(df: pd.DataFrame, date: str) -> pd.Series:
    """Extrait la courbe ZC à la date *date*.

    Parameters
    ----------
    df : pd.DataFrame
        Matrice de taux ZC indexée par date.
    date : str
        Date au format ``YYYY-MM-DD``.

    Returns
    -------
    pd.Series
        Taux ZC par pilier à la date demandée.

    Raises
    ------
    KeyError
        Si *date* n'existe pas dans l'index.
    """
    return df.loc[pd.to_datetime(date)]


def interpolate_zc_rate(curve: pd.Series, maturity: float) -> float:
    """Interpole linéairement un taux ZC pour une maturité intermédiaire.

    Parameters
    ----------
    curve : pd.Series
        Courbe ZC (index = piliers float, valeurs = taux).
    maturity : float
        Maturité cible (années).

    Returns
    -------
    float
        Taux interpolé.
    """
    pillars = curve.index.astype(float).to_numpy()
    rates = curve.to_numpy(dtype=float)
    return float(np.interp(maturity, pillars, rates))
