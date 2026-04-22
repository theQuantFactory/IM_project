"""
Construction des facteurs de risque à partir de la courbe ZC.

Cf. Note méthodologique, §5 :
    - §5.1 : Prix ZC → P(t, T) = N / (1 + r(t, T))^T     (éq. 2)
    - §5.2 : Returns HP → R(t, T) = P(t, T) / P(t-HP, T) - 1  (éq. 3)
"""

from __future__ import annotations

import pandas as pd


def build_zero_coupon_price_matrix(
    zc_curve_df: pd.DataFrame,
    nominal: float = 100.0,
) -> pd.DataFrame:
    """Convertit la matrice de taux ZC en matrice de prix ZC.

    Parameters
    ----------
    zc_curve_df : pd.DataFrame
        Matrice de taux ZC (index = dates, colonnes = piliers float).
    nominal : float
        Nominal de référence (défaut 100).

    Returns
    -------
    pd.DataFrame
        Matrice de prix ZC, mêmes dimensions.
    """
    price_df = zc_curve_df.copy()
    for T in price_df.columns:
        price_df[T] = nominal / (1.0 + price_df[T]) ** float(T)
    return price_df


def compute_historical_returns(
    zc_price_df: pd.DataFrame,
    HP: int,
) -> pd.DataFrame:
    """Calcule les returns relatifs à horizon HP jours ouvrés.

    Parameters
    ----------
    zc_price_df : pd.DataFrame
        Matrice de prix ZC.
    HP : int
        Holding Period (nombre de jours ouvrés).

    Returns
    -------
    pd.DataFrame
        Returns relatifs, lignes avec NaN supprimées.
    """
    returns_df = zc_price_df / zc_price_df.shift(HP) - 1.0
    return returns_df.dropna()
