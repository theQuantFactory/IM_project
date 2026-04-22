"""
Fonctions d'actualisation et d'interpolation de la courbe.

Fournit l'interpolation linéaire sur la courbe ZC (taux ou prix),
le calcul du facteur d'actualisation et du prix zéro-coupon.
L'extrapolation est plate aux bornes (convention standard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_curve(curve: pd.Series, maturity: float) -> float:
    """Interpole linéairement la courbe pour une maturité quelconque.

    Extrapolation plate : ``np.interp`` retourne la première valeur pour
    ``maturity < min(pillar)`` et la dernière pour ``maturity > max(pillar)``.

    Parameters
    ----------
    curve : pd.Series
        Courbe (taux ou prix) indexée par pilier (float).
    maturity : float
        Maturité cible (années).

    Returns
    -------
    float
        Valeur interpolée.
    """
    pillars = curve.index.astype(float).to_numpy()
    values = curve.to_numpy(dtype=float)
    return float(np.interp(maturity, pillars, values))


def get_discount_factor(curve: pd.Series, maturity: float) -> float:
    """Calcule le facteur d'actualisation DF(t₀, T) = 1 / (1 + r(T))^T.

    Parameters
    ----------
    curve : pd.Series
        Courbe de taux ZC.
    maturity : float
        Maturité en années.

    Returns
    -------
    float
        Facteur d'actualisation.
    """
    rate = interpolate_curve(curve, maturity)
    return 1.0 / (1.0 + rate) ** maturity


def get_zero_coupon_price(
    curve: pd.Series, maturity: float, nominal: float = 100.0
) -> float:
    """Prix d'une obligation zéro-coupon : N · DF(t₀, T).

    Parameters
    ----------
    curve : pd.Series
        Courbe de taux ZC.
    maturity : float
        Maturité en années.
    nominal : float
        Nominal (défaut 100).

    Returns
    -------
    float
        Prix de l'obligation ZC.
    """
    return nominal * get_discount_factor(curve, maturity)


def get_discount_factor_from_zc_price_curve(
    zc_price_curve: pd.Series,
    maturity: float,
    zc_nominal: float = 100.0,
) -> float:
    """DF depuis une courbe de prix ZC : P(t₀, T) / N.

    Parameters
    ----------
    zc_price_curve : pd.Series
        Courbe de prix ZC (index = piliers, valeurs = prix).
    maturity : float
        Maturité en années.
    zc_nominal : float
        Nominal des ZC (défaut 100).

    Returns
    -------
    float
        Facteur d'actualisation.
    """
    zc_price = interpolate_curve(zc_price_curve, maturity)
    return zc_price / zc_nominal
