"""
Valorisation d'obligations à taux fixe par actualisation des flux.

Cf. Note méthodologique, §8.1, équation (10) :
    V_bond = Σ C_k · DF(t₀, t_k)

Deux variantes :
    - depuis une courbe de taux ZC (interpolation → DF),
    - depuis une courbe de prix ZC (interpolation directe → DF = P/N).
"""

from __future__ import annotations

import pandas as pd

from src.pricing.discounting import (
    get_discount_factor,
    get_discount_factor_from_zc_price_curve,
)


def _generate_payment_times(maturity: float, frequency: int) -> list[float]:
    """Génère les dates de paiement en reculant depuis la maturité.

    Préserve la stub period éventuelle (premier coupon court).

    Parameters
    ----------
    maturity : float
        Maturité résiduelle en années.
    frequency : int
        Nombre de coupons par an.

    Returns
    -------
    list[float]
        Dates de paiement triées par ordre croissant.
    """
    dt = 1.0 / frequency
    payment_times: list[float] = []
    t = maturity
    while t > 1e-6:
        payment_times.append(t)
        t -= dt
    return sorted(payment_times)


def price_fixed_rate_bond(
    curve: pd.Series,
    maturity: float,
    coupon_rate: float,
    nominal: float = 100.0,
    frequency: int = 1,
) -> float:
    """Prix d'une obligation à taux fixe depuis une courbe de **taux** ZC.

    Parameters
    ----------
    curve : pd.Series
        Courbe de taux ZC.
    maturity : float
        Maturité résiduelle (années).
    coupon_rate : float
        Taux de coupon annuel (ex. 0.03 pour 3 %).
    nominal : float
        Nominal de l'obligation.
    frequency : int
        Fréquence de coupon (1 = annuel, 2 = semestriel).

    Returns
    -------
    float
        Prix (dirty price).
    """
    coupon = nominal * coupon_rate / frequency
    payment_times = _generate_payment_times(maturity, frequency)

    price = 0.0
    for i, t in enumerate(payment_times):
        cashflow = coupon
        if i == len(payment_times) - 1:
            cashflow += nominal
        price += cashflow * get_discount_factor(curve, t)

    return price


def price_fixed_rate_bond_from_zc_prices(
    zc_price_curve: pd.Series,
    maturity: float,
    coupon_rate: float,
    nominal: float = 100.0,
    frequency: int = 1,
    zc_nominal: float = 100.0,
) -> float:
    """Prix d'une obligation à taux fixe depuis une courbe de **prix** ZC.

    Parameters
    ----------
    zc_price_curve : pd.Series
        Courbe de prix ZC (index = piliers, valeurs = prix).
    maturity : float
        Maturité résiduelle (années).
    coupon_rate : float
        Taux de coupon annuel.
    nominal : float
        Nominal de l'obligation.
    frequency : int
        Fréquence de coupon.
    zc_nominal : float
        Nominal des obligations ZC de la courbe.

    Returns
    -------
    float
        Prix (dirty price).
    """
    coupon = nominal * coupon_rate / frequency
    payment_times = _generate_payment_times(maturity, frequency)

    price = 0.0
    for i, t in enumerate(payment_times):
        cashflow = coupon
        if i == len(payment_times) - 1:
            cashflow += nominal
        df = get_discount_factor_from_zc_price_curve(
            zc_price_curve, maturity=t, zc_nominal=zc_nominal
        )
        price += cashflow * df

    return price
