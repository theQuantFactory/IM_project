"""
Calcul du PnL d'un portefeuille obligataire sous scénarios.

Cf. Note méthodologique, §9 :
    - Éq. (16) : P^(s)(t₀, T_k) = P(t₀, T_k) · (1 + R̃^(s)(T_k))
    - Éq. (17) : PnL^(s) = V^(s) - V(t₀)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.pricing.bonds import price_fixed_rate_bond_from_zc_prices


def apply_scenario_returns_to_current_curve(
    current_zc_price_curve: pd.Series,
    scenario_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Applique les returns scénario à la courbe courante (perturbation multiplicative).

    P^(s)(t₀, T_k) = P(t₀, T_k) · (1 + R̃^(s)(T_k))

    Parameters
    ----------
    current_zc_price_curve : pd.Series
        Prix ZC courants (un par pilier).
    scenario_returns_df : pd.DataFrame
        Returns scénarios (lignes = scénarios, colonnes = piliers).

    Returns
    -------
    pd.DataFrame
        Courbes de prix ZC stressées (lignes = scénarios).
    """
    return scenario_returns_df.mul(current_zc_price_curve, axis=1) + current_zc_price_curve


def price_bond_position_from_zc_prices(
    zc_price_curve: pd.Series,
    position: dict[str, Any],
    zc_nominal: float = 100.0,
) -> float:
    """Valorise une position obligataire (quantité × prix unitaire).

    Parameters
    ----------
    zc_price_curve : pd.Series
        Courbe de prix ZC.
    position : dict
        Dictionnaire décrivant la position (maturity, coupon_rate,
        nominal, frequency, quantity).
    zc_nominal : float
        Nominal des ZC de la courbe.

    Returns
    -------
    float
        Valeur de la position.
    """
    unit_price = price_fixed_rate_bond_from_zc_prices(
        zc_price_curve=zc_price_curve,
        maturity=position["maturity"],
        coupon_rate=position["coupon_rate"],
        nominal=position.get("nominal", 100.0),
        frequency=position.get("frequency", 1),
        zc_nominal=zc_nominal,
    )
    return position.get("quantity", 1) * unit_price


def compute_portfolio_initial_value(
    current_zc_price_curve: pd.Series,
    portfolio: list[dict[str, Any]],
    zc_nominal: float = 100.0,
) -> float:
    """Calcule la valeur mark-to-market du portefeuille à t₀.

    Parameters
    ----------
    current_zc_price_curve : pd.Series
        Prix ZC courants.
    portfolio : list[dict]
        Liste de positions obligataires.
    zc_nominal : float
        Nominal des ZC.

    Returns
    -------
    float
        Valeur totale V(t₀).
    """
    return sum(
        price_bond_position_from_zc_prices(current_zc_price_curve, pos, zc_nominal)
        for pos in portfolio
    )


def compute_portfolio_values_under_scenarios(
    current_zc_price_curve: pd.Series,
    scenario_returns_df: pd.DataFrame,
    portfolio: list[dict[str, Any]],
    zc_nominal: float = 100.0,
) -> pd.Series:
    """Calcule la valeur du portefeuille sous chaque scénario (full revaluation).

    Parameters
    ----------
    current_zc_price_curve : pd.Series
        Prix ZC courants.
    scenario_returns_df : pd.DataFrame
        Returns scénarios.
    portfolio : list[dict]
        Positions obligataires.
    zc_nominal : float
        Nominal des ZC.

    Returns
    -------
    pd.Series
        Valeur V^(s) par scénario.
    """
    stressed_curves = apply_scenario_returns_to_current_curve(
        current_zc_price_curve, scenario_returns_df
    )

    return stressed_curves.apply(
        lambda curve: sum(
            price_bond_position_from_zc_prices(curve, pos, zc_nominal)
            for pos in portfolio
        ),
        axis=1,
    )


def compute_portfolio_pnl_under_scenarios(
    current_zc_price_curve: pd.Series,
    scenario_returns_df: pd.DataFrame,
    portfolio: list[dict[str, Any]],
    zc_nominal: float = 100.0,
) -> pd.Series:
    """Calcule le PnL du portefeuille sous chaque scénario.

    PnL^(s) = V^(s) - V(t₀)

    Parameters
    ----------
    current_zc_price_curve : pd.Series
        Prix ZC courants.
    scenario_returns_df : pd.DataFrame
        Returns scénarios.
    portfolio : list[dict]
        Positions obligataires.
    zc_nominal : float
        Nominal des ZC.

    Returns
    -------
    pd.Series
        PnL par scénario.
    """
    initial_value = compute_portfolio_initial_value(
        current_zc_price_curve, portfolio, zc_nominal
    )
    scenario_values = compute_portfolio_values_under_scenarios(
        current_zc_price_curve, scenario_returns_df, portfolio, zc_nominal
    )
    return scenario_values - initial_value


def compute_losses_from_pnl(pnl: pd.Series) -> pd.Series:
    """Convertit les PnL en pertes (convention : positif = perte).

    L^(s) = -PnL^(s)
    """
    return -pnl
