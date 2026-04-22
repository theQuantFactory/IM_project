"""
Calcul de l'Expected Shortfall empirique.

Cf. Note méthodologique, §10, éq. (18) :
    ES_alpha = (1/k) * Sigma_{j=1}^{k} L_{(j)}
    avec k = ceil((1 - alpha) * n)

Propriete : mesure de risque coherente (Artzner et al., 1999).

CHANGELOG (v1.1.1)
------------------
* CORRECTION : bug numerique sur le calcul de k.
    L'expression `math.ceil((1 - alpha) * n)` produisait k=26 au lieu
    de k=25 pour alpha=0.99 et n=2500, parce que `1 - 0.99` vaut
    0.010000000000000009 en flottant double precision, donc le produit
    donne 25.000000000000023, ceil -> 26.

    Conséquence avant correction : ES dilué d'environ 4 % (la 26e pire
    perte est plus petite que les 25 premières et tire la moyenne vers
    le bas), entrainant une sous-estimation de l'IM.

    Correction : arrondi a 12 décimales avant ceil pour neutraliser le
    bruit machine sans affecter les cas où k devrait légitimement etre
    arrondi vers le haut (ex: alpha=0.975, n=250 -> k=ceil(6.25)=7).
"""

from __future__ import annotations

import math

import pandas as pd


def _safe_tail_size(alpha: float, n: int) -> int:
    """Calcule k = ceil((1-alpha) * n) en neutralisant le bruit flottant.

    Parameters
    ----------
    alpha : float
        Niveau de confiance (in ]0, 1[).
    n : int
        Taille de l'echantillon.

    Returns
    -------
    int
        Nombre d'observations en queue, borne inferieure 1.

    Examples
    --------
    >>> _safe_tail_size(0.99, 2500)   # avant: 26 (bug), apres: 25
    25
    >>> _safe_tail_size(0.975, 250)   # arrondi legitime: 6.25 -> 7
    7
    >>> _safe_tail_size(0.99, 250)    # 2.5 -> 3
    3
    """
    raw = (1.0 - alpha) * n
    # Arrondi a 12 decimales : suffisant pour neutraliser ~1e-15 d'erreur
    # de double precision, insuffisant pour affecter un vrai 0.5 ou 0.25
    return max(1, math.ceil(round(raw, 12)))


def compute_expected_shortfall(
    losses: pd.Series,
    alpha: float = 0.99,
) -> float:
    """ES empirique depuis une serie de **pertes** (positif = perte).

    Parameters
    ----------
    losses : pd.Series
        Pertes (L = -PnL).
    alpha : float
        Niveau de confiance (ex. 0.99 -> moyenne des 1 % pires).

    Returns
    -------
    float
        Expected Shortfall.
    """
    losses = pd.Series(losses).dropna().sort_values(ascending=False)
    n = len(losses)
    if n == 0:
        raise ValueError("Serie de pertes vide")
    k = _safe_tail_size(alpha, n)
    return float(losses.iloc[:k].mean())


def compute_es_from_pnl(
    pnl: pd.Series,
    alpha: float = 0.99,
) -> float:
    """ES depuis une serie de **PnL** (conversion automatique en pertes)."""
    losses = -pd.Series(pnl).dropna()
    return compute_expected_shortfall(losses, alpha=alpha)


def compute_expected_shortfall_from_tail(
    losses: pd.Series,
    var_level: float = 0.05,
) -> float:
    """ES avec *var_level* interprete comme probabilite de queue."""
    losses = pd.Series(losses).dropna().sort_values(ascending=False)
    n = len(losses)
    if n == 0:
        raise ValueError("Serie de pertes vide")
    k = max(1, math.ceil(round(var_level * n, 12)))
    return float(losses.iloc[:k].mean())


def compute_es_from_pnl_tail(
    pnl: pd.Series,
    var_level: float = 0.05,
) -> float:
    """ES depuis PnL avec probabilite de queue."""
    losses = -pd.Series(pnl).dropna()
    return compute_expected_shortfall_from_tail(losses, var_level=var_level)