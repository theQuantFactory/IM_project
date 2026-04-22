"""
Pipeline de nettoyage de la matrice de taux zéro-coupon.

Étapes (cf. Note méthodologique, §4.2) :
    1. Déduplication — conservation de la première occurrence par date.
    2. Tri chronologique.
    3. Typage numérique — conversion en flottants, erreurs → NaN.
    4. Complétude — élimination des lignes contenant au moins un NaN.
"""

from __future__ import annotations

import pandas as pd


def clean_zero_coupon_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame de taux ZC en place et renvoie une copie propre.

    Parameters
    ----------
    df : pd.DataFrame
        Matrice de taux brute (sortie de :func:`load_zero_coupon_curve`).

    Returns
    -------
    pd.DataFrame
        Matrice nettoyée, sans doublons ni valeurs manquantes.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df
