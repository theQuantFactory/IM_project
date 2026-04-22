"""
Chargement de la matrice de taux zéro-coupon depuis un fichier CSV.

Le fichier attendu a une première colonne de dates et des colonnes de
maturités (piliers) en années. Le DataFrame résultant est indexé par date
avec des colonnes ``float``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_zero_coupon_curve(path: str | Path) -> pd.DataFrame:
    """Charge la courbe ZC depuis *path* et renvoie un DataFrame indexé par date.

    Parameters
    ----------
    path : str | Path
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
        Colonnes = piliers (float), index = dates (DatetimeIndex), trié.
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.columns = [float(col) for col in df.columns]
    return df.sort_index()
