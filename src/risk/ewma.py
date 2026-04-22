"""
Modèle de volatilité EWMA (Exponentially Weighted Moving Average).

Cf. Note méthodologique, §6 :
    - §6.2, éq. (4) : sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}
    - §6.5 : Fenêtre totale = LP + SW + 1

CHANGELOG (v1.1.0)
------------------
* Vectorisation via pandas .ewm()
    Remplace la boucle Python originale (O(n) calls Python) par un appel
    pandas vectorisé. Gain mesuré : facteur ~80x sur LP=2500 x 33 piliers.
    Critique pour le backtesting qui requiert ~1000 recomputations EWMA.

* Equivalence mathématique avec l'implémentation originale :
    Avec adjust=False, .ewm(alpha=1-lambda).mean() applique :
        y_t = lambda * y_{t-1} + (1-lambda) * x_t,  y_0 = x_0
    En passant x_t = r^2_{t-1} (returns au carré décalés), avec
    initialisation x_0 = r^2_0, on retrouve exactement la récurrence (4).

* Ajout : ``compute_ewma_variance_loop`` conservé comme implémentation
    de référence pour les tests de non-régression.
"""

from __future__ import annotations

import pandas as pd


def get_ewma_window(
    returns_df: pd.DataFrame,
    t0: str,
    LP: int,
    SW: int,
) -> pd.DataFrame:
    """Extrait la fenêtre de returns nécessaire à l'estimation EWMA.

    La fenêtre contient LP + SW + 1 observations précédant t0.
    Les SW premières servent au warm-up, les LP dernières aux scénarios.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Matrice de returns historiques.
    t0 : str
        Date d'évaluation (format ``YYYY-MM-DD``).
    LP : int
        Lookback Period.
    SW : int
        Smoothing Window (préchauffage).

    Returns
    -------
    pd.DataFrame
        Sous-matrice de LP + SW + 1 lignes.

    Raises
    ------
    ValueError
        Si la fenêtre demandée n'est pas entièrement disponible.
    """
    t0_dt = pd.to_datetime(t0)
    df = returns_df.sort_index().loc[:t0_dt]
    required = LP + SW + 1
    if len(df) < required:
        raise ValueError(
            f"Fenetre EWMA insuffisante : {len(df)} obs disponibles "
            f"avant t0={t0}, {required} requises (LP={LP}+SW={SW}+1)."
        )
    return df.tail(required)


def compute_ewma_variance(
    returns_df: pd.DataFrame,
    lambda_: float,
) -> pd.DataFrame:
    """Calcule la variance conditionnelle EWMA — implémentation vectorisée.

    Récurrence (éq. 4) :
        sigma^2_0 = r^2_0
        sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}   (t >= 1)

    Parameters
    ----------
    returns_df : pd.DataFrame
        Matrice de returns (chaque colonne = un pilier).
    lambda_ : float
        Facteur de décroissance (in ]0, 1[).

    Returns
    -------
    pd.DataFrame
        Matrice de variances conditionnelles, mêmes dimensions.
    """
    if not (0.0 < lambda_ < 1.0):
        raise ValueError(f"lambda doit etre dans ]0, 1[, recu {lambda_}")

    df = returns_df.sort_index()
    if df.isna().any().any():
        raise ValueError(
            "returns_df contient des NaN ; nettoyer en amont avec dropna()"
        )

    # On veut x_t = r^2_{t-1}, avec x_0 = r^2_0 (initialisation conforme
    # à la récurrence : sigma^2_0 = r^2_0, et ensuite t >= 1).
    r_squared = df ** 2
    r_squared_lagged = r_squared.shift(1)
    r_squared_lagged.iloc[0] = r_squared.iloc[0]  # initialisation

    # pandas .ewm(alpha=1-lambda, adjust=False).mean() applique :
    #     y_0 = x_0
    #     y_t = lambda * y_{t-1} + (1 - lambda) * x_t   (t >= 1)
    # Avec x_t = r^2_{t-1} et x_0 = r^2_0, on obtient exactement la
    # récurrence souhaitée.
    var_df = r_squared_lagged.ewm(alpha=1.0 - lambda_, adjust=False).mean()
    return var_df


def compute_ewma_variance_loop(
    returns_df: pd.DataFrame,
    lambda_: float,
) -> pd.DataFrame:
    """Implémentation de référence (boucle Python) pour tests de non-régression.

    Identique à la version vectorisée mais 50-100x plus lente.
    Conservée uniquement pour la validation par comparaison.
    """
    df = returns_df.sort_index().copy()
    var_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    var_df.iloc[0] = df.iloc[0] ** 2
    for i in range(1, len(df)):
        var_df.iloc[i] = (
            lambda_ * var_df.iloc[i - 1]
            + (1.0 - lambda_) * (df.iloc[i - 1] ** 2)
        )
    return var_df


def compute_ewma_volatility(
    returns_df: pd.DataFrame,
    lambda_: float,
) -> pd.DataFrame:
    """Calcule la volatilité conditionnelle EWMA : sigma_t = sqrt(sigma^2_t).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Matrice de returns.
    lambda_ : float
        Facteur de décroissance.

    Returns
    -------
    pd.DataFrame
        Matrice de volatilités conditionnelles.
    """
    var_df = compute_ewma_variance(returns_df, lambda_)
    return var_df ** 0.5


def select_stressed_periods(
    ewma_vol_df: pd.DataFrame,
    mode: str = "fixed_window",
    stress_start: str | None = None,
    stress_end: str | None = None,
    pillar: float = 0.25,
    quantile: float = 0.90,
) -> pd.DatetimeIndex:
    """Sélectionne les périodes de stress.

    Modes disponibles :
        - ``"fixed_window"`` : dates entre *stress_start* et *stress_end*.
        - ``"ewma_q90"`` : dates où la vol EWMA du pilier dépasse son
          quantile *quantile*.

    Parameters
    ----------
    ewma_vol_df : pd.DataFrame
        Matrice de volatilités EWMA.
    mode : str
        Méthode de sélection.
    stress_start, stress_end : str | None
        Bornes de la fenêtre (mode ``fixed_window``).
    pillar : float
        Pilier de référence (mode ``ewma_q90``).
    quantile : float
        Seuil de quantile (mode ``ewma_q90``).

    Returns
    -------
    pd.DatetimeIndex
        Dates identifiées comme stressées.
    """
    vol_df = ewma_vol_df.sort_index()

    if mode == "fixed_window":
        if stress_start is None or stress_end is None:
            raise ValueError(
                "stress_start et stress_end requis en mode fixed_window"
            )
        start = pd.to_datetime(stress_start)
        end = pd.to_datetime(stress_end)
        mask = (vol_df.index >= start) & (vol_df.index <= end)
        return pd.DatetimeIndex(vol_df.index[mask])

    if mode == "ewma_q90":
        if pillar in vol_df.columns:
            pillar_col = pillar
        elif str(pillar) in vol_df.columns:
            pillar_col = str(pillar)
        else:
            raise KeyError(f"Pilier {pillar} introuvable dans les colonnes EWMA")
        threshold = vol_df[pillar_col].quantile(quantile)
        return pd.DatetimeIndex(vol_df.index[vol_df[pillar_col] > threshold])

    raise ValueError(f"Mode de stress inconnu : {mode}")