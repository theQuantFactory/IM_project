"""
Configuration du modèle de calcul de la Marge Initiale.

Fournit ``ModelConfig``, un dataclass immutable encapsulant tous les
paramètres du modèle avec validation intégrée.

CHANGELOG (v1.1.0)
------------------
* Ajout : ``validate_against_data(returns_df)``
    Vérifie que la fenêtre LP+SW+1 demandée est effectivement disponible
    dans l'historique avant t0. Évite les KeyError silencieux ou les
    EWMA tronquées en pratique.

* Ajout : champs ``apc_buffer_pct``, ``apc_floor_pct``, ``model_version``
    Pour traçabilité et conformité ESMA Art. 28.

* Ajout : ``__post_init__`` exécutant ``validate()`` automatiquement.
    Empêche la création d'objets ModelConfig invalides (fail-fast).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.constants import (
    DEFAULT_ALPHA,
    DEFAULT_APC_BUFFER_PCT,
    DEFAULT_APC_FLOOR_PCT,
    DEFAULT_FHS_WEIGHT,
    DEFAULT_HP,
    DEFAULT_LAMBDA,
    DEFAULT_LP,
    DEFAULT_STRESS_END,
    DEFAULT_STRESS_START,
    DEFAULT_STRESS_WEIGHT,
    DEFAULT_SW,
    DEFAULT_T0,
    DEFAULT_ZC_NOMINAL,
    MIN_LOOKBACK_DAYS,
    MIN_STRESS_WINDOW_DAYS,
    MODEL_VERSION,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ModelConfig:
    """Paramétrage complet d'un run IM.

    Attributes
    ----------
    LP : int
        Lookback Period — nombre de scénarios FHS (defaut 2500 = ~10 ans).
    HP : int
        Holding Period — horizon de détention / MPOR (jours ouvrés).
    SW : int
        Smoothing Window — préchauffage EWMA.
    lambda_ewma : float
        Facteur de décroissance EWMA (in ]0, 1[).
    t0 : str
        Date d'évaluation (format ``YYYY-MM-DD``).
    stress_start, stress_end : str
        Bornes de la fenêtre de stress historique.
    alpha : float
        Niveau de confiance de l'Expected Shortfall (in ]0, 1[).
    FHS_w, Stress_w : float
        Pondérations des composantes (somme = 1).
    apc_buffer_pct : float
        Buffer APC option (a) EMIR.
    apc_floor_pct : float
        Plancher APC option (c) EMIR.
    metric : str
        Mesure de risque (``"ES"`` par défaut).
    nominal : float
        Nominal de référence des obligations ZC.
    model_version : str
        Version du modèle pour traçabilité (figée à l'instanciation).
    """

    LP: int = DEFAULT_LP
    HP: int = DEFAULT_HP
    SW: int = DEFAULT_SW
    lambda_ewma: float = DEFAULT_LAMBDA
    t0: str = DEFAULT_T0
    stress_start: str = DEFAULT_STRESS_START
    stress_end: str = DEFAULT_STRESS_END
    alpha: float = DEFAULT_ALPHA
    FHS_w: float = DEFAULT_FHS_WEIGHT
    Stress_w: float = DEFAULT_STRESS_WEIGHT
    apc_buffer_pct: float = DEFAULT_APC_BUFFER_PCT
    apc_floor_pct: float = DEFAULT_APC_FLOOR_PCT
    metric: str = "ES"
    nominal: float = DEFAULT_ZC_NOMINAL
    model_version: str = field(default=MODEL_VERSION)

    def __post_init__(self) -> None:
        """Validation automatique à la construction (fail-fast)."""
        self.validate()

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Vérifie la cohérence interne des paramètres.

        Raises
        ------
        ValueError
            Si un paramètre est hors domaine ou incohérent.
        """
        if self.LP < MIN_LOOKBACK_DAYS:
            raise ValueError(
                f"LP doit etre >= {MIN_LOOKBACK_DAYS} (1 an EMIR), recu {self.LP}"
            )
        if self.HP < 1:
            raise ValueError(f"HP doit etre >= 1, recu {self.HP}")
        if self.SW < 0:
            raise ValueError(f"SW doit etre >= 0, recu {self.SW}")
        if not (0.0 < self.lambda_ewma < 1.0):
            raise ValueError(
                f"lambda_ewma doit etre dans ]0, 1[, recu {self.lambda_ewma}"
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha doit etre dans ]0, 1[, recu {self.alpha}")
        if abs(self.FHS_w + self.Stress_w - 1.0) > 1e-9:
            raise ValueError(
                f"FHS_w + Stress_w doit valoir 1.0, recu "
                f"{self.FHS_w + self.Stress_w}"
            )
        if not (0.0 <= self.FHS_w <= 1.0):
            raise ValueError(f"FHS_w doit etre dans [0, 1], recu {self.FHS_w}")
        if self.nominal <= 0:
            raise ValueError(f"nominal doit etre > 0, recu {self.nominal}")
        if self.stress_start >= self.stress_end:
            raise ValueError("stress_start doit preceder stress_end")
        # Vérification que la fenêtre stress couvre au moins MIN_STRESS_WINDOW_DAYS
        import pandas as pd

        stress_business_days = len(
            pd.bdate_range(self.stress_start, self.stress_end)
        )
        if stress_business_days < MIN_STRESS_WINDOW_DAYS:
            raise ValueError(
                f"Fenetre stress trop courte : {stress_business_days} jours "
                f"ouvres < {MIN_STRESS_WINDOW_DAYS} requis"
            )

    def validate_against_data(self, returns_df: "pd.DataFrame") -> None:
        """Vérifie que la fenêtre demandée est disponible dans les données.

        Cette validation doit être appelée APRÈS chargement des returns,
        pour éviter qu'un EWMA soit calculé sur une fenêtre tronquée
        sans alerte.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Matrice de returns historiques (sortie de
            :func:`compute_historical_returns`).

        Raises
        ------
        ValueError
            Si l'historique disponible avant t0 est insuffisant.
        KeyError
            Si t0 n'est pas une date valide / dépasse l'historique.
        """
        import pandas as pd

        t0_dt = pd.to_datetime(self.t0)
        df_sorted = returns_df.sort_index()

        if t0_dt > df_sorted.index.max():
            raise KeyError(
                f"t0={self.t0} depasse la derniere date disponible "
                f"({df_sorted.index.max().date()})"
            )

        available = len(df_sorted.loc[:t0_dt])
        required = self.total_window
        if available < required:
            raise ValueError(
                f"Historique insuffisant avant t0={self.t0} : "
                f"{available} obs disponibles, {required} requises "
                f"(LP={self.LP} + SW={self.SW} + 1). "
                f"Reduire LP ou utiliser t0 ulterieur."
            )

        # Vérification fenêtre stress disponible
        stress_start_dt = pd.to_datetime(self.stress_start)
        stress_end_dt = pd.to_datetime(self.stress_end)
        stress_obs = df_sorted.loc[stress_start_dt:stress_end_dt]
        if len(stress_obs) < MIN_STRESS_WINDOW_DAYS:
            raise ValueError(
                f"Fenetre stress {self.stress_start} -> {self.stress_end} "
                f"contient seulement {len(stress_obs)} obs dans les donnees, "
                f"< {MIN_STRESS_WINDOW_DAYS} requises."
            )

    # ------------------------------------------------------------------ #
    #  Propriétés dérivées
    # ------------------------------------------------------------------ #
    @property
    def total_window(self) -> int:
        """Nombre total d'observations requises avant t0 : LP + SW + 1."""
        return self.LP + self.SW + 1

    @property
    def es_tail_size(self) -> int:
        """Taille de l'échantillon de queue ES : k = ceil((1-alpha) * LP).

        Délègue à ``_safe_tail_size`` pour neutraliser le bruit numérique
        flottant (cf. ``es.py`` CHANGELOG v1.1.1). Garantit la cohérence
        entre le k publié dans ``summary()`` et le k réellement utilisé
        par ``compute_expected_shortfall``.
        """
        from src.risk.es import _safe_tail_size

        return _safe_tail_size(self.alpha, self.LP)

    def summary(self) -> dict:
        """Résumé des paramètres pour reporting / audit."""
        return {
            "model_version": self.model_version,
            "t0": self.t0,
            "LP": self.LP,
            "HP": self.HP,
            "SW": self.SW,
            "lambda_ewma": self.lambda_ewma,
            "alpha": self.alpha,
            "es_tail_size_k": self.es_tail_size,
            "FHS_w": self.FHS_w,
            "Stress_w": self.Stress_w,
            "stress_window": f"{self.stress_start} -> {self.stress_end}",
            "apc_buffer_pct": self.apc_buffer_pct,
            "apc_floor_pct": self.apc_floor_pct,
        }
