"""
Paramètres par défaut du modèle IM — cf. Note méthodologique, Tableau 4.

Ces constantes servent de valeurs par défaut pour ``ModelConfig``.
Elles peuvent être surchargées à l'instanciation.

CHANGELOG (v1.1.0)
------------------
* DEFAULT_LP : 250 → 2500
    Justification : avec LP=250 et alpha=0.99, l'ES est calculé sur
    k = ceil(0.01 * 250) = 3 observations seulement, conduisant à une
    erreur d'estimation de l'ordre de 60-80 % et à un effet "cliff"
    procyclique inacceptable pour un usage CCP.

    Cette extension à 10 ans (~2500 jours ouvrés) :
      - porte k à 25, ramenant l'erreur d'estimation à ~20 % ;
      - couvre désormais 2015-2025, intégrant le bull obligataire,
        le retournement 2022-2023 et la stabilisation 2024-2025 ;
      - aligne le modèle sur l'option (b) du RTS 153/2013 d'ESMA
        ("lookback period d'au moins 10 ans") parmi les trois options
        anti-procyclicité prévues à l'article 28 d'EMIR.

* Ajout : DEFAULT_APC_FLOOR_PCT (option a EMIR), DEFAULT_APC_BUFFER_PCT
* Ajout : MIN_LOOKBACK_DAYS, MIN_STRESS_WINDOW_DAYS pour la validation
"""

# --------------------------------------------------------------------------- #
#  Fenêtre et horizon
# --------------------------------------------------------------------------- #
DEFAULT_LP: int = 2500         # Lookback Period (~10 ans) — cf. CHANGELOG
DEFAULT_HP: int = 5            # Holding Period / MPOR (jours ouvrés)
DEFAULT_SW: int = 60           # Smoothing Window (préchauffage EWMA)

# --------------------------------------------------------------------------- #
#  EWMA
# --------------------------------------------------------------------------- #
DEFAULT_LAMBDA: float = 0.94   # Facteur de décroissance RiskMetrics mensuel

# --------------------------------------------------------------------------- #
#  Expected Shortfall
# --------------------------------------------------------------------------- #
DEFAULT_ALPHA: float = 0.99    # Niveau de confiance ES

# --------------------------------------------------------------------------- #
#  Pondération hybride FHS-Stress
# --------------------------------------------------------------------------- #
DEFAULT_FHS_WEIGHT: float = 0.75
DEFAULT_STRESS_WEIGHT: float = 0.25

# --------------------------------------------------------------------------- #
#  Anti-procyclicité (EMIR Art. 28, options du RTS 153/2013)
# --------------------------------------------------------------------------- #
# Option (a) : buffer constitué en période calme, libérable en stress.
DEFAULT_APC_BUFFER_PCT: float = 0.25
# Option (c) : plancher = max IM(t) sur les 10 dernières années / coefficient.
DEFAULT_APC_FLOOR_PCT: float = 0.25

# --------------------------------------------------------------------------- #
#  Bornes de validation (cohérence des données vs paramètres)
# --------------------------------------------------------------------------- #
MIN_LOOKBACK_DAYS: int = 250          # 1 an minimum exigé par EMIR
MIN_STRESS_WINDOW_DAYS: int = 60      # ~3 mois minimum pour un stress utile
MAX_DAILY_IM_CHANGE_WARN: float = 0.10  # Seuil d'alerte APC (>10% J/J)

# --------------------------------------------------------------------------- #
#  Nominal de référence des obligations ZC
# --------------------------------------------------------------------------- #
DEFAULT_ZC_NOMINAL: float = 100.0

# --------------------------------------------------------------------------- #
#  Dates de référence
# --------------------------------------------------------------------------- #
DEFAULT_T0: str = "2025-05-30"
DEFAULT_STRESS_START: str = "2022-01-01"
DEFAULT_STRESS_END: str = "2023-12-31"

# --------------------------------------------------------------------------- #
#  Grille de piliers (maturités en années)
# --------------------------------------------------------------------------- #
PILLAR_GRID: list[float] = [
    0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
]

# --------------------------------------------------------------------------- #
#  Métadonnées modèle (pour gouvernance et reporting)
# --------------------------------------------------------------------------- #
MODEL_VERSION: str = "1.1.0"
MODEL_NAME: str = "IM_CCP_HYBRID_FHS_STRESS"