"""
Calcul de la Marge Initiale hybride FHS–Stress.

Cf. Note méthodologique, §11, éq. (19-20) :
    ES_hybrid = w_FHS · ES^FHS_α + w_stress · ES^stress_α
    IM = max(ES^FHS_α, ES_hybrid)

Le plancher (opérateur max) garantit que la composante stress ne réduit
jamais la marge en deçà de l'ES FHS courant.
"""

from __future__ import annotations


def compute_initial_margin(
    es_fhs: float,
    es_stress: float,
    fhs_w: float = 0.75,
    stress_w: float = 0.25,
) -> float:
    """Calcule la Marge Initiale hybride avec plancher.

    Parameters
    ----------
    es_fhs : float
        Expected Shortfall sur les scénarios FHS scalés.
    es_stress : float
        Expected Shortfall sur les scénarios de stress bruts.
    fhs_w : float
        Poids de la composante FHS (défaut 0.75).
    stress_w : float
        Poids de la composante stress (défaut 0.25).

    Returns
    -------
    float
        Marge Initiale : max(ES_FHS, w_FHS · ES_FHS + w_stress · ES_stress).
    """
    es_hybrid = fhs_w * es_fhs + stress_w * es_stress
    return max(es_fhs, es_hybrid)
