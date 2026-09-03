"""
Schema and risk-scoring formulas shared between:
  - UnsupervisedKG.ipynb (which builds the initial graph + anomaly score)
  - fraud.py (the Streamlit app that visualizes the graph and handles reports)

Keeping the formulas in one place avoids the notebook and the app computing
the risk score in two slightly different ways (the classic team-integration
bug). Follows the project PDF's schema (sections 4, 10, 11, 12):

  Transaction { id, amount, timestamp, anomaly_score, cluster_id, risk_score, risk_class }
  Risk Score = alpha * Anomaly Evidence + beta * Report Evidence + gamma * Neighbour Risk (+ delta * Similarity Evidence, not implemented in the MVP)
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Node types / columns that become shareable entities in the graph
# (same list used in UnsupervisedKG.ipynb to build the edges)
# ---------------------------------------------------------------------------
ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "DeviceInfo", "DeviceType", "id_31", "id_30", "P_emaildomain",
]

# Human-readable labels for the report-submission form UI (PDF sections 8-9)
REPORTABLE_ENTITIES = {
    "Transaction (TransactionID)": "__transaction__",
    "Card (card1)": "card1",
    "Device (DeviceInfo)": "DeviceInfo",
    "Email domain (P_emaildomain)": "P_emaildomain",
    "External entity (phone / URL / bank account / other)": "__external__",
}

# Confidence weight per report type (PDF section 12, "Illustrative weight" table)
REPORT_CONFIDENCE_WEIGHTS = {
    "Verified evidence (e.g. confirmed chargeback)": 1.0,
    "Screenshot / transaction evidence": 0.8,
    "Detailed report": 0.6,
    "Minimal, unverified report": 0.3,
    "Duplicate / low-information report": 0.1,
}

# ---------------------------------------------------------------------------
# Risk-class thresholds (PDF section 10, "Example risk classes")
# ---------------------------------------------------------------------------
RISK_LOW_MAX = 0.39
RISK_SUSPICIOUS_MAX = 0.69

RISK_LOW = "Low risk"
RISK_SUSPICIOUS = "Suspicious"
RISK_HIGH = "High risk"

RISK_CLASS_ORDER = [RISK_LOW, RISK_SUSPICIOUS, RISK_HIGH]

RISK_CLASS_ICON = {
    RISK_LOW: "✅",
    RISK_SUSPICIOUS: "⚠️",
    RISK_HIGH: "🚨",
}

# Weights of the Risk Score formula (PDF section 10).
# Note: "Similarity Evidence" (delta) is reserved for a similarity-matching
# feature not implemented in this MVP (see PDF section 9, "Advanced features
# if time remains") - intentionally left at 0.
TX_ANOMALY_W = 0.50   # alpha
TX_REPORT_W = 0.30    # beta
TX_NEIGHBOUR_W = 0.20 # gamma
TX_SIMILARITY_W = 0.0 # delta (not implemented)

# With no reports at all yet, redistribute alpha/gamma so they remain a
# convex combination (they sum to 1).
_TX_BASE_SUM = TX_ANOMALY_W + TX_NEIGHBOUR_W
TX_BASELINE_ANOMALY_W = TX_ANOMALY_W / _TX_BASE_SUM
TX_BASELINE_NEIGHBOUR_W = TX_NEIGHBOUR_W / _TX_BASE_SUM

# Entities (card/device/email/external) have no anomaly score of their own:
# their risk comes only from reports and from their neighbours' risk.
ENTITY_REPORT_W = 0.6
ENTITY_NEIGHBOUR_W = 0.4

# Risk propagation to neighbours (PDF section 11)
PROPAGATION_DECAY_PER_HOP = 0.5
PROPAGATION_MAX_HOPS = 2
PROPAGATION_MIN_DELTA = 0.01  # below this we don't log/apply the update (noise)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def risk_class(score: float) -> str:
    if score >= 0.70:
        return RISK_HIGH
    if score >= 0.40:
        return RISK_SUSPICIOUS
    return RISK_LOW


def noisy_or(weights: list[float]) -> float:
    """Combine several independent reports into a single 'report evidence' in [0,1].

    We use a noisy-OR combination (1 - product(1-w_i)) instead of a plain
    average: several weak reports still add up, but a single 'verified'
    report (w=1.0) immediately saturates the signal - consistent with the
    PDF's idea that a report is evidence, not absolute proof, yet a strong
    one is still immediately strong evidence.
    """
    if not weights:
        return 0.0
    prod = 1.0
    for w in weights:
        prod *= (1.0 - clip01(w))
    return clip01(1.0 - prod)


def transaction_baseline_risk(anomaly_evidence: float, neighbour_risk: float) -> float:
    """Initial risk score of a transaction, before any report exists."""
    return clip01(
        TX_BASELINE_ANOMALY_W * clip01(anomaly_evidence)
        + TX_BASELINE_NEIGHBOUR_W * clip01(neighbour_risk)
    )


def transaction_risk(anomaly_evidence: float, report_evidence: float, neighbour_risk: float) -> float:
    """Full risk score (PDF section 10) once reports are linked to this node."""
    return clip01(
        TX_ANOMALY_W * clip01(anomaly_evidence)
        + TX_REPORT_W * clip01(report_evidence)
        + TX_NEIGHBOUR_W * clip01(neighbour_risk)
    )


def entity_risk(report_evidence: float, neighbour_risk: float) -> float:
    """Risk score of an entity (card/device/email/external): no anomaly score of its own."""
    if report_evidence <= 0:
        return clip01(neighbour_risk)
    return clip01(
        ENTITY_REPORT_W * clip01(report_evidence)
        + ENTITY_NEIGHBOUR_W * clip01(neighbour_risk)
    )


def propagation_influence(source_risk: float, edge_weight: float, hop: int) -> float:
    """Influence = Source Risk x Edge Weight x Decay^hop (PDF section 11)."""
    decay = PROPAGATION_DECAY_PER_HOP ** hop
    return clip01(source_risk) * clip01(edge_weight) * decay


def apply_influence(current_risk: float, influence: float) -> float:
    """Propagated risk pushes the value toward 1; it never replaces it and never lowers it."""
    current_risk = clip01(current_risk)
    return clip01(current_risk + influence * (1.0 - current_risk))


def entity_node_id(col: str, value) -> str:
    """Same node-naming convention used in UnsupervisedKG.ipynb: '{column}_{value}'."""
    return f"{col}_{value}"


def external_node_id(value: str) -> str:
    normalized = str(value).strip().lower()
    return f"External_{normalized}"
