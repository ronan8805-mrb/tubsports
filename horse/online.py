"""
Continuous learning and model monitoring.
Lightweight online updates between full retrains.
"""

import json
import logging
from datetime import date
from typing import Dict

import numpy as np

from .config import DATA_DIR

logger = logging.getLogger(__name__)

MONITORING_PATH = DATA_DIR / "models" / "monitoring.json"


def rolling_metrics(
    predictions: list,
    actuals: list,
    window_days: int = 30,
) -> Dict[str, float]:
    """Compute rolling AUC, Brier, calibration from recent predictions."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    if len(predictions) < 20 or len(actuals) < 20:
        return {}

    preds = np.array(predictions[:len(actuals)])
    acts = np.array(actuals[:len(preds)])

    try:
        auc = roc_auc_score(acts, preds)
    except ValueError:
        auc = np.nan

    try:
        brier = brier_score_loss(acts, preds)
    except ValueError:
        brier = np.nan

    n_bins = 5
    cal_error = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            cal_error += abs(preds[mask].mean() - acts[mask].mean()) * mask.sum()
    ece = cal_error / len(preds)

    return {
        "rolling_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "rolling_brier": round(float(brier), 4),
        "rolling_ece": round(float(ece), 4),
        "sample_size": len(preds),
    }


def update_calibration(
    calibrator,
    new_predictions: np.ndarray,
    new_actuals: np.ndarray,
    max_samples: int = 5000,
):
    """Incrementally update isotonic calibration with new data.
    Returns updated calibrator.
    """
    from sklearn.isotonic import IsotonicRegression

    old_x = calibrator.X_thresholds_ if hasattr(calibrator, 'X_thresholds_') else np.array([])
    old_y = calibrator.y_thresholds_ if hasattr(calibrator, 'y_thresholds_') else np.array([])

    all_x = np.concatenate([old_x, new_predictions]) if len(old_x) > 0 else new_predictions
    all_y = np.concatenate([old_y, new_actuals]) if len(old_y) > 0 else new_actuals

    if len(all_x) > max_samples:
        idx = np.random.choice(len(all_x), max_samples, replace=False)
        all_x = all_x[idx]
        all_y = all_y[idx]

    new_cal = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    new_cal.fit(all_x, all_y)

    return new_cal


def detect_drift(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    auc_threshold: float = 0.03,
    brier_threshold: float = 0.01,
) -> Dict[str, bool]:
    """Detect if model performance has drifted from baseline."""
    alerts = {}

    c_auc = current_metrics.get("rolling_auc")
    b_auc = baseline_metrics.get("auc")
    if c_auc is not None and b_auc is not None:
        drift = b_auc - c_auc
        alerts["auc_degraded"] = drift > auc_threshold
        if alerts["auc_degraded"]:
            logger.warning(f"AUC drift detected: {b_auc:.4f} -> {c_auc:.4f} (delta={drift:.4f})")

    c_brier = current_metrics.get("rolling_brier")
    b_brier = baseline_metrics.get("brier")
    if c_brier is not None and b_brier is not None:
        drift = c_brier - b_brier
        alerts["brier_degraded"] = drift > brier_threshold
        if alerts["brier_degraded"]:
            logger.warning(f"Brier drift: {b_brier:.4f} -> {c_brier:.4f}")

    alerts["needs_retrain"] = any(alerts.values())
    return alerts


def save_monitoring_snapshot(metrics: dict):
    """Append monitoring metrics to history."""
    history = []
    if MONITORING_PATH.exists():
        try:
            history = json.loads(MONITORING_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    entry = {**metrics, "date": str(date.today())}
    history.append(entry)

    if len(history) > 365:
        history = history[-365:]

    MONITORING_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITORING_PATH.write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8"
    )


def load_monitoring_history() -> list:
    if MONITORING_PATH.exists():
        try:
            return json.loads(MONITORING_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


class ABModelTracker:
    """Track performance of two model versions side by side."""

    def __init__(self):
        self.model_a_preds = []
        self.model_b_preds = []
        self.actuals = []

    def record(self, pred_a: float, pred_b: float, actual: int):
        self.model_a_preds.append(pred_a)
        self.model_b_preds.append(pred_b)
        self.actuals.append(actual)

    def compare(self) -> Dict[str, dict]:
        if len(self.actuals) < 50:
            return {"status": "insufficient_data", "n": len(self.actuals)}

        metrics_a = rolling_metrics(self.model_a_preds, self.actuals)
        metrics_b = rolling_metrics(self.model_b_preds, self.actuals)

        winner = "A" if (metrics_a.get("rolling_auc", 0) or 0) >= (metrics_b.get("rolling_auc", 0) or 0) else "B"

        return {
            "model_a": metrics_a,
            "model_b": metrics_b,
            "winner": winner,
            "n": len(self.actuals),
        }
