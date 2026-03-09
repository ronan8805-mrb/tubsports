"""
Race shape and pace analysis module.
Classifies run styles, computes pace scenarios, and predicts race shape.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .nlp import classify_run_style

logger = logging.getLogger(__name__)


STYLE_LABELS = {0: "unknown", 1: "front_runner", 2: "stalker", 3: "closer"}


def classify_horse_style(
    comments: List[Optional[str]],
    early_positions: List[Optional[int]] = None,
) -> Dict[str, float]:
    """Classify a horse's dominant run style from career data.

    Uses comment NLP as primary signal, with early position data as backup.
    Returns style encoding and confidence.
    """
    styles = [classify_run_style(c) for c in comments if c]
    styles = [s for s in styles if s > 0]

    if not styles:
        return {"run_style": np.nan, "style_confidence": np.nan}

    from collections import Counter
    counts = Counter(styles)
    dominant = counts.most_common(1)[0][0]
    confidence = counts[dominant] / len(styles)

    return {
        "run_style": float(dominant),
        "style_confidence": round(confidence, 3),
    }


def compute_pace_scenario(run_styles: List[float]) -> Dict[str, float]:
    """Compute pace scenario features for a race field.

    Args:
        run_styles: list of run_style values for each runner in the field

    Returns:
        pace features dict
    """
    valid = [s for s in run_styles if not np.isnan(s)]
    if not valid:
        return {
            "n_front_runners": np.nan,
            "n_closers": np.nan,
            "pace_pressure": np.nan,
        }

    n_front = sum(1 for s in valid if s == 1.0)
    n_stalk = sum(1 for s in valid if s == 2.0)
    n_close = sum(1 for s in valid if s == 3.0)

    total = len(valid)
    pace_pressure = n_front / total if total > 0 else 0.0

    return {
        "n_front_runners": float(n_front),
        "n_closers": float(n_close),
        "pace_pressure": round(pace_pressure, 3),
    }


def pace_advantage_for_style(
    style: float,
    n_front_runners: int,
    course_pace_bias: float = 0.5,
) -> float:
    """Calculate pace advantage for a given run style.

    Strong pace (many front-runners) favours closers.
    Weak pace (few/no front-runners) favours front-runners.
    """
    if np.isnan(style):
        return np.nan

    if style == 1.0:
        if n_front_runners <= 1:
            base = 1.0
        elif n_front_runners == 2:
            base = 0.6
        else:
            base = 0.2
    elif style == 3.0:
        if n_front_runners >= 3:
            base = 1.0
        elif n_front_runners == 2:
            base = 0.7
        else:
            base = 0.3
    else:
        base = 0.5

    # Adjust for course bias (1.0 = galloping/favours closers, 0.0 = sharp/favours speed)
    if style == 3.0:
        base = base * (0.7 + 0.3 * course_pace_bias)
    elif style == 1.0:
        base = base * (0.7 + 0.3 * (1.0 - course_pace_bias))

    return round(min(base, 1.0), 3)


def predict_race_shape(
    field_styles: List[float],
    field_names: List[str] = None,
) -> Dict[str, Any]:
    """Predict the shape of a race from the field's run styles.

    Returns narrative and tactical assessment.
    """
    valid = [(s, n) for s, n in zip(field_styles, field_names or range(len(field_styles)))
             if not np.isnan(s)]

    if not valid:
        return {"shape": "unknown", "narrative": "Insufficient data"}

    n_front = sum(1 for s, _ in valid if s == 1.0)
    n_close = sum(1 for s, _ in valid if s == 3.0)

    if n_front >= 3:
        shape = "strong_pace"
        narrative = f"{n_front} confirmed front-runners -- expect strong pace, favours hold-up horses"
    elif n_front == 2:
        shape = "fair_pace"
        narrative = "Two pace-setters -- genuine pace likely, tactical race"
    elif n_front == 1:
        shape = "steady_pace"
        narrative = "Single pace-setter -- could dictate, front-runner advantage"
    else:
        shape = "no_pace"
        narrative = "No confirmed front-runners -- risk of muddling pace, advantage to those who can make their own"

    front_runners = [n for s, n in valid if s == 1.0]
    closers = [n for s, n in valid if s == 3.0]

    return {
        "shape": shape,
        "narrative": narrative,
        "n_front_runners": n_front,
        "n_closers": n_close,
        "front_runners": front_runners,
        "closers": closers,
        "favoured_style": "closer" if n_front >= 3 else ("front_runner" if n_front <= 1 else "stalker"),
    }
