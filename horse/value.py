"""
Value betting layer -- expected value, Kelly criterion, value scoring.
Sits on top of model probability predictions.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_VALUE_SCORE = 1.15
MIN_KELLY = 0.02
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.02


def expected_value(prob: float, decimal_odds: float) -> float:
    """EV = P(win) * (odds - 1) - (1 - P(win))"""
    if decimal_odds <= 1.0 or prob <= 0 or prob >= 1.0:
        return -1.0
    return prob * (decimal_odds - 1.0) - (1.0 - prob)


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full Kelly: f* = (p*b - q) / b where b = odds-1, q = 1-p"""
    if decimal_odds <= 1.0 or prob <= 0 or prob >= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(f, 0.0)


def value_score(model_prob: float, market_prob: float) -> float:
    """Ratio of model probability to market probability. >1 = value."""
    if market_prob <= 0:
        return 0.0
    return model_prob / market_prob


def remove_overround(odds_list: List[float]) -> List[float]:
    """Normalize bookmaker odds by removing overround.
    Returns implied probabilities summing to 1.0.
    """
    implied = [1.0 / o for o in odds_list if o > 1.0]
    if not implied:
        return []
    total = sum(implied)
    return [p / total for p in implied]


def compute_value_features(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    odds_col: str = "back_odds",
) -> pd.DataFrame:
    """Add value columns to a predictions DataFrame.

    Expects columns: prob_col (model probability), odds_col (decimal odds).
    Adds: ev, value_score, kelly, recommended_stake_pct, is_value_bet.
    """
    result = df.copy()

    if odds_col not in result.columns or prob_col not in result.columns:
        result["ev"] = np.nan
        result["value_score"] = np.nan
        result["kelly"] = np.nan
        result["recommended_stake_pct"] = np.nan
        result["is_value_bet"] = False
        return result

    probs = result[prob_col].values
    odds = result[odds_col].values

    evs = np.array([expected_value(p, o) for p, o in zip(probs, odds)])
    kellys = np.array([kelly_fraction(p, o) for p, o in zip(probs, odds)])

    market_probs = np.where(odds > 1.0, 1.0 / odds, 0.0)
    v_scores = np.where(market_probs > 0, probs / market_probs, 0.0)

    stakes = np.clip(kellys * KELLY_FRACTION, 0, MAX_STAKE_PCT)

    is_value = (v_scores >= MIN_VALUE_SCORE) & (kellys >= MIN_KELLY)

    result["ev"] = np.round(evs, 4)
    result["value_score"] = np.round(v_scores, 4)
    result["kelly"] = np.round(kellys, 4)
    result["recommended_stake_pct"] = np.round(stakes, 4)
    result["is_value_bet"] = is_value

    n_value = is_value.sum()
    if n_value > 0:
        logger.info(f"Value bets found: {n_value}/{len(result)}")

    return result


def rank_value_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and rank value bets by expected value."""
    if "is_value_bet" not in df.columns:
        return pd.DataFrame()

    value_df = df[df["is_value_bet"]].copy()
    if value_df.empty:
        return value_df

    value_df = value_df.sort_values("ev", ascending=False)
    return value_df
