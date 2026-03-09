"""
Staking strategy engine -- Kelly criterion, bank management, confidence tiers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

FRACTIONAL_KELLY = 0.25
MAX_STAKE_PCT = 0.02
DAILY_LOSS_LIMIT_PCT = 0.05


@dataclass
class BankState:
    """Tracks betting bank and daily limits."""
    starting_bank: float
    current_bank: float
    daily_profit: float = 0.0
    bets_today: int = 0
    total_bets: int = 0
    total_profit: float = 0.0

    @property
    def daily_loss_limit(self) -> float:
        return self.starting_bank * DAILY_LOSS_LIMIT_PCT

    @property
    def is_daily_limit_hit(self) -> bool:
        return self.daily_profit <= -self.daily_loss_limit

    def reset_daily(self):
        self.daily_profit = 0.0
        self.bets_today = 0


@dataclass
class StakeRecommendation:
    """Output of the staking engine for a single bet."""
    horse_name: str
    race_id: int
    model_prob: float
    decimal_odds: float
    kelly_full: float
    kelly_fractional: float
    stake_amount: float
    stake_pct: float
    ev: float
    value_score: float
    confidence_tier: str


def compute_kelly(prob: float, odds: float) -> float:
    if odds <= 1.0 or prob <= 0 or prob >= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    return max((prob * b - q) / b, 0.0)


def confidence_tier(value_score: float) -> str:
    if value_score >= 1.5:
        return "high"
    if value_score >= 1.3:
        return "medium"
    if value_score >= 1.15:
        return "low"
    return "none"


def tier_multiplier(tier: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4, "none": 0.0}.get(tier, 0.0)


def calculate_stake(
    prob: float,
    decimal_odds: float,
    bank: float,
    market_prob: Optional[float] = None,
) -> StakeRecommendation:
    """Calculate optimal stake for a single bet."""
    if decimal_odds <= 1.0:
        return StakeRecommendation(
            horse_name="", race_id=0,
            model_prob=prob, decimal_odds=decimal_odds,
            kelly_full=0, kelly_fractional=0,
            stake_amount=0, stake_pct=0, ev=-1, value_score=0,
            confidence_tier="none",
        )

    k_full = compute_kelly(prob, decimal_odds)
    k_frac = k_full * FRACTIONAL_KELLY

    if market_prob is None:
        market_prob = 1.0 / decimal_odds
    vs = prob / market_prob if market_prob > 0 else 0

    tier = confidence_tier(vs)
    mult = tier_multiplier(tier)

    stake_pct = min(k_frac * mult, MAX_STAKE_PCT)
    stake_amount = round(bank * stake_pct, 2)

    ev = prob * (decimal_odds - 1.0) - (1.0 - prob)

    return StakeRecommendation(
        horse_name="", race_id=0,
        model_prob=prob, decimal_odds=decimal_odds,
        kelly_full=round(k_full, 4),
        kelly_fractional=round(k_frac, 4),
        stake_amount=stake_amount,
        stake_pct=round(stake_pct, 4),
        ev=round(ev, 4),
        value_score=round(vs, 4),
        confidence_tier=tier,
    )
