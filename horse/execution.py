"""
Betfair Exchange API integration for automated bet placement.
Requires BETFAIR_APP_KEY and BETFAIR_SESSION_TOKEN in horse/.env

Pipeline:
  Model produces P(win) -> compare to exchange odds ->
  if value_score > threshold and kelly > min ->
  place bet at calculated stake -> log to execution_log
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATA_DIR
from .staking import BankState, calculate_stake
from .value import MIN_VALUE_SCORE, MIN_KELLY

logger = logging.getLogger(__name__)

BETFAIR_API_BASE = "https://api.betfair.com/exchange"
BETFAIR_APP_KEY = os.getenv("BETFAIR_APP_KEY", "")
BETFAIR_SESSION_TOKEN = os.getenv("BETFAIR_SESSION_TOKEN", "")

EXECUTION_LOG_PATH = DATA_DIR / "execution_log.json"


@dataclass
class BetRecord:
    bet_id: str
    race_id: int
    horse_name: str
    predicted_prob: float
    exchange_odds: float
    value_score: float
    kelly_fraction: float
    stake: float
    actual_odds_matched: Optional[float]
    result: Optional[str]
    pnl: Optional[float]
    placed_at: str


class BetfairClient:
    """Betfair Exchange API client."""

    def __init__(self):
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            self.session = None
        self.app_key = BETFAIR_APP_KEY
        self.session_token = BETFAIR_SESSION_TOKEN

    @property
    def is_configured(self) -> bool:
        return bool(self.app_key and self.session_token and self.session)

    def _headers(self) -> dict:
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def get_market_odds(self, market_id: str) -> Dict[str, float]:
        """Get current back odds for all runners in a market.
        Returns {selection_id: best_back_price}
        """
        if not self.is_configured:
            logger.warning("Betfair not configured")
            return {}

        url = f"{BETFAIR_API_BASE}/betting/rest/v1.0/listRunnerBook/"
        payload = {
            "marketId": market_id,
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        }

        try:
            resp = self.session.post(url, json=payload, headers=self._headers(), timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Betfair API error: {resp.status_code}")
                return {}
            data = resp.json()
            odds = {}
            for runner in data.get("runners", []):
                sel_id = str(runner.get("selectionId", ""))
                backs = runner.get("ex", {}).get("availableToBack", [])
                if backs:
                    odds[sel_id] = backs[0].get("price", 0)
            return odds
        except Exception as e:
            logger.warning(f"Betfair odds fetch failed: {e}")
            return {}

    def place_bet(
        self,
        market_id: str,
        selection_id: str,
        stake: float,
        price: float,
    ) -> Optional[str]:
        """Place a back bet on Betfair Exchange.
        Returns bet_id if successful.
        """
        if not self.is_configured:
            logger.warning("Betfair not configured -- dry run")
            return f"DRY_{int(time.time())}"

        url = f"{BETFAIR_API_BASE}/betting/rest/v1.0/placeOrders/"
        payload = {
            "marketId": market_id,
            "instructions": [{
                "selectionId": int(selection_id),
                "side": "BACK",
                "orderType": "LIMIT",
                "limitOrder": {
                    "size": round(stake, 2),
                    "price": price,
                    "persistenceType": "LAPSE",
                },
            }],
        }

        try:
            resp = self.session.post(url, json=payload, headers=self._headers(), timeout=10)
            data = resp.json()
            status = data.get("status", "")
            if status == "SUCCESS":
                bet_id = data["instructionReports"][0].get("betId", "")
                logger.info(f"Bet placed: {bet_id} - {stake} @ {price}")
                return str(bet_id)
            else:
                logger.warning(f"Bet failed: {data}")
                return None
        except Exception as e:
            logger.warning(f"Bet placement failed: {e}")
            return None


def log_bet(record: BetRecord):
    """Append bet record to execution log."""
    log_data = []
    if EXECUTION_LOG_PATH.exists():
        try:
            log_data = json.loads(EXECUTION_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    log_data.append({
        "bet_id": record.bet_id,
        "race_id": record.race_id,
        "horse_name": record.horse_name,
        "predicted_prob": record.predicted_prob,
        "exchange_odds": record.exchange_odds,
        "value_score": record.value_score,
        "kelly_fraction": record.kelly_fraction,
        "stake": record.stake,
        "actual_odds_matched": record.actual_odds_matched,
        "result": record.result,
        "pnl": record.pnl,
        "placed_at": record.placed_at,
    })

    EXECUTION_LOG_PATH.write_text(
        json.dumps(log_data, indent=2), encoding="utf-8"
    )


def evaluate_bets(
    predictions: list,
    bank_state: BankState,
    client: Optional[BetfairClient] = None,
    dry_run: bool = True,
) -> List[BetRecord]:
    """Evaluate predictions and place bets where value exists.

    predictions: list of dicts with keys:
        race_id, horse_name, model_prob, exchange_odds, market_id, selection_id

    Returns list of BetRecords for placed bets.
    """
    records = []

    if bank_state.is_daily_limit_hit:
        logger.info("Daily loss limit hit -- no more bets")
        return records

    for pred in predictions:
        prob = pred["model_prob"]
        odds = pred["exchange_odds"]

        if odds <= 1.0 or prob <= 0:
            continue

        market_prob = 1.0 / odds
        vs = prob / market_prob
        kelly = max((prob * (odds - 1) - (1 - prob)) / (odds - 1), 0)

        if vs < MIN_VALUE_SCORE or kelly < MIN_KELLY:
            continue

        rec = calculate_stake(prob, odds, bank_state.current_bank, market_prob)
        if rec.stake_amount < 2.0:
            continue

        bet_id = None
        if not dry_run and client and client.is_configured:
            bet_id = client.place_bet(
                pred.get("market_id", ""),
                pred.get("selection_id", ""),
                rec.stake_amount,
                odds,
            )
        else:
            bet_id = f"DRY_{int(time.time())}_{pred['horse_name'][:5]}"

        record = BetRecord(
            bet_id=bet_id or "FAILED",
            race_id=pred["race_id"],
            horse_name=pred["horse_name"],
            predicted_prob=prob,
            exchange_odds=odds,
            value_score=round(vs, 4),
            kelly_fraction=round(kelly, 4),
            stake=rec.stake_amount,
            actual_odds_matched=odds if bet_id else None,
            result=None,
            pnl=None,
            placed_at=datetime.now().isoformat(),
        )

        log_bet(record)
        records.append(record)

        bank_state.current_bank -= rec.stake_amount
        bank_state.daily_profit -= rec.stake_amount
        bank_state.bets_today += 1
        bank_state.total_bets += 1

        is_dry = bet_id.startswith("DRY_") if bet_id else True
        logger.info(
            f"{'DRY ' if is_dry else ''}BET: {pred['horse_name']} @ {odds} "
            f"(prob={prob:.3f}, vs={vs:.2f}, stake={rec.stake_amount})"
        )

        if bank_state.is_daily_limit_hit:
            logger.info("Daily loss limit hit -- stopping")
            break

    return records
