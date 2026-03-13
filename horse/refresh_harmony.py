"""
refresh_harmony.py

Reads morning bookmaker odds from the market_data table (horse.duckdb, read-only)
and patches today's predictions cache JSON with harmony scores for best-bet picks.

Run after MORNING_ODDS.bat has fetched fresh odds:
    python horse/refresh_harmony.py
    python horse/refresh_harmony.py --date 2026-03-14
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _harmony_pct(model_prob: float, implied_prob: float) -> float:
    """Agreement between model probability and market implied probability.
    100 = perfectly in sync.  Drops toward 0 as they diverge.
    """
    hi = max(model_prob, implied_prob)
    lo = min(model_prob, implied_prob)
    if hi <= 0:
        return 0.0
    return round(lo / hi * 100, 1)


def _value_flag(fair_odds: float, back_odds: float) -> str:
    if back_odds > fair_odds * 1.10:
        return "VALUE"
    if back_odds < fair_odds * 0.90:
        return "SHORT"
    return "FAIR"


def refresh_harmony(target_date: str) -> int:
    """Patch the predictions cache for *target_date* with harmony data.

    Returns the number of best-bet picks that were updated.
    """
    from horse.config import DATA_DIR, DB_PATH

    cache_dir = DATA_DIR / "predictions_cache"
    cache_path = cache_dir / f"{target_date}.json"

    if not cache_path.exists():
        logger.error(f"Cache file not found: {cache_path}")
        return 0

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    picks = cache.get("best_bets", {}).get("picks", [])
    if not picks:
        logger.info("No best-bet picks in cache — nothing to patch.")
        return 0

    # ------------------------------------------------------------------ #
    # Load odds from market_data (read-only connection)                   #
    # ------------------------------------------------------------------ #
    try:
        import duckdb
        con = duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as exc:
        logger.error(f"Cannot open DuckDB: {exc}")
        return 0

    try:
        rows = con.execute(
            """SELECT race_id, horse_name, back_odds, morning_price,
                      market_move_pct, steam_flag
               FROM market_data WHERE back_odds > 1.0"""
        ).fetchall()
    except Exception as exc:
        logger.error(f"market_data query failed: {exc}")
        con.close()
        return 0
    finally:
        con.close()

    # Build lookup: (race_id, horse_name_lower) -> odds row
    odds_lookup: dict[tuple, dict] = {
        (int(r[0]), str(r[1]).strip().lower()): {
            "back_odds": float(r[2]),
            "morning_price": float(r[3]) if r[3] else None,
            "market_move_pct": float(r[4]) if r[4] is not None else 0.0,
            "steam_flag": bool(r[5]) if r[5] is not None else False,
        }
        for r in rows
    }
    logger.info(f"Loaded {len(odds_lookup)} odds entries from market_data")

    # ------------------------------------------------------------------ #
    # Patch best-bet picks                                                #
    # ------------------------------------------------------------------ #
    updated = 0
    for pick in picks:
        race_id = int(pick.get("race_id", -1))
        horse_name = str(pick.get("horse_name", "")).strip().lower()
        key = (race_id, horse_name)

        odds_row = odds_lookup.get(key)
        if odds_row is None:
            logger.debug(f"No odds found for {pick.get('horse_name')} (race {race_id})")
            continue

        back_odds = odds_row["back_odds"]
        morning_price = odds_row["morning_price"]
        move_pct = odds_row["market_move_pct"]
        steam = odds_row["steam_flag"]

        win_prob = float(pick.get("win_prob", 0.0))
        fair_odds = round(1.0 / max(win_prob, 0.001), 2)
        implied_prob = round(1.0 / back_odds, 4)
        harmony = _harmony_pct(win_prob, implied_prob)
        flag = _value_flag(fair_odds, back_odds)

        pick["back_odds"] = round(back_odds, 2)
        pick["morning_price"] = round(morning_price, 2) if morning_price else None
        pick["market_move_pct"] = round(move_pct, 1)
        pick["steam_flag"] = steam
        pick["implied_prob"] = implied_prob
        pick["fair_odds"] = fair_odds
        pick["value_flag"] = flag
        pick["harmony_score"] = harmony

        move_str = f"{'STEAM ' if steam else ''}{move_pct:+.1f}%" if move_pct else "no move"
        morning_str = f"{morning_price:.2f}" if morning_price else "?"
        updated += 1
        logger.info(
            f"  {pick.get('horse_name')} @ {back_odds:.2f} | "
            f"morning {morning_str} | {move_str} | "
            f"{flag} | harmony {harmony:.0f}%"
        )

    if updated == 0:
        logger.warning(
            "No picks were matched to odds. "
            "Run MORNING_ODDS.bat first to populate market_data."
        )
    else:
        cache["best_bets"]["picks"] = picks
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        logger.info(f"Cache updated: {updated} pick(s) patched → {cache_path.name}")

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch predictions cache with harmony scores")
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Date to process (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    n = refresh_harmony(args.date)
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
