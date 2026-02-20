"""
Master backfill orchestrator for the Horse Racing system.

9 phases, each independently resumable:
  1. Courses       - sync all courses from API
  2. Results       - last 12 months, all countries, month-by-month
  3. Validation    - check backfill_log for warnings
  4. Horse Profiles + Form History - every horse profile + full career history
  5. Jockey Data   - every jockey
  6. Trainer Data  - every trainer
  7. Sire/Dam/Damsire Data - breeding stats
  8. Cleanup       - dedup check, stats dashboard
  9. Racecards     - upcoming race entries

API constraint: /results endpoint limited to 12 months in the past.
Per-horse /horses/{id}/results gives full career history (years of data).

Usage:
    python -m horse.scrapers.backfill               # run all phases
    python -m horse.scrapers.backfill --phase 2      # run only phase 2
    python -m horse.scrapers.backfill --phase 4      # run only horse profiles
"""

import argparse
import logging
import sys
import time

from ..db import get_connection, init_database, get_db_stats
from .racing_api import (
    RacingAPIClient,
    sync_courses,
    backfill_all_results,
    fetch_all_horse_form,
    fetch_all_jockey_data,
    fetch_all_trainer_data,
    fetch_all_sire_data,
    fetch_all_dam_data,
    fetch_all_damsire_data,
    fetch_upcoming_racecards,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("horse.backfill")


def print_banner():
    print()
    print("=" * 64)
    print("  HORSE RACING - FULL DATA BACKFILL")
    print("  The Racing API (PRO) -> horse.duckdb")
    print("  Last 12 months results + full career form per horse")
    print("  All countries | All surfaces | All race types")
    print("=" * 64)
    print()


def print_stats():
    """Print current DB row counts."""
    con = get_connection()
    try:
        stats = get_db_stats(con)
        print()
        print("-" * 40)
        print("  HORSE DB STATUS")
        print("-" * 40)
        for table, count in stats.items():
            print(f"  {table:20s} : {count:>10,}")
        print("-" * 40)
        total = sum(stats.values())
        print(f"  {'TOTAL':20s} : {total:>10,}")
        print()
    finally:
        con.close()


def print_backfill_log():
    """Print the backfill log summary."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT chunk_key, phase, status, api_total, db_race_count,
                   db_result_count, mean_runners_per_race, error_message
            FROM backfill_log
            ORDER BY chunk_key, phase
        """).fetchall()

        if not rows:
            print("  No backfill log entries yet.")
            return

        print()
        print("-" * 85)
        print(f"  {'Chunk':<10} {'Phase':<10} {'Status':<15} {'API Total':>10} "
              f"{'Races':>8} {'Runners':>10} {'Avg R/R':>7}")
        print("-" * 85)
        for r in rows:
            chunk, phase, status, api_total, races, runners, avg, err = r
            api_total = api_total or 0
            races = races or 0
            runners = runners or 0
            avg = avg or 0
            flag = " !!!" if status in ("warning", "drift_detected", "error") else ""
            print(f"  {str(chunk):<10} {phase:<10} {status:<15} {api_total:>10,} "
                  f"{races:>8,} {runners:>10,} {avg:>7.1f}{flag}")
            if err:
                print(f"         -> {err}")
        print("-" * 85)
        print()
    finally:
        con.close()


def run_phase(phase_num: int, client: RacingAPIClient):
    """Run a specific phase."""
    t0 = time.monotonic()

    if phase_num == 1:
        logger.info("PHASE 1: Syncing courses...")
        sync_courses(client)

    elif phase_num == 2:
        logger.info("PHASE 2: Backfilling results (last 12 months)...")
        backfill_all_results(client)

    elif phase_num == 3:
        logger.info("PHASE 3: Validation report...")
        print_backfill_log()

    elif phase_num == 4:
        logger.info("PHASE 4: Fetching ALL horse career form history...")
        logger.info("  (Horse profiles skipped -- API returns 404 for most)")
        fetch_all_horse_form(client)

    elif phase_num == 5:
        logger.info("PHASE 5: Fetching ALL jockey + trainer data...")
        fetch_all_jockey_data(client)
        fetch_all_trainer_data(client)

    elif phase_num == 6:
        logger.info("PHASE 6: Fetching ALL sire/dam/damsire data...")
        fetch_all_sire_data(client)
        fetch_all_dam_data(client)
        fetch_all_damsire_data(client)

    elif phase_num == 7:
        logger.info("PHASE 7: Cleanup and stats...")
        from ..db import cleanup_duplicate_draws
        con = get_connection()
        try:
            removed = cleanup_duplicate_draws(con)
            if removed:
                logger.info(f"Cleaned up {removed} duplicate draw entries")
        finally:
            con.close()

    elif phase_num == 8:
        logger.info("PHASE 8: Fetching upcoming racecards...")
        fetch_upcoming_racecards(client)

    else:
        logger.error(f"Unknown phase: {phase_num}")
        return

    elapsed = time.monotonic() - t0
    logger.info(f"Phase {phase_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Horse Racing Data Backfill")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run a specific phase (1-9)")
    args = parser.parse_args()

    print_banner()

    init_database()
    logger.info("Database initialized.")

    try:
        client = RacingAPIClient()
        logger.info("Racing API client ready.")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.phase:
        run_phase(args.phase, client)
    else:
        for phase in range(1, 9):
            run_phase(phase, client)

    print_stats()
    logger.info("BACKFILL COMPLETE.")


if __name__ == "__main__":
    main()
