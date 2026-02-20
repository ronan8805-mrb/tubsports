"""
Backfill speed figures for all historical results.
Step 1: Extract finish_time from results.raw_json
Step 2: Compute speed_figure with hierarchical standard times
"""
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_speed")

from horse.db import get_connection
from horse.speed import backfill_finish_times, compute_all_speed_figures

logger.info("=" * 60)
logger.info("  SPEED FIGURE BACKFILL")
logger.info("=" * 60)

con = get_connection()
try:
    logger.info("Step 1: Extracting finish_time from raw_json...")
    t0 = time.time()
    n_times = backfill_finish_times(con)
    logger.info(f"  Done: {n_times:,} finish times extracted ({time.time()-t0:.1f}s)")

    logger.info("Step 2: Computing speed figures...")
    t0 = time.time()
    n_speed = compute_all_speed_figures(con)
    logger.info(f"  Done: {n_speed:,} speed figures computed ({time.time()-t0:.1f}s)")

    # Verify
    r1 = con.execute("SELECT COUNT(*) FROM results WHERE finish_time IS NOT NULL AND finish_time > 0").fetchone()
    r2 = con.execute("SELECT COUNT(*) FROM results WHERE speed_figure IS NOT NULL AND speed_figure > 0").fetchone()
    r3 = con.execute("SELECT COUNT(*) FROM horse_form WHERE speed_figure IS NOT NULL AND speed_figure > 0").fetchone()
    r4 = con.execute("SELECT COUNT(*) FROM results").fetchone()
    r5 = con.execute("SELECT COUNT(*) FROM horse_form").fetchone()

    logger.info("=" * 60)
    logger.info("  VERIFICATION")
    logger.info(f"  Results: {r1[0]:,}/{r4[0]:,} have finish_time")
    logger.info(f"  Results: {r2[0]:,}/{r4[0]:,} have speed_figure")
    logger.info(f"  Horse form: {r3[0]:,}/{r5[0]:,} have speed_figure")
    logger.info("=" * 60)

    # Sample speed figures
    samples = con.execute("""
        SELECT r.speed_figure, r.horse_name, r.position,
               ra.distance_furlongs, m.course
        FROM results r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE r.speed_figure IS NOT NULL
        ORDER BY r.speed_figure DESC
        LIMIT 10
    """).fetchall()
    logger.info("  Top 10 speed figures:")
    for s in samples:
        logger.info(f"    {s[0]:.1f} - {s[1]} (pos {s[2]}, {s[3]}f @ {s[4]})")

finally:
    con.close()

logger.info("DONE. Ready for retrain.")
