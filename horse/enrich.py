"""
Data enrichment engine -- populates derived statistics tables from raw results.

Computes:
  1. damsire_stats   (grandoffspring win rates from 2.28M results)
  2. ratings_history (official rating snapshots per horse per race)
  3. courses         (draw bias, pace bias, direction for UK/IRE courses)
  4. jockeys         (win rate, place rate from results)
  5. trainers        (win rate, place rate from results)

Run standalone:  python -m horse.enrich
Or import:       from horse.enrich import enrich_all
"""

import logging
import math
import time

from .db import get_connection

logger = logging.getLogger(__name__)

UK_COURSE_DIRECTIONS = {
    "Ascot": "RH", "Ayr": "LH", "Bangor-on-Dee": "LH", "Bath": "LH",
    "Beverley": "RH", "Brighton": "LH", "Carlisle": "RH", "Cartmel": "LH",
    "Catterick": "LH", "Chelmsford": "LH", "Cheltenham": "LH",
    "Chepstow": "LH", "Chester": "LH", "Doncaster": "LH",
    "Epsom": "LH", "Exeter": "RH", "Fakenham": "LH", "Ffos Las": "LH",
    "Fontwell": "LH/RH", "Goodwood": "RH", "Hamilton": "RH",
    "Haydock": "LH", "Hereford": "RH", "Hexham": "LH",
    "Huntingdon": "RH", "Kelso": "LH", "Kempton": "RH",
    "Leicester": "RH", "Lingfield": "LH", "Ludlow": "RH",
    "Market Rasen": "RH", "Musselburgh": "RH", "Newbury": "LH",
    "Newcastle": "LH", "Newmarket": "RH", "Newton Abbot": "LH",
    "Nottingham": "LH", "Perth": "RH", "Plumpton": "LH",
    "Pontefract": "LH", "Redcar": "LH", "Ripon": "RH",
    "Salisbury": "RH", "Sandown": "RH", "Sedgefield": "LH",
    "Southwell": "LH", "Stratford": "LH", "Taunton": "RH",
    "Thirsk": "LH", "Towcester": "RH", "Uttoxeter": "LH",
    "Warwick": "LH", "Wetherby": "LH", "Wincanton": "RH",
    "Windsor": "RH", "Wolverhampton": "LH", "Worcester": "LH",
    "Yarmouth": "LH", "York": "LH",
    "Curragh": "RH", "Leopardstown": "LH", "Fairyhouse": "RH",
    "Navan": "LH", "Punchestown": "RH", "Galway": "RH",
    "Limerick": "RH", "Cork": "RH", "Down Royal": "RH",
    "Dundalk": "LH", "Gowran Park": "RH", "Killarney": "LH",
    "Listowel": "LH", "Naas": "LH", "Roscommon": "RH",
    "Sligo": "RH", "Tipperary": "LH", "Tramore": "RH",
    "Wexford": "RH", "Ballinrobe": "RH", "Bellewstown": "LH",
    "Clonmel": "RH", "Downpatrick": "RH", "Kilbeggan": "RH",
    "Laytown": "RH", "Thurles": "RH",
}


def enrich_damsire_stats(con) -> int:
    """Compute damsire stats from results -- grandoffspring win rates."""
    logger.info("Enriching damsire_stats...")
    t0 = time.time()

    rows = con.execute("""
        SELECT
            res.api_damsire_id,
            COALESCE(res.damsire, 'Unknown') as damsire_name,
            COUNT(*) as grandoffspring_count,
            SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) as grandoffspring_wins,
            ROUND(SUM(CASE WHEN res.position = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as grandoffspring_win_rate
        FROM results res
        WHERE res.api_damsire_id IS NOT NULL
          AND res.api_damsire_id != ''
        GROUP BY res.api_damsire_id, COALESCE(res.damsire, 'Unknown')
        HAVING COUNT(*) >= 3
    """).fetchall()

    if not rows:
        logger.warning("No damsire data found in results")
        return 0

    con.execute("DELETE FROM damsire_stats")

    inserted = 0
    for i, row in enumerate(rows):
        api_id, name, count, wins, wr = row
        try:
            con.execute("""
                INSERT INTO damsire_stats (damsire_id, api_damsire_id, damsire_name,
                    grandoffspring_count, grandoffspring_wins, grandoffspring_win_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [i + 1, api_id, name, count, wins, wr])
            inserted += 1
        except Exception:
            pass

    logger.info(f"  damsire_stats: {inserted:,} damsires in {time.time()-t0:.1f}s")
    return inserted


def enrich_ratings_history(con) -> int:
    """Build ratings_history from results.official_rating + meeting dates."""
    logger.info("Enriching ratings_history...")
    t0 = time.time()

    rows = con.execute("""
        SELECT DISTINCT
            h.horse_id,
            res.horse_name,
            m.meeting_date as rating_date,
            res.official_rating as rating
        FROM results res
        JOIN races ra ON res.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        JOIN horses h ON res.api_horse_id = h.api_horse_id
        WHERE res.official_rating IS NOT NULL
          AND res.official_rating > 0
        ORDER BY res.horse_name, m.meeting_date
    """).fetchall()

    if not rows:
        logger.warning("No official_rating data in results")
        return 0

    con.execute("DELETE FROM ratings_history")
    try:
        con.execute("DROP SEQUENCE IF EXISTS seq_rating_id")
        con.execute("CREATE SEQUENCE seq_rating_id START 1")
    except Exception:
        pass

    inserted = 0
    batch = []
    for row in rows:
        horse_id, horse_name, rating_date, rating = row
        batch.append((horse_id, horse_name, str(rating_date), "official", int(rating), "results"))
        if len(batch) >= 5000:
            _insert_ratings_batch(con, batch)
            inserted += len(batch)
            batch = []

    if batch:
        _insert_ratings_batch(con, batch)
        inserted += len(batch)

    logger.info(f"  ratings_history: {inserted:,} entries in {time.time()-t0:.1f}s")
    return inserted


def _insert_ratings_batch(con, batch):
    for row in batch:
        try:
            con.execute("""
                INSERT INTO ratings_history (rating_id, horse_id, horse_name,
                    rating_date, rating_type, rating, source)
                VALUES (nextval('seq_rating_id'), ?, ?, ?, ?, ?, ?)
            """, list(row))
        except Exception:
            pass


def enrich_course_stats(con) -> int:
    """Compute draw bias and pace bias for UK/IRE courses from historical results."""
    logger.info("Enriching course stats (draw bias, pace bias, direction)...")
    t0 = time.time()
    updated = 0

    courses = con.execute("""
        SELECT DISTINCT course FROM courses
        WHERE country IN ('GB', 'IRE', 'UK')
    """).fetchall()
    course_names = [c[0] for c in courses]

    for course_name in course_names:
        draw_bias = _compute_draw_bias(con, course_name)
        pace_bias = _compute_pace_bias(con, course_name)
        direction = UK_COURSE_DIRECTIONS.get(course_name)

        con.execute("""
            UPDATE courses SET
                draw_bias_score = ?,
                pace_bias = ?,
                direction = COALESCE(?, direction)
            WHERE course = ?
        """, [draw_bias, pace_bias, direction, course_name])
        updated += 1

    logger.info(f"  courses: {updated:,} updated in {time.time()-t0:.1f}s")
    return updated


def _compute_draw_bias(con, course_name: str) -> float:
    """Compute draw bias score for a course.
    Positive = high draws advantaged, negative = low draws advantaged, near 0 = no bias.
    Uses correlation between draw position and finish position (inverted).
    """
    rows = con.execute("""
        SELECT res.draw, res.position
        FROM results res
        JOIN races ra ON res.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE m.course = ?
          AND res.draw IS NOT NULL AND res.draw > 0
          AND res.position IS NOT NULL AND res.position > 0
          AND ra.num_runners >= 8
          AND ra.distance_furlongs <= 10
    """, [course_name]).fetchall()

    if len(rows) < 100:
        return 0.0

    draws = [r[0] for r in rows]
    positions = [r[1] for r in rows]

    n = len(draws)
    mean_d = sum(draws) / n
    mean_p = sum(positions) / n

    cov = sum((d - mean_d) * (p - mean_p) for d, p in zip(draws, positions)) / n
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in draws) / n)
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in positions) / n)

    if std_d == 0 or std_p == 0:
        return 0.0

    correlation = cov / (std_d * std_p)
    return round(-correlation, 4)


def _compute_pace_bias(con, course_name: str) -> str:
    """Compute whether a course favours front-runners or closers.
    Returns 'front', 'hold_up', or 'neutral'.
    Uses win rate of horses drawn low (proxy for front-runners on flat).
    """
    rows = con.execute("""
        SELECT
            res.position,
            res.comment
        FROM results res
        JOIN races ra ON res.race_id = ra.race_id
        JOIN meetings m ON ra.meeting_id = m.meeting_id
        WHERE m.course = ?
          AND res.position IS NOT NULL AND res.position > 0
          AND res.comment IS NOT NULL
    """, [course_name]).fetchall()

    if len(rows) < 200:
        return "neutral"

    front_keywords = ["led", "made all", "prominent", "set pace", "made most"]
    hold_up_keywords = ["held up", "came from behind", "stayed on", "finished well"]

    front_wins = 0
    front_total = 0
    hold_up_wins = 0
    hold_up_total = 0

    for pos, comment in rows:
        if not comment:
            continue
        comment_lower = comment.lower()
        is_front = any(kw in comment_lower for kw in front_keywords)
        is_hold_up = any(kw in comment_lower for kw in hold_up_keywords)

        if is_front:
            front_total += 1
            if pos == 1:
                front_wins += 1
        if is_hold_up:
            hold_up_total += 1
            if pos == 1:
                hold_up_wins += 1

    front_wr = front_wins / front_total if front_total > 20 else 0
    hold_up_wr = hold_up_wins / hold_up_total if hold_up_total > 20 else 0

    if front_wr > hold_up_wr * 1.3:
        return "front"
    elif hold_up_wr > front_wr * 1.3:
        return "hold_up"
    return "neutral"


def enrich_jockey_stats(con) -> int:
    """Compute jockey win/place rates from results."""
    logger.info("Enriching jockey stats...")
    t0 = time.time()

    rows = con.execute("""
        SELECT
            j.jockey_id,
            j.api_id,
            COUNT(*) as total_rides,
            SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) as total_wins,
            ROUND(SUM(CASE WHEN res.position = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as win_rate,
            ROUND(SUM(CASE WHEN res.position <= 3 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as place_rate
        FROM jockeys j
        JOIN results res ON res.api_jockey_id = j.api_id
        WHERE res.position IS NOT NULL AND res.position > 0
        GROUP BY j.jockey_id, j.api_id
    """).fetchall()

    updated = 0
    for row in rows:
        jockey_id, api_id, total_rides, total_wins, win_rate, place_rate = row
        con.execute("""
            UPDATE jockeys SET
                total_rides = ?,
                total_wins = ?,
                win_rate = ?,
                place_rate = ?,
                updated_at = current_timestamp
            WHERE jockey_id = ?
        """, [total_rides, total_wins, win_rate, place_rate, jockey_id])
        updated += 1

    logger.info(f"  jockeys: {updated:,} updated in {time.time()-t0:.1f}s")
    return updated


def enrich_trainer_stats(con) -> int:
    """Compute trainer win/place rates from results."""
    logger.info("Enriching trainer stats...")
    t0 = time.time()

    rows = con.execute("""
        SELECT
            t.trainer_id,
            t.api_id,
            COUNT(*) as total_runners,
            SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) as total_wins,
            ROUND(SUM(CASE WHEN res.position = 1 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as win_rate,
            ROUND(SUM(CASE WHEN res.position <= 3 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as place_rate
        FROM trainers t
        JOIN results res ON res.api_trainer_id = t.api_id
        WHERE res.position IS NOT NULL AND res.position > 0
        GROUP BY t.trainer_id, t.api_id
    """).fetchall()

    updated = 0
    for row in rows:
        trainer_id, api_id, total_runners, total_wins, win_rate, place_rate = row
        con.execute("""
            UPDATE trainers SET
                total_runners = ?,
                total_wins = ?,
                win_rate = ?,
                place_rate = ?,
                updated_at = current_timestamp
            WHERE trainer_id = ?
        """, [total_runners, total_wins, win_rate, place_rate, trainer_id])
        updated += 1

    logger.info(f"  trainers: {updated:,} updated in {time.time()-t0:.1f}s")
    return updated


def enrich_all():
    """Run all enrichment steps."""
    logger.info("=" * 60)
    logger.info("DATA ENRICHMENT ENGINE")
    logger.info("=" * 60)
    t0 = time.time()

    con = get_connection()
    try:
        counts = {}
        counts["damsire_stats"] = enrich_damsire_stats(con)
        counts["ratings_history"] = enrich_ratings_history(con)
        counts["courses"] = enrich_course_stats(con)
        counts["jockeys"] = enrich_jockey_stats(con)
        counts["trainers"] = enrich_trainer_stats(con)

        logger.info("=" * 60)
        logger.info(f"ENRICHMENT COMPLETE in {time.time()-t0:.1f}s")
        for table, count in counts.items():
            logger.info(f"  {table}: {count:,} rows")
        logger.info("=" * 60)

        return counts
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    enrich_all()
