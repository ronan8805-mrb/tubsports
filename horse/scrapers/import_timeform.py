"""
import_timeform.py  –  Import Timeform Sectional Archive into sectionals table.

Usage:
    python -m horse.scrapers.import_timeform --file SectionalArchive10240325.xlsx
    python -m horse.scrapers.import_timeform --file SectionalArchive10240325.xlsx --dry-run

Maps the Timeform Excel columns to our sectionals table:
    FIN SPD %     → finishing_speed_pct
    SECT TIME     → late_split_avg
    INDIVID-SECT  → early_split_avg
    DIST          → total_furlongs (rounded)
    rank(MARG@SECT) → pos_at_2f  (position entering final section, 1=leader)
    MARG AT SECT  → gap_at_2f_secs (lengths behind leader entering final section)
    same as above → pos_at_halfway / gap_at_halfway_secs
    final_pos – pos_at_2f → position_gain  (positive = improved in closing stages)
    late_pace/early_pace  → pace_consistency  (<1 = finished faster, >1 = slowed)
    raw metrics   → splits_json
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import duckdb
import openpyxl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Course name mapping:  EXCEL (UPPERCASE) → DB meetings.course
# ---------------------------------------------------------------------------
_COURSE_MAP: dict[str, list[str]] = {
    # All-Weather
    "NEWCASTLE":        ["Newcastle (AW)"],
    "WOLVERHAMPTON":    ["Wolverhampton (AW)"],
    "KEMPTON PARK":     ["Kempton (AW)", "Kempton"],
    "LINGFIELD PARK":   ["Lingfield (AW)", "Lingfield"],
    "CHELMSFORD CITY":  ["Chelmsford (AW)"],
    "SOUTHWELL":        ["Southwell (AW)", "Southwell"],
    "DUNDALK":          ["Dundalk (AW)", "Dundalk (AW) (IRE)", "Dundalk (IRE)"],
    # Turf
    "ASCOT":                ["Ascot"],
    "AYR":                  ["Ayr"],
    "BATH":                 ["Bath"],
    "BEVERLEY":             ["Beverley"],
    "CARLISLE":             ["Carlisle"],
    "CATTERICK BRIDGE":     ["Catterick"],
    "CHEPSTOW":             ["Chepstow"],
    "CHESTER":              ["Chester"],
    "CURRAGH":              ["Curragh (IRE)", "Curragh"],
    "DONCASTER":            ["Doncaster"],
    "EPSOM DOWNS":          ["Epsom Downs", "Epsom"],
    "FAIRYHOUSE":           ["Fairyhouse (IRE)", "Fairyhouse"],
    "FFOS LAS":             ["Ffos Las"],
    "GOODWOOD":             ["Goodwood"],
    "HAMILTON PARK":        ["Hamilton"],
    "HAYDOCK PARK":         ["Haydock"],
    "LEICESTER":            ["Leicester"],
    "LEOPARDSTOWN":         ["Leopardstown (IRE)", "Leopardstown"],
    "MUSSELBURGH":          ["Musselburgh"],
    "NEWBURY":              ["Newbury"],
    "NEWMARKET (JULY)":     ["Newmarket (July)"],
    "NEWMARKET (ROWLEY)":   ["Newmarket"],
    "NOTTINGHAM":           ["Nottingham"],
    "PONTEFRACT":           ["Pontefract"],
    "REDCAR":               ["Redcar"],
    "RIPON":                ["Ripon"],
    "SALISBURY":            ["Salisbury"],
    "SANDOWN PARK":         ["Sandown"],
    "THIRSK":               ["Thirsk"],
    "WETHERBY":             ["Wetherby"],
    "WINDSOR":              ["Windsor"],
    "YARMOUTH":             ["Yarmouth", "Great Yarmouth"],
    "YORK":                 ["York"],
}

# Regex to strip breeding suffix like (GB), (IRE), (USA), (FR), (GER), (AUS) etc.
_SUFFIX_RE = re.compile(r"\s*\([A-Z]{2,5}\)\s*$")


def _strip_suffix(name: str) -> str:
    """Remove country/breeding suffix from DB horse name."""
    return _SUFFIX_RE.sub("", name).strip()


def _load_excel(path: Path) -> list[dict]:
    """Load All-Weather and Turf sheets, return combined list of row dicts."""
    log.info("Loading workbook %s …", path.name)
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    rows: list[dict] = []
    for sheet in ("All-Weather", "Turf"):
        ws = wb[sheet]
        raw = list(ws.iter_rows(values_only=True))
        header = raw[0]
        col = {h: i for i, h in enumerate(header)}
        for r in raw[1:]:
            if r[col["COURSE"]] is None:
                continue
            rows.append({
                "sheet":         sheet,
                "excel_course":  str(r[col["COURSE"]]).strip().upper(),
                "date":          str(r[col["DATE"]]),
                "race_seq":      int(r[col["RACE"]]),
                "horse_name":    str(r[col["HORSE NAME"]]).strip().upper(),
                "margin":        r[col["MARGIN"]],
                "cum_margin":    r[col["CUM MARGIN"]],
                "individ_times": r[col["INDIVID TIMES"]],
                "marg_at_sect":  r[col["MARG AT SECT"]],
                "sect_time":     r[col["SECT TIME"]],
                "fin_spd_pct":   r[col["FIN SPD %"]],
                "upgrade":       r[col["UPGRADE"]],
                "dist":          r[col["DIST"]],
                "sect_dist":     r[col["SECT DIST"]],
            })
    wb.close()
    log.info("Loaded %d rows from Excel (%d AW, %d Turf)",
             len(rows),
             sum(1 for r in rows if r["sheet"] == "All-Weather"),
             sum(1 for r in rows if r["sheet"] == "Turf"))
    return rows


def _build_race_index(con: duckdb.DuckDBPyConnection) -> dict[tuple, int]:
    """
    Returns dict keyed by (db_course_lower, date_str, seq_1based) → race_id.
    seq_1based = 1 for the first race of the day by off_dt, 2 for second, etc.
    """
    log.info("Building race sequence index from DB …")
    df = con.execute("""
        SELECT m.course, CAST(m.meeting_date AS VARCHAR) AS dt,
               r.race_id, r.off_dt
        FROM races r
        JOIN meetings m ON r.meeting_id = m.meeting_id
        WHERE m.meeting_date >= '2024-10-01'
        ORDER BY m.course, m.meeting_date, r.off_dt
    """).fetchdf()

    index: dict[tuple, int] = {}
    for (course, dt), grp in df.groupby(["course", "dt"], sort=False):
        for seq, (_, rrow) in enumerate(grp.iterrows(), start=1):
            index[(course.lower(), dt, seq)] = int(rrow["race_id"])
    log.info("Race index built: %d entries", len(index))
    return index


def _build_result_index(
    con: duckdb.DuckDBPyConnection,
) -> dict[tuple, dict]:
    """
    Returns dict keyed by (race_id, normalized_horse_name_lower) →
        {result_id, api_horse_id, position}
    Also keyed by (race_id, position) for fallback.
    """
    log.info("Building result index from DB …")
    df = con.execute("""
        SELECT result_id, race_id, horse_name, api_horse_id, position
        FROM results
        WHERE race_id IN (
            SELECT DISTINCT r.race_id FROM races r
            JOIN meetings m ON r.meeting_id = m.meeting_id
            WHERE m.meeting_date >= '2024-10-01'
        )
    """).fetchdf()

    index: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        norm = _strip_suffix(str(row["horse_name"])).upper().strip()
        key = (int(row["race_id"]), norm)
        pos_val = row["position"]
        rid_val = row["result_id"]
        ahid_val = row["api_horse_id"]
        index[key] = {
                "result_id":    int(rid_val)  if (rid_val  is not None and str(rid_val)  != "<NA>") else None,
                "api_horse_id": str(ahid_val) if (ahid_val is not None and str(ahid_val) != "<NA>") else None,
                "position":     int(pos_val)  if (pos_val  is not None and str(pos_val)  != "<NA>") else None,
            }
    log.info("Result index built: %d entries", len(index))
    return index


def _compute_derived(race_rows: list[dict]) -> list[dict]:
    """
    Given all rows for a single race, compute positional and pace metrics.
    Returns rows with added derived fields.
    """
    # Rank by MARG AT SECT ascending (0 = leader = rank 1)
    sorted_by_sect = sorted(race_rows, key=lambda r: (r["marg_at_sect"] or 999))
    for rank, r in enumerate(sorted_by_sect, start=1):
        r["_sect_rank"] = rank

    for r in race_rows:
        indiv = r["individ_times"]
        sect  = r["sect_time"]
        dist  = r["dist"]
        sdist = r["sect_dist"]

        # Early split = time for the portion before the final sectional
        early_time = (indiv - sect) if (indiv and sect) else None
        early_dist = (dist - sdist) if (dist and sdist) else None

        # Pace consistency: (late secs/furlong) / (early secs/furlong)
        # <1 = horse accelerated in final section (good closer)
        # >1 = horse decelerated (front runner)
        if early_time and early_dist and early_dist > 0 and sect and sdist and sdist > 0:
            late_pace  = sect / sdist
            early_pace = early_time / early_dist
            pace_cons  = late_pace / early_pace if early_pace > 0 else None
        else:
            pace_cons = None

        r["derived_early_split"]    = early_time
        r["derived_late_split"]     = sect
        r["derived_pace_cons"]      = pace_cons
        r["derived_fsp"]            = r["fin_spd_pct"]
        r["derived_total_furlongs"] = round(dist) if dist else None
        r["derived_pos_at_sect"]    = r["_sect_rank"]
        r["derived_gap_at_sect"]    = r["marg_at_sect"]

    return race_rows


def _next_sectional_id(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT COALESCE(MAX(sectional_id), 0) FROM sectionals").fetchone()
    return (row[0] or 0) + 1


def run(excel_path: Path, dry_run: bool = False) -> None:
    rows = _load_excel(excel_path)

    con = duckdb.connect("horse/data/horse.duckdb", read_only=dry_run)

    race_index   = _build_race_index(con)
    result_index = _build_result_index(con)

    # Resolve DB courses for each Excel course
    # Build lookup: excel_course → list of db_course (lowered)
    db_course_set = set(
        r[0].lower() for r in
        con.execute("SELECT DISTINCT course FROM meetings").fetchall()
    )

    def resolve_courses(excel_course: str) -> list[str]:
        candidates = _COURSE_MAP.get(excel_course, [])
        return [c.lower() for c in candidates if c.lower() in db_course_set]

    # Group rows by (excel_course, date, race_seq) → compute race-level derived metrics
    from collections import defaultdict
    race_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        race_groups[(r["excel_course"], r["date"], r["race_seq"])].append(r)

    stats = {"races": 0, "matched": 0, "inserted": 0, "skipped": 0,
             "no_race": 0, "no_result": 0}

    next_id = _next_sectional_id(con) if not dry_run else 999_000

    t0 = time.time()
    for (excel_course, date_str, race_seq), group in sorted(race_groups.items()):
        stats["races"] += 1

        # Find race_id
        db_courses = resolve_courses(excel_course)
        race_id = None
        for db_c in db_courses:
            race_id = race_index.get((db_c, date_str, race_seq))
            if race_id:
                break

        if not race_id:
            stats["no_race"] += 1
            log.debug("No race_id for %s %s race=%d", excel_course, date_str, race_seq)
            continue

        # Compute positional + pace derived fields for the whole race group
        group = _compute_derived(group)

        for r in group:
            stats["matched"] += 1
            horse_upper = r["horse_name"].upper().strip()
            # Try exact match first, then with suffix stripped from Excel name
            result_info = result_index.get((race_id, horse_upper))
            if not result_info:
                horse_stripped = _strip_suffix(horse_upper).strip()
                result_info = result_index.get((race_id, horse_stripped))
            if not result_info:
                stats["no_result"] += 1
                log.debug("No result match: race_id=%d horse=%s", race_id, horse_upper)
                continue

            final_pos = result_info["position"]
            sect_rank = r["derived_pos_at_sect"]
            position_gain = (sect_rank - final_pos) if (sect_rank and final_pos) else None

            splits_raw = {
                "source":         "timeform",
                "sheet":          r["sheet"],
                "margin":         r["margin"],
                "cum_margin":     r["cum_margin"],
                "individ_times":  r["individ_times"],
                "marg_at_sect":   r["marg_at_sect"],
                "sect_time":      r["sect_time"],
                "fin_spd_pct":    r["fin_spd_pct"],
                "upgrade":        r["upgrade"],
                "dist":           r["dist"],
                "sect_dist":      r["sect_dist"],
            }

            record = {
                "sectional_id":        next_id,
                "race_id":             race_id,
                "result_id":           result_info["result_id"],
                "horse_name":          r["horse_name"],
                "api_horse_id":        result_info["api_horse_id"],
                "meeting_date":        date_str,
                "course":              db_courses[0] if db_courses else excel_course,
                "finishing_speed_pct": r["derived_fsp"],
                "total_furlongs":      r["derived_total_furlongs"],
                "pos_at_2f":           r["derived_pos_at_sect"],
                "pos_at_halfway":      r["derived_pos_at_sect"],
                "gap_at_2f_secs":      r["derived_gap_at_sect"],
                "gap_at_halfway_secs": r["derived_gap_at_sect"],
                "early_split_avg":     r["derived_early_split"],
                "late_split_avg":      r["derived_late_split"],
                "pace_consistency":    r["derived_pace_cons"],
                "position_gain":       position_gain,
                "splits_json":         json.dumps(splits_raw),
            }

            if dry_run:
                if stats["inserted"] < 5:
                    log.info("[DRY-RUN] Would insert: race_id=%d horse=%s pos_at_sect=%s fsp=%.1f gain=%s",
                             race_id, horse_upper,
                             sect_rank, r["fin_spd_pct"] or 0,
                             position_gain)
                stats["inserted"] += 1
                next_id += 1
                continue

            try:
                con.execute("""
                    INSERT OR IGNORE INTO sectionals (
                        sectional_id, race_id, result_id, horse_name, api_horse_id,
                        meeting_date, course, finishing_speed_pct, total_furlongs,
                        pos_at_2f, pos_at_halfway, gap_at_2f_secs, gap_at_halfway_secs,
                        early_split_avg, late_split_avg, pace_consistency,
                        position_gain, splits_json
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?
                    )
                """, [
                    record["sectional_id"], record["race_id"], record["result_id"],
                    record["horse_name"],   record["api_horse_id"],
                    record["meeting_date"], record["course"],
                    record["finishing_speed_pct"], record["total_furlongs"],
                    record["pos_at_2f"],    record["pos_at_halfway"],
                    record["gap_at_2f_secs"], record["gap_at_halfway_secs"],
                    record["early_split_avg"], record["late_split_avg"],
                    record["pace_consistency"], record["position_gain"],
                    record["splits_json"],
                ])
                stats["inserted"] += 1
                next_id += 1
            except Exception as exc:
                log.warning("Insert failed race_id=%d horse=%s: %s", race_id, horse_upper, exc)
                stats["skipped"] += 1

    elapsed = time.time() - t0
    log.info("─" * 60)
    log.info("Done in %.1fs", elapsed)
    log.info("  Excel race groups:    %d", stats["races"])
    log.info("  Races not in DB:      %d", stats["no_race"])
    log.info("  Horse rows matched:   %d", stats["matched"])
    log.info("  Results not matched:  %d", stats["no_result"])
    log.info("  Rows inserted:        %d", stats["inserted"])
    log.info("  Rows skipped (dup):   %d", stats["skipped"])

    if not dry_run:
        log.info("Committing …")
        con.commit()
        log.info("✓ Import complete – %d rows in sectionals table", stats["inserted"])

    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import Timeform Sectional Excel into sectionals table")
    ap.add_argument("--file", default="SectionalArchive10240325.xlsx",
                    help="Path to Excel file (default: SectionalArchive10240325.xlsx)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and match without writing to DB")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        # Try relative to project root
        project = Path(__file__).parents[2]
        path = project / args.file
    if not path.exists():
        log.error("Excel file not found: %s", args.file)
        sys.exit(1)

    run(path, dry_run=args.dry_run)
