import json
from horse.db import get_connection

con = get_connection(read_only=True)

# Check runner time formats
rows = con.execute("""
    SELECT r.raw_json, r.horse_name, r.position, ra.winning_time_detail,
           ra.distance_furlongs, ra.going_description, ra.surface, m.course
    FROM results r
    JOIN races ra ON r.race_id = ra.race_id
    JOIN meetings m ON ra.meeting_id = m.meeting_id
    WHERE r.raw_json IS NOT NULL AND r.position IS NOT NULL AND r.position <= 3
    LIMIT 20
""").fetchall()

for row in rows:
    d = json.loads(row[0])
    t = d.get("time", "MISSING")
    print(f"pos={row[2]}, time='{t}', winning_time='{row[3]}', dist={row[4]}f, going={row[5]}, surface={row[6]}, course={row[7]}, horse={row[1]}")

# Count how many have non-null time
r2 = con.execute("""
    SELECT COUNT(*) FROM results 
    WHERE raw_json IS NOT NULL 
    AND json_extract_string(raw_json, '$.time') IS NOT NULL
    AND json_extract_string(raw_json, '$.time') != ''
""").fetchone()
r3 = con.execute("SELECT COUNT(*) FROM results WHERE raw_json IS NOT NULL").fetchone()
print(f"\nResults with time in raw_json: {r2[0]:,} / {r3[0]:,}")

con.close()
