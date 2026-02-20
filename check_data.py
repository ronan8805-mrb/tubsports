import json
from horse.db import get_connection

con = get_connection(read_only=True)

r1 = con.execute('SELECT COUNT(*) FROM races WHERE winning_time_detail IS NOT NULL').fetchone()
r2 = con.execute('SELECT COUNT(*) FROM races').fetchone()
print(f'Races with winning_time_detail: {r1[0]:,} / {r2[0]:,}')

r3 = con.execute("SELECT winning_time_detail FROM races WHERE winning_time_detail IS NOT NULL LIMIT 5").fetchall()
print(f'Sample winning_time_detail: {[r[0] for r in r3]}')

r4 = con.execute("SELECT raw_json FROM results WHERE raw_json IS NOT NULL LIMIT 1").fetchone()
if r4:
    d = json.loads(r4[0])
    time_keys = [k for k in d.keys() if 'time' in k.lower()]
    print(f'Time-related keys in runner raw_json: {time_keys}')
    print(f'All keys in runner raw_json: {list(d.keys())}')

# Check race raw_json for time data
r5 = con.execute("SELECT raw_json FROM races WHERE raw_json IS NOT NULL LIMIT 1").fetchone()
if r5:
    d2 = json.loads(r5[0])
    time_keys2 = [k for k in d2.keys() if 'time' in k.lower() or 'winning' in k.lower()]
    print(f'Time-related keys in race raw_json: {time_keys2}')
    for k in time_keys2:
        print(f'  {k}: {d2[k]}')

# How many horses do we have with api_horse_id?
r6 = con.execute("SELECT COUNT(DISTINCT api_horse_id) FROM horses WHERE api_horse_id IS NOT NULL AND api_horse_id != ''").fetchone()
print(f'Horses with api_horse_id: {r6[0]:,}')

# How many unique horse_ids in horse_form?
r7 = con.execute("SELECT COUNT(DISTINCT api_horse_id) FROM horse_form WHERE api_horse_id IS NOT NULL").fetchone()
print(f'Unique horses in horse_form: {r7[0]:,}')

con.close()
