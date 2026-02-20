"""Scrape upcoming racecards (today + tomorrow) from The Racing API."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from horse.scrapers.racing_api import RacingAPIClient, fetch_upcoming_racecards
from horse.db import get_connection

client = RacingAPIClient()
fetch_upcoming_racecards(client)

con = get_connection(read_only=True)
from datetime import date, timedelta
today = date.today()
tomorrow = today + timedelta(days=1)

for d in [today, tomorrow]:
    ds = d.strftime("%Y-%m-%d")
    r = con.execute("""
        SELECT m.course, COUNT(ra.race_id)
        FROM meetings m JOIN races ra ON ra.meeting_id = m.meeting_id
        WHERE m.meeting_date = ?
        GROUP BY m.course ORDER BY m.course
    """, [ds]).fetchall()
    print(f"\n{ds}:")
    if r:
        for row in r:
            print(f"  {row[0]:<30s} {row[1]} races")
    else:
        print("  No races found")

con.close()
print("\nDone. Restart API to serve new racecards.")
