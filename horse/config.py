"""
Configuration for the UK/IRE Horse Racing Prediction System.
Completely independent from the greyhound system.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "horse.duckdb"
MODELS_DIR = DATA_DIR / "models"

# Load .env from horse/ directory
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
API_PORT = 8002
FRONTEND_PORT = 5174

# ---------------------------------------------------------------------------
# The Racing API (primary data source)
# ---------------------------------------------------------------------------
RACING_API_BASE = "https://api.theracingapi.com/v1"
RACING_API_USERNAME = os.getenv("RACING_API_USERNAME", "")
RACING_API_PASSWORD = os.getenv("RACING_API_PASSWORD", "")
RACING_API_RATE_LIMIT = 4  # requests per second (API allows 5, we stay safe)

# ---------------------------------------------------------------------------
# Scraping (legacy scrapers, kept as fallback)
# ---------------------------------------------------------------------------
SCRAPE_DELAY = 2.0  # seconds between requests (polite)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

BACKFILL_START_YEAR = 1988

# ---------------------------------------------------------------------------
# UK + IRE Courses with GPS coordinates (for weather API)
# ---------------------------------------------------------------------------
COURSES = {
    # Ireland
    "leopardstown": {"country": "IE", "lat": 53.2873, "lon": -6.2013, "direction": "left", "surface": "turf", "type": "galloping"},
    "curragh": {"country": "IE", "lat": 53.1591, "lon": -6.7536, "direction": "right", "surface": "turf", "type": "galloping"},
    "fairyhouse": {"country": "IE", "lat": 53.5334, "lon": -6.5436, "direction": "right", "surface": "turf", "type": "galloping"},
    "punchestown": {"country": "IE", "lat": 53.1756, "lon": -6.6700, "direction": "right", "surface": "turf", "type": "galloping"},
    "dundalk": {"country": "IE", "lat": 54.0053, "lon": -6.4016, "direction": "left", "surface": "aw", "type": "sharp"},
    "galway": {"country": "IE", "lat": 53.2879, "lon": -8.9917, "direction": "right", "surface": "turf", "type": "undulating"},
    "cork": {"country": "IE", "lat": 51.9148, "lon": -8.4900, "direction": "right", "surface": "turf", "type": "undulating"},
    "limerick": {"country": "IE", "lat": 52.5825, "lon": -8.6283, "direction": "right", "surface": "turf", "type": "sharp"},
    "navan": {"country": "IE", "lat": 53.6457, "lon": -6.6921, "direction": "left", "surface": "turf", "type": "galloping"},
    "naas": {"country": "IE", "lat": 53.2039, "lon": -6.6535, "direction": "left", "surface": "turf", "type": "galloping"},
    "thurles": {"country": "IE", "lat": 52.6858, "lon": -7.8034, "direction": "right", "surface": "turf", "type": "sharp"},
    "wexford": {"country": "IE", "lat": 52.3432, "lon": -6.4560, "direction": "right", "surface": "turf", "type": "sharp"},
    "downpatrick": {"country": "IE", "lat": 54.3238, "lon": -5.7213, "direction": "right", "surface": "turf", "type": "sharp"},
    "killarney": {"country": "IE", "lat": 52.0578, "lon": -9.5127, "direction": "left", "surface": "turf", "type": "sharp"},
    "listowel": {"country": "IE", "lat": 52.4457, "lon": -9.4847, "direction": "left", "surface": "turf", "type": "sharp"},
    "tramore": {"country": "IE", "lat": 52.1638, "lon": -7.1479, "direction": "right", "surface": "turf", "type": "sharp"},
    "tipperary": {"country": "IE", "lat": 52.4735, "lon": -8.1602, "direction": "left", "surface": "turf", "type": "sharp"},
    "gowran_park": {"country": "IE", "lat": 52.6133, "lon": -7.0676, "direction": "right", "surface": "turf", "type": "galloping"},
    "roscommon": {"country": "IE", "lat": 53.6271, "lon": -8.1909, "direction": "right", "surface": "turf", "type": "sharp"},
    "ballinrobe": {"country": "IE", "lat": 53.6370, "lon": -9.2303, "direction": "right", "surface": "turf", "type": "sharp"},
    "clonmel": {"country": "IE", "lat": 52.3497, "lon": -7.7058, "direction": "right", "surface": "turf", "type": "sharp"},
    "sligo": {"country": "IE", "lat": 54.2682, "lon": -8.4697, "direction": "right", "surface": "turf", "type": "sharp"},
    "kilbeggan": {"country": "IE", "lat": 53.3663, "lon": -7.4981, "direction": "right", "surface": "turf", "type": "sharp"},
    "laytown": {"country": "IE", "lat": 53.6807, "lon": -6.2399, "direction": "left", "surface": "turf", "type": "sharp"},
    "bellewstown": {"country": "IE", "lat": 53.6874, "lon": -6.4232, "direction": "left", "surface": "turf", "type": "sharp"},

    # UK — Major
    "ascot": {"country": "GB", "lat": 51.4107, "lon": -0.6730, "direction": "right", "surface": "turf", "type": "galloping"},
    "cheltenham": {"country": "GB", "lat": 51.9199, "lon": -2.0743, "direction": "left", "surface": "turf", "type": "undulating"},
    "aintree": {"country": "GB", "lat": 53.4767, "lon": -2.9566, "direction": "left", "surface": "turf", "type": "galloping"},
    "epsom": {"country": "GB", "lat": 51.3313, "lon": -0.2594, "direction": "left", "surface": "turf", "type": "undulating"},
    "newmarket": {"country": "GB", "lat": 52.2459, "lon": 0.3804, "direction": "right", "surface": "turf", "type": "galloping"},
    "york": {"country": "GB", "lat": 53.9477, "lon": -1.0866, "direction": "left", "surface": "turf", "type": "galloping"},
    "goodwood": {"country": "GB", "lat": 50.8860, "lon": -0.7503, "direction": "right", "surface": "turf", "type": "undulating"},
    "doncaster": {"country": "GB", "lat": 53.5208, "lon": -1.1384, "direction": "left", "surface": "turf", "type": "galloping"},
    "sandown": {"country": "GB", "lat": 51.3737, "lon": -0.3588, "direction": "right", "surface": "turf", "type": "galloping"},
    "newbury": {"country": "GB", "lat": 51.3997, "lon": -1.3118, "direction": "left", "surface": "turf", "type": "galloping"},
    "kempton": {"country": "GB", "lat": 51.4183, "lon": -0.4105, "direction": "right", "surface": "aw", "type": "sharp"},
    "lingfield": {"country": "GB", "lat": 51.1702, "lon": -0.0229, "direction": "left", "surface": "aw", "type": "sharp"},
    "wolverhampton": {"country": "GB", "lat": 52.5908, "lon": -2.1367, "direction": "left", "surface": "aw", "type": "sharp"},
    "southwell": {"country": "GB", "lat": 53.0768, "lon": -0.9021, "direction": "left", "surface": "aw", "type": "sharp"},
    "newcastle": {"country": "GB", "lat": 54.9792, "lon": -1.6345, "direction": "left", "surface": "aw", "type": "galloping"},
    "chelmsford": {"country": "GB", "lat": 51.7168, "lon": 0.5018, "direction": "left", "surface": "aw", "type": "sharp"},
    "haydock": {"country": "GB", "lat": 53.4673, "lon": -2.6265, "direction": "left", "surface": "turf", "type": "galloping"},
    "wetherby": {"country": "GB", "lat": 53.9214, "lon": -1.3860, "direction": "left", "surface": "turf", "type": "galloping"},
    "chester": {"country": "GB", "lat": 53.1828, "lon": -2.8952, "direction": "left", "surface": "turf", "type": "sharp"},
    "windsor": {"country": "GB", "lat": 51.4849, "lon": -0.5986, "direction": "right", "surface": "turf", "type": "sharp"},
    "leicester": {"country": "GB", "lat": 52.6244, "lon": -1.0836, "direction": "right", "surface": "turf", "type": "galloping"},
    "catterick": {"country": "GB", "lat": 54.3663, "lon": -1.6497, "direction": "left", "surface": "turf", "type": "sharp"},
    "huntingdon": {"country": "GB", "lat": 52.3366, "lon": -0.1732, "direction": "right", "surface": "turf", "type": "galloping"},
    "exeter": {"country": "GB", "lat": 50.7396, "lon": -3.5114, "direction": "right", "surface": "turf", "type": "undulating"},
    "plumpton": {"country": "GB", "lat": 50.9297, "lon": -0.0679, "direction": "left", "surface": "turf", "type": "sharp"},
    "musselburgh": {"country": "GB", "lat": 55.9427, "lon": -3.0671, "direction": "right", "surface": "turf", "type": "sharp"},
    "warwick": {"country": "GB", "lat": 52.2842, "lon": -1.5903, "direction": "left", "surface": "turf", "type": "sharp"},
    "market_rasen": {"country": "GB", "lat": 53.3872, "lon": -0.3344, "direction": "right", "surface": "turf", "type": "sharp"},
    "fontwell": {"country": "GB", "lat": 50.8647, "lon": -0.6697, "direction": "left", "surface": "turf", "type": "sharp"},
    "wincanton": {"country": "GB", "lat": 51.0628, "lon": -2.3876, "direction": "right", "surface": "turf", "type": "galloping"},
    "ayr": {"country": "GB", "lat": 55.4562, "lon": -4.6135, "direction": "left", "surface": "turf", "type": "galloping"},
    "kelso": {"country": "GB", "lat": 55.5956, "lon": -2.4310, "direction": "left", "surface": "turf", "type": "sharp"},
}

# ---------------------------------------------------------------------------
# Going encoding (higher = softer ground)
# ---------------------------------------------------------------------------
GOING_ENCODE = {
    "firm": 1.0, "good to firm": 2.0, "good": 3.0,
    "good to yielding": 3.5, "yielding": 4.0, "yielding to soft": 4.5,
    "good to soft": 4.0, "soft": 5.0, "soft to heavy": 5.5,
    "heavy": 6.0, "standard": 3.0, "standard to slow": 4.0,
    "slow": 5.0,
}

# ---------------------------------------------------------------------------
# Race type mapping
# ---------------------------------------------------------------------------
RACE_TYPES = {
    "flat": "flat",
    "hurdle": "nh",
    "chase": "nh",
    "nhf": "nh",
    "bumper": "nh",
    "national_hunt_flat": "nh",
}

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MAX_DAYS_LOOKBACK = 365
FORM_WINDOW = 6
SHORT_FORM_WINDOW = 3
MONTE_CARLO_SIMULATIONS = 10_000
MAX_CACHED_DATES = 10
