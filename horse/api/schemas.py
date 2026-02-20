"""
Pydantic response models for the Horse Racing 13D Dashboard API.
"""

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    features: Dict[str, float] = Field(default_factory=dict)
    tooltip: str = ""
    color: str = "#6b7280"


class DriverItem(BaseModel):
    feature: str
    value: str
    impact: str
    description: str


class RunnerPrediction(BaseModel):
    horse_name: str
    draw: Optional[int] = None
    jockey: Optional[str] = None
    trainer: Optional[str] = None
    age: Optional[int] = None
    weight_lbs: Optional[int] = None
    official_rating: Optional[int] = None
    win_prob: float = 0.0
    place_prob: float = 0.0
    win_rank: int = 0
    place_rank: int = 0
    fair_odds: float = Field(0.0, description="1 / win_prob")
    back_odds: Optional[float] = Field(None, description="Live bookmaker decimal odds")
    value_flag: Optional[str] = Field(None, description="VALUE / FAIR / SHORT")
    odds_updated_at: Optional[str] = Field(None, description="When odds were last fetched")
    silk_url: Optional[str] = Field(None, description="Jockey silks image URL")
    dimensions: List[DimensionScore] = []
    positive_drivers: List[DriverItem] = []
    negative_drivers: List[DriverItem] = []


class RaceInfo(BaseModel):
    race_id: int
    meeting_date: str
    course: str
    race_number: Optional[int] = None
    race_name: Optional[str] = None
    race_time: Optional[str] = None
    distance_furlongs: Optional[float] = None
    race_type: Optional[str] = None
    race_class: Optional[str] = None
    going: Optional[str] = None
    surface: Optional[str] = None
    num_runners: int = 0
    region_code: Optional[str] = None


class RaceCard(BaseModel):
    races: List[RaceInfo]
    meeting_date: str
    course: Optional[str] = None
    total_races: int = 0
    timestamp: str


class ThirteenDRace(BaseModel):
    race_info: RaceInfo
    runners: List[RunnerPrediction]
    model_version: str = "v1"
    timestamp: str
    predictions_status: str = "ready"


class DateInfo(BaseModel):
    date: str
    meetings: int = 0
    races: int = 0


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    meetings: int = 0
    races: int = 0
    results: int = 0
    horse_form: int = 0
    engine: str = "13D"
