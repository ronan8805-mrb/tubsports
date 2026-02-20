"""
Multi-Dimensional Outcome Engine for Horse Racing.

Maps ~76 ML features into 13 human-readable dimension scores (0-100).
Each dimension aggregates related features into a single normalized score.

Dimensions:
  1.  Schedule   -- freshness / race spacing
  2.  Age        -- maturity stage
  3.  Gender     -- mare vs colt/gelding advantage
  4.  Weight     -- carried weight burden
  5.  Rating     -- official rating (raw ability)
  6.  Form       -- recent top-3 strike rate, consistency
  7.  Jockey     -- jockey quality + course record
  8.  Load       -- weight-to-ability ratio (penalty proxy)
  9.  Race Type  -- flat/hurdle/chase fit
  10. History    -- average finishing position trend
  11. Ground     -- going + surface match
  12. Field      -- competitive field size + dynamics
  13. Trainer    -- trainer strike rate + course record
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension definitions: feature -> weight within dimension
# Positive weight = higher is better; negative weight = lower is better
# ---------------------------------------------------------------------------

DIMENSION_MAP = {
    "Schedule": {
        "days_since_last_race": -1.5,
        "races_last_30d": 1.0,
        "races_last_90d": 0.5,
        "career_starts": 0.3,
    },
    "Age": {
        "age": -0.5,
    },
    "Gender": {
        "sex_encoded": 1.0,
    },
    "Weight": {
        "weight_lbs": -1.5,
        "weight_diff_from_median": -2.0,
        "jockey_claim_lbs": 1.5,
    },
    "Rating": {
        "official_rating": 3.0,
        "or_vs_field_avg": 2.5,
        "or_rank_in_field": -2.0,
        "best_or_6": 2.0,
        "avg_or_6": 1.5,
    },
    "Form": {
        "form_place_rate": 3.0,
        "form_win_rate": 3.0,
        "form_avg_pos_3": -2.5,
        "form_momentum": -2.0,
        "form_consistency": -1.0,
        "form_best_pos": -1.5,
        "avg_beaten_lengths": -1.5,
        "or_trend": 1.5,
    },
    "Jockey": {
        "jockey_win_rate_365d": 3.0,
        "jockey_place_rate_365d": 2.0,
        "jockey_course_win_rate": 2.5,
        "jockey_runs_365d": 0.5,
        "jt_combo_win_rate": 2.0,
    },
    "Load": {
        "weight_lbs": -1.5,
        "official_rating": 2.0,
        "or_vs_class": 2.5,
    },
    "Race Type": {
        "race_type_encoded": 1.0,
        "type_match_win_rate": 3.0,
        "class_match_win_rate": 2.0,
        "race_class_numeric": -1.5,
        "handicap_flag": 0.5,
    },
    "History": {
        "form_avg_pos_6": -3.0,
        "form_avg_pos_3": -2.5,
        "form_worst_pos": -1.0,
        "dist_match_win_rate": 2.5,
        "dist_match_place_rate": 2.0,
        "course_match_win_rate": 2.5,
    },
    "Ground": {
        "going_numeric": 1.0,
        "going_match_win_rate": 3.0,
        "surface_encoded": 0.5,
        "weather_precipitation": -1.5,
        "weather_wind_speed": -1.0,
        "weather_temp": 0.3,
        "precip_x_going": -2.0,
    },
    "Field": {
        "num_runners": -1.0,
        "draw": -0.5,
        "draw_pct_of_field": -0.5,
        "draw_hist_win_rate": 2.0,
    },
    "Trainer": {
        "trainer_win_rate_365d": 3.0,
        "trainer_place_rate_365d": 2.0,
        "trainer_course_win_rate": 2.5,
        "trainer_runs_365d": 0.5,
    },
    "Speed": {
        "best_speed_last_6": 3.0,
        "avg_speed_last_3": 2.5,
        "speed_trend": 2.0,
        "speed_vs_field_avg": 3.0,
        "speed_rank_in_field": -2.5,
    },
}

DIMENSION_NAMES = list(DIMENSION_MAP.keys())

DIM_COLORS = {
    "Schedule": "#8b5cf6",
    "Age": "#06b6d4",
    "Gender": "#ec4899",
    "Weight": "#f59e0b",
    "Rating": "#3b82f6",
    "Form": "#10b981",
    "Jockey": "#7c3aed",
    "Load": "#f97316",
    "Race Type": "#14b8a6",
    "History": "#f43f5e",
    "Ground": "#84cc16",
    "Field": "#0ea5e9",
    "Trainer": "#6366f1",
    "Speed": "#ef4444",
}


def compute_thirteen_d_scores(
    features_df: pd.DataFrame,
    race_id: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute 13-dimension scores for all runners in a race.

    Returns dict mapping horse_name -> list of 13 dimension dicts:
      [{"name": "Rating", "score": 78.3, "features": {...}, "tooltip": "..."}, ...]
    """
    if race_id is not None:
        df = features_df[features_df["race_id"] == race_id].copy()
    else:
        df = features_df.copy()

    if len(df) == 0:
        return {}

    results = {}

    for _, runner in df.iterrows():
        horse_name = runner.get("horse_name", "")
        result_id = runner.get("result_id", 0)
        dimensions = []

        for dim_name, feature_weights in DIMENSION_MAP.items():
            weighted_scores = []
            total_weight = 0
            feature_details = {}

            for feat_name, weight in feature_weights.items():
                val = runner.get(feat_name)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue

                if feat_name in df.columns:
                    col = pd.to_numeric(df[feat_name], errors="coerce")
                    if col.notna().sum() > 0:
                        pct = col.rank(pct=True, na_option="keep")
                        runner_pct = pct.loc[runner.name] if runner.name in pct.index else 0.5
                        if pd.isna(runner_pct):
                            runner_pct = 0.5
                    else:
                        runner_pct = 0.5
                else:
                    runner_pct = 0.5

                if weight < 0:
                    score = (1.0 - runner_pct) * 100
                else:
                    score = runner_pct * 100

                abs_weight = abs(weight)
                weighted_scores.append(score * abs_weight)
                total_weight += abs_weight
                feature_details[feat_name] = round(score, 1)

            if total_weight > 0:
                dim_score = sum(weighted_scores) / total_weight
            else:
                dim_score = 50.0

            dim_score = max(0, min(100, dim_score))

            dimensions.append({
                "name": dim_name,
                "score": round(dim_score, 1),
                "features": feature_details,
                "tooltip": _dimension_tooltip(dim_name, dim_score),
                "color": DIM_COLORS.get(dim_name, "#6b7280"),
            })

        if horse_name:
            results[horse_name] = dimensions
        results[result_id] = dimensions

    return results


def compute_drivers(
    features_df: pd.DataFrame,
    horse_name: str,
    race_id: int,
) -> tuple:
    """Compute positive and negative drivers for a runner."""
    df = features_df[features_df["race_id"] == race_id].copy()
    runner = df[df["horse_name"] == horse_name]
    if runner.empty:
        return [], []

    runner = runner.iloc[0]
    positives = []
    negatives = []

    checks = [
        ("form_win_rate", "Recent Win Rate", True),
        ("form_place_rate", "Recent Place Rate", True),
        ("jockey_win_rate_365d", "Jockey Win Rate", True),
        ("trainer_win_rate_365d", "Trainer Win Rate", True),
        ("official_rating", "Official Rating", True),
        ("days_since_last_race", "Days Since Last Race", False),
        ("weight_lbs", "Weight Carried", False),
        ("form_avg_pos_6", "Avg Position (last 6)", False),
        ("course_match_win_rate", "Course Win Rate", True),
        ("going_match_win_rate", "Going Win Rate", True),
        ("dist_match_win_rate", "Distance Win Rate", True),
        ("weather_precipitation", "Race Day Rain (mm)", False),
        ("weather_wind_speed", "Race Day Wind (km/h)", False),
        ("best_speed_last_6", "Best Speed Figure", True),
        ("avg_speed_last_3", "Avg Speed (last 3)", True),
        ("speed_vs_field_avg", "Speed vs Field", True),
    ]

    for feat, label, higher_is_better in checks:
        val = runner.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue

        col = pd.to_numeric(df[feat], errors="coerce") if feat in df.columns else None
        if col is None or col.notna().sum() == 0:
            continue

        pct = col.rank(pct=True, na_option="keep").loc[runner.name]
        if pd.isna(pct):
            continue

        if higher_is_better:
            if pct >= 0.75:
                positives.append({
                    "feature": label,
                    "value": _format_value(feat, val),
                    "impact": "positive",
                    "description": f"Top {int((1 - pct) * 100)}% in field",
                })
            elif pct <= 0.25:
                negatives.append({
                    "feature": label,
                    "value": _format_value(feat, val),
                    "impact": "negative",
                    "description": f"Bottom {int(pct * 100)}% in field",
                })
        else:
            if pct <= 0.25:
                positives.append({
                    "feature": label,
                    "value": _format_value(feat, val),
                    "impact": "positive",
                    "description": f"Best {int(pct * 100)}% in field",
                })
            elif pct >= 0.75:
                negatives.append({
                    "feature": label,
                    "value": _format_value(feat, val),
                    "impact": "negative",
                    "description": f"Worst {int((1 - pct) * 100)}% in field",
                })

    return positives[:5], negatives[:5]


def _format_value(feat: str, val) -> str:
    """Format a feature value for display."""
    if "rate" in feat or "pct" in feat:
        return f"{float(val) * 100:.0f}%"
    if "days" in feat:
        return f"{int(val)}d"
    if "weight" in feat:
        return f"{int(val)}lbs"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _dimension_tooltip(name: str, score: float) -> str:
    """Human-readable tooltip for a dimension score."""
    if score >= 80:
        strength = "Elite"
    elif score >= 65:
        strength = "Strong"
    elif score >= 50:
        strength = "Average"
    elif score >= 35:
        strength = "Below average"
    else:
        strength = "Weak"

    tooltips = {
        "Schedule": f"{strength} race spacing ({score:.0f}/100). Freshness and recent activity.",
        "Age": f"{strength} age profile ({score:.0f}/100). Maturity and peak fitness window.",
        "Gender": f"{strength} gender factor ({score:.0f}/100). Sex-based performance pattern.",
        "Weight": f"{strength} weight burden ({score:.0f}/100). Carried weight vs field.",
        "Rating": f"{strength} ability rating ({score:.0f}/100). RPR, OR, TSR combined.",
        "Form": f"{strength} recent form ({score:.0f}/100). Win/place rate, consistency, momentum.",
        "Jockey": f"{strength} jockey ({score:.0f}/100). Win rate, course record, combo.",
        "Load": f"{strength} weight-to-ability ({score:.0f}/100). Penalty/advantage ratio.",
        "Race Type": f"{strength} race type fit ({score:.0f}/100). Flat/jumps/class match.",
        "History": f"{strength} race history ({score:.0f}/100). Avg position, distance/course form.",
        "Ground": f"{strength} ground match ({score:.0f}/100). Going and surface suitability.",
        "Field": f"{strength} field dynamics ({score:.0f}/100). Draw bias, field size.",
        "Trainer": f"{strength} trainer ({score:.0f}/100). Win rate, course record.",
        "Speed": f"{strength} speed profile ({score:.0f}/100). Pace ability and trend.",
    }
    return tooltips.get(name, f"{strength} ({score:.0f}/100)")
