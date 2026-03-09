"""
Lightweight NLP pipeline for horse racing comments.
Keyword-based extraction -- no external APIs, no transformers.
"""

import re
from typing import Dict, List, Optional

import numpy as np

# Positive keywords (good run, unlucky, better than result suggests)
_POSITIVE = [
    "stayed on well", "stayed on strongly", "ran on well", "finished strongly",
    "won easily", "impressive", "value for more", "better than bare form",
    "eye-catching", "much improved", "comfortably", "going away", "readily",
    "made all", "never in doubt", "idled", "could have won further",
    "looked the winner", "well on top", "pulling clear", "in command",
    "travelled strongly", "cruised", "tanked along",
]

# Negative keywords (poor performance, no excuses)
_NEGATIVE = [
    "never travelling", "never traveled", "weakened", "pulled up", "eased",
    "lost interest", "no impression", "always behind", "tailed off",
    "never dangerous", "beaten a long way", "no chance", "outpaced",
    "struggling", "well beaten", "no threat", "detached",
    "refused", "unseated", "fell", "brought down",
]

# Trip trouble (unlucky, may have done better with clear run)
_TRIP_TROUBLE = [
    "hampered", "bumped", "squeezed", "short of room", "denied clear run",
    "switched", "checked", "badly hampered", "carried wide", "crowded",
    "interfered with", "stumbled", "clipped heels", "lost place",
    "not much room", "tight for room", "no room", "blocked",
    "had to wait", "held up in rear", "wide throughout",
]

# Excuses (conditions didn't suit)
_EXCUSES = [
    "unsuited by going", "unsuited by ground", "too fast",
    "trip too short", "trip too far", "wrong trip",
    "wrong ground", "didn't handle", "never happy on",
    "found ground too", "not suited by",
]

# Front-runner keywords (for pace classification)
_FRONT_RUNNER = [
    "led", "made all", "made most", "prominent", "pressed leader",
    "disputed lead", "set pace", "blazed trail", "went clear",
    "led from", "front rank",
]

# Hold-up / closer keywords
_CLOSER = [
    "held up", "came from behind", "stayed on", "finished well",
    "flew home", "came with run", "produced late", "from rear",
    "patient ride", "waited with", "came wide",
]

# Stalker keywords
_STALKER = [
    "tracked leader", "tracked leaders", "handy", "close up",
    "prominent", "mid-division", "midfield", "in touch",
    "chased leader", "chased leaders",
]


def _count_matches(text: str, keywords: List[str]) -> int:
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def analyse_comment(comment: Optional[str]) -> Dict[str, float]:
    """Extract features from a single race comment."""
    if not comment or not isinstance(comment, str):
        return {
            "positive_signals": 0.0,
            "negative_signals": 0.0,
            "trip_trouble": 0.0,
            "had_excuse": 0.0,
            "comment_sentiment": 0.0,
        }

    text = comment.strip().lower()

    pos = _count_matches(text, _POSITIVE)
    neg = _count_matches(text, _NEGATIVE)
    trouble = _count_matches(text, _TRIP_TROUBLE)
    excuse = _count_matches(text, _EXCUSES)

    sentiment = (pos * 1.0 + trouble * 0.5) - neg * 1.0

    return {
        "positive_signals": float(pos),
        "negative_signals": float(neg),
        "trip_trouble": float(trouble),
        "had_excuse": float(min(excuse, 1)),
        "comment_sentiment": sentiment,
    }


def classify_run_style(comment: Optional[str]) -> int:
    """Classify run style from comment text.
    Returns: 1=front-runner, 2=stalker, 3=closer, 0=unknown
    """
    if not comment or not isinstance(comment, str):
        return 0

    text = comment.strip().lower()

    front = _count_matches(text, _FRONT_RUNNER)
    stalk = _count_matches(text, _STALKER)
    close = _count_matches(text, _CLOSER)

    if front > stalk and front > close:
        return 1
    if close > stalk and close > front:
        return 3
    if stalk > 0:
        return 2
    if front > 0:
        return 1
    return 0


def compute_comment_features(comments: List[Optional[str]]) -> Dict[str, float]:
    """Aggregate comment features from a list of recent race comments.
    Returns averaged features over the comment history.
    """
    if not comments:
        return {
            "comment_sentiment_avg": np.nan,
            "comment_sentiment_trend": np.nan,
            "trip_trouble_count": np.nan,
            "positive_comment_count": np.nan,
            "negative_comment_count": np.nan,
            "had_excuse_count": np.nan,
        }

    analyses = [analyse_comment(c) for c in comments if c]
    if not analyses:
        return {
            "comment_sentiment_avg": np.nan,
            "comment_sentiment_trend": np.nan,
            "trip_trouble_count": np.nan,
            "positive_comment_count": np.nan,
            "negative_comment_count": np.nan,
            "had_excuse_count": np.nan,
        }

    sentiments = [a["comment_sentiment"] for a in analyses]
    feats = {
        "comment_sentiment_avg": np.mean(sentiments),
        "trip_trouble_count": sum(a["trip_trouble"] for a in analyses),
        "positive_comment_count": sum(a["positive_signals"] for a in analyses),
        "negative_comment_count": sum(a["negative_signals"] for a in analyses),
        "had_excuse_count": sum(a["had_excuse"] for a in analyses),
    }

    if len(sentiments) >= 2:
        half = len(sentiments) // 2
        recent = np.mean(sentiments[half:])
        older = np.mean(sentiments[:half])
        feats["comment_sentiment_trend"] = recent - older
    else:
        feats["comment_sentiment_trend"] = np.nan

    return feats


def compute_run_style_features(
    comments: List[Optional[str]],
    positions: List[Optional[int]],
) -> Dict[str, float]:
    """Classify a horse's preferred run style from career comments.
    Returns style encoding and confidence.
    """
    if not comments:
        return {"run_style": np.nan}

    styles = [classify_run_style(c) for c in comments if c]
    styles = [s for s in styles if s > 0]

    if not styles:
        return {"run_style": np.nan}

    from collections import Counter
    counts = Counter(styles)
    dominant = counts.most_common(1)[0][0]

    return {"run_style": float(dominant)}
