"""Convert model probabilities into betting odds.

Provides fair odds (100% book, no margin) and bookmaker odds with a configurable
overround (margin). Both decimal and American formats.
"""
from __future__ import annotations


def decimal_odds(p: float, eps: float = 1e-9) -> float:
    return round(1.0 / max(p, eps), 2)


def american_odds(dec: float):
    if dec <= 1.0:
        return None
    if dec >= 2.0:
        return f"+{round(100 * (dec - 1))}"
    return f"-{round(100 / (dec - 1))}"


def odds_line(p: float) -> dict:
    """Fair odds for a single probability."""
    d = decimal_odds(p)
    return {"prob": round(p, 4), "decimal": d, "american": american_odds(d)}


def market_odds(probs: dict, margin: float = 0.05) -> dict:
    """Odds for a set of mutually-exclusive outcomes.

    `margin` is the bookmaker overround: displayed implied probabilities sum to
    (1 + margin), i.e. the book's built-in edge. margin=0 gives fair odds.
    """
    out = {}
    for name, p in probs.items():
        fair_d = decimal_odds(p)
        book_p = p * (1 + margin)
        book_d = decimal_odds(book_p)
        out[name] = {
            "prob": round(p, 4),
            "fair_decimal": fair_d,
            "fair_american": american_odds(fair_d),
            "book_decimal": book_d,
            "book_american": american_odds(book_d),
        }
    return out
