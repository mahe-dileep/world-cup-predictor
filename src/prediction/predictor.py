"""Production Predictor — the single entry point between the trained model and
any future API/UI. Loads once, caches everything, deterministic."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.prediction import loaders, validation as V
from src.prediction.feature_builder import FeatureBuilder, DEFAULT_COMPETITION, DEFAULT_NEUTRAL
from src.prediction import scoreline as SL
from src.prediction import betting as BET

OUTPUT_KEYS = ["home_team", "away_team", "home_win", "draw", "away_win", "predicted_result"]


class Predictor:
    def __init__(self):
        self.model = loaders.get_model()
        self.feature_columns = loaders.get_feature_columns()
        self.label_encoder = loaders.get_label_encoder()
        self.metadata = loaders.get_metadata()
        self.builder = FeatureBuilder()
        self.classes_ = list(self.label_encoder.classes_)   # ['A','D','H']
        self._idx = {c: i for i, c in enumerate(self.classes_)}

    # ---- low level ----
    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        V.validate_feature_frame(feature_df, self.feature_columns)
        P = np.asarray(self.model.predict_proba(feature_df), dtype=float)
        return V.validate_probabilities(P)

    def _format(self, home, away, proba_row) -> dict:
        pred = self.classes_[int(np.argmax(proba_row))]
        return {
            "home_team": home,
            "away_team": away,
            "home_win": float(proba_row[self._idx["H"]]),
            "draw": float(proba_row[self._idx["D"]]),
            "away_win": float(proba_row[self._idx["A"]]),
            "predicted_result": pred,
        }

    # ---- high level ----
    def predict_match(self, home_team: str, away_team: str,
                      competition: str = DEFAULT_COMPETITION,
                      neutral: bool = DEFAULT_NEUTRAL) -> dict:
        X = self.builder.build_match_features(home_team, away_team, competition, neutral)
        P = self.predict_proba(X)
        return self._format(home_team, away_team, P[0])

    def predict_many(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        X = self.builder.build_many(fixtures_df)
        P = self.predict_proba(X)
        records = [self._format(r.home_team, r.away_team, P[i])
                   for i, r in enumerate(fixtures_df.itertuples(index=False))]
        return pd.DataFrame.from_records(records, columns=OUTPUT_KEYS)

    # ---- convenience for the simulator ----
    def proba_dict(self, home_team, away_team,
                   competition=DEFAULT_COMPETITION, neutral=DEFAULT_NEUTRAL) -> dict:
        out = self.predict_match(home_team, away_team, competition, neutral)
        return {"H": out["home_win"], "D": out["draw"], "A": out["away_win"]}

    # ---- exact scoreline ----
    def predict_score(self, home_team, away_team,
                      competition=DEFAULT_COMPETITION, neutral=DEFAULT_NEUTRAL,
                      max_goals=SL.MAX_GOALS) -> dict:
        """Most-likely scoreline + distribution, consistent with the H/D/A model.

        Expected goals start from each team's attack/defence form (avg goals
        scored/conceded over last 10) and are fitted so the Poisson result
        probabilities match the model's probabilities.
        """
        probs = self.proba_dict(home_team, away_team, competition, neutral)
        hf = self.builder.build_features(home_team)
        af = self.builder.build_features(away_team)
        lh0 = 0.5 * (hf["avg_goals_scored_10"] + af["avg_goals_conceded_10"])
        la0 = 0.5 * (af["avg_goals_scored_10"] + hf["avg_goals_conceded_10"])
        lh, la = SL.fit_lambdas(probs["H"], probs["D"], probs["A"], init=(lh0, la0),
                                max_goals=max_goals)
        M = SL.score_matrix(lh, la, max_goals)
        ts = SL.top_scores(M, 5)
        (i, j), p = ts[0]
        return {
            "home_team": home_team, "away_team": away_team,
            "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
            "most_likely_score": f"{i}-{j}",
            "most_likely_score_prob": round(p, 4),
            "top_scores": [{"score": f"{a}-{b}", "prob": round(pp, 4)} for (a, b), pp in ts],
            "result_probs": {k: round(v, 4) for k, v in probs.items()},
            "markets": {k: round(v, 4) for k, v in SL.markets(M).items()},
        }

    # ---- betting odds ----
    def predict_odds(self, home_team, away_team, margin: float = 0.05,
                     competition=DEFAULT_COMPETITION, neutral=DEFAULT_NEUTRAL) -> dict:
        """Betting odds (fair + with bookmaker margin) for the main markets."""
        s = self.predict_score(home_team, away_team, competition, neutral)
        p = s["result_probs"]
        m = s["markets"]
        lh, la = s["expected_goals"]["home"], s["expected_goals"]["away"]
        M = SL.score_matrix(lh, la)
        one_x_two = {home_team + " win": p["H"], "Draw": p["D"], away_team + " win": p["A"]}

        # correct-score odds for the six likeliest scorelines
        correct_score = BET.market_odds(
            {f"{i}-{j}": prob for (i, j), prob in SL.top_scores(M, 6)}, margin)

        # Asian handicap on the main line (push-adjusted two-way pricing)
        ah = SL.asian_handicap(M, SL.main_handicap_line(lh, la))
        denom = ah["home_cover"] + ah["away_cover"] or 1.0
        line = ah["line"]
        asian = BET.market_odds({
            f"{home_team} {line:+g}": ah["home_cover"] / denom,
            f"{away_team} {-line:+g}": ah["away_cover"] / denom,
        }, margin)
        asian["line"] = line
        asian["push_prob"] = round(ah["push"], 4)

        return {
            "home_team": home_team, "away_team": away_team, "margin": margin,
            "match_result": BET.market_odds(one_x_two, margin),
            "over_under_2.5": BET.market_odds(
                {"Over 2.5": m["over_2.5"], "Under 2.5": m["under_2.5"]}, margin),
            "both_teams_score": BET.market_odds(
                {"Yes": m["btts_yes"], "No": m["btts_no"]}, margin),
            "correct_score": correct_score,
            "asian_handicap": asian,
            "most_likely_score": s["most_likely_score"],
            "expected_goals": s["expected_goals"],
        }
