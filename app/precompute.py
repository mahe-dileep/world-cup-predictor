"""Precompute the heavy tournament simulations into app/data/tournament.json so
the Streamlit app loads instantly. Run once (or whenever the model changes):

    python -m app.precompute
"""
import json
from pathlib import Path

from src.prediction import WorldCup2026, Predictor, loaders
from src.prediction.utils import safe_json

OUT = Path(__file__).resolve().parent / "data" / "tournament.json"
SNAPSHOT = loaders.ROOT / "data" / "processed" / "team_form_snapshot.json"


def main():
    pred = Predictor()
    wc = WorldCup2026(pred, seed=42)

    # Bake each team's latest form snapshot from matches_ml so the app never needs
    # the 50 MB file at runtime.
    print("Writing team form snapshot...")
    snap = pred.builder._build_team_snapshots(loaders.get_matches_ml())
    SNAPSHOT.write_text(json.dumps(safe_json(snap)))
    print(f"Wrote {SNAPSHOT} ({len(snap)} teams)")

    print("Simulating from the group stage (real fixed bracket, 10k sims)...")
    from_groups = wc.simulate_from_groups(n_sims=10000)

    print("Projecting the current real bracket (played through R16, 20k sims)...")
    current = wc.project_current_bracket(n_sims=20000)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(safe_json({
        "from_groups": from_groups,
        "current_bracket": current,
        "groups": wc.groups,
    }), indent=2))
    print(f"Wrote {OUT}")
    top = list(from_groups["champion"].items())[:5]
    print("Top 5 champions (from groups):", [(t, round(p * 100, 1)) for t, p in top])


if __name__ == "__main__":
    main()
