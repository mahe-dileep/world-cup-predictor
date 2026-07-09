"""
Phase 1 — Canonical team identity layer.

Builds data/interim/team_name_mapping.csv, the single source of truth that
reconciles the different name each data source uses for the 48 World Cup 2026
teams. EVERY join in the project should go through this file.

Sources
-------
- data/raw/world_cup_2026/teams.csv      -> canonical name + team_id (the universe)
- data/raw/teams/teams.csv               -> historical stats naming
- data/processed/latest_elo.csv          -> Elo naming
- data/raw/transfermarkt/team_mapping.csv-> Transfermarkt naming + club id

Output columns
--------------
worldcup_name, historical_name, elo_name, transfermarkt_name, team_id, transfermarkt_id
"""
from pathlib import Path
import pandas as pd

WORLDCUP = Path("data/raw/world_cup_2026/teams.csv")
HISTORICAL = Path("data/raw/teams/teams.csv")
ELO = Path("data/processed/latest_elo.csv")
TRANSFERMARKT = Path("data/raw/transfermarkt/team_mapping.csv")
OUTPUT = Path("data/interim/team_name_mapping.csv")

# WC canonical name -> name used by the HISTORICAL stats file.
# Only overrides are listed; everything else matches the canonical name exactly.
HISTORICAL_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia–Herz",  # en-dash form in source
    "South Korea": "Korea Republic",
    "USA": "United States",
}

# WC canonical name -> name used by the ELO system.
ELO_ALIASES = {
    "USA": "United States",
    "Congo DR": "DR Congo",          # NB: elo also has "Congo" (Rep. of Congo) - do not use
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
}

# Transfermarkt's own world_cup_name -> WC canonical name (only the variants).
TM_TO_CANONICAL = {
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Turkey": "Türkiye",
}


def build_mapping():
    wc = pd.read_csv(WORLDCUP)
    hist = pd.read_csv(HISTORICAL)
    elo = pd.read_csv(ELO)
    tm = pd.read_csv(TRANSFERMARKT)

    hist_names = set(hist["team"].astype(str))
    elo_names = set(elo["team"].astype(str))

    # Transfermarkt lookup keyed by canonical WC name.
    tm = tm.copy()
    tm["canonical"] = tm["world_cup_name"].map(lambda n: TM_TO_CANONICAL.get(n, n))
    tm_lookup = tm.set_index("canonical")[["transfermarkt_name", "club_id"]].to_dict("index")

    rows = []
    unresolved = []
    for _, r in wc.iterrows():
        canon = r["team_name"]

        hist_name = HISTORICAL_ALIASES.get(canon, canon)
        if hist_name not in hist_names:
            unresolved.append(("historical", canon, hist_name))
            hist_name = ""

        elo_name = ELO_ALIASES.get(canon, canon)
        if elo_name not in elo_names:
            unresolved.append(("elo", canon, elo_name))
            elo_name = ""

        tm_entry = tm_lookup.get(canon)
        if tm_entry is None:
            unresolved.append(("transfermarkt", canon, "(no entry)"))
            tm_name, tm_id = "", ""
        else:
            tm_name = tm_entry["transfermarkt_name"]
            tm_id = tm_entry["club_id"]

        rows.append({
            "worldcup_name": canon,
            "historical_name": hist_name,
            "elo_name": elo_name,
            "transfermarkt_name": tm_name,
            "team_id": r["team_id"],
            "transfermarkt_id": tm_id,
        })

    mapping = pd.DataFrame(rows, columns=[
        "worldcup_name", "historical_name", "elo_name",
        "transfermarkt_name", "team_id", "transfermarkt_id",
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT} with {len(mapping)} teams.")
    print(f"  historical resolved   : {(mapping['historical_name'] != '').sum()}/48")
    print(f"  elo resolved          : {(mapping['elo_name'] != '').sum()}/48")
    print(f"  transfermarkt resolved: {(mapping['transfermarkt_name'] != '').sum()}/48")

    if unresolved:
        print("\nUNRESOLVED mappings:")
        for source, canon, attempted in unresolved:
            print(f"  [{source}] {canon!r} -> {attempted!r} not found")
    else:
        print("\nAll mappings resolved.")

    return mapping, unresolved


if __name__ == "__main__":
    build_mapping()
