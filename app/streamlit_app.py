"""TOUCHLINE — World Cup 2026 match model, odds & tournament simulator.

Streamlit front-end over the inference package (src.prediction). Run:

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import Predictor, WorldCup2026  # noqa: E402

st.set_page_config(page_title="Touchline · WC2026", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the model…")
def get_predictor():
    return Predictor()


@st.cache_data
def load_tournament():
    return json.loads((ROOT / "app" / "data" / "tournament.json").read_text())


@st.cache_data
def load_features():
    return pd.read_csv(ROOT / "data" / "processed" / "features.csv")


@st.cache_data
def load_codes():
    t = pd.read_csv(ROOT / "data" / "raw" / "world_cup_2026" / "teams.csv")
    return dict(zip(t["team_name"], t["fifa_code"]))


@st.cache_data(show_spinner=False)
def score(home, away, neutral):
    return get_predictor().predict_score(home, away, neutral=neutral)


@st.cache_data(show_spinner=False)
def odds(home, away, neutral, margin):
    return get_predictor().predict_odds(home, away, margin=margin, neutral=neutral)


@st.cache_data(show_spinner=False)
def run_match_mc(home, away, neutral, n, seed):
    """Replay a match n times by sampling scorelines from the model's goal model."""
    import numpy as np
    from collections import Counter
    from src.prediction import scoreline as SL
    s = get_predictor().predict_score(home, away, neutral=neutral)
    lh, la = s["expected_goals"]["home"], s["expected_goals"]["away"]
    flat = SL.score_matrix(lh, la).ravel()
    flat = flat / flat.sum()
    rng = np.random.default_rng(seed)
    ii, jj = np.unravel_index(rng.choice(flat.size, size=n, p=flat), (11, 11))
    hw, dw, aw = int((ii > jj).sum()), int((ii == jj).sum()), int((ii < jj).sum())
    total = ii + jj
    cnt = Counter(zip(ii.tolist(), jj.tolist()))
    return {
        "predicted": s["result_probs"],
        "emp": {"H": hw / n, "D": dw / n, "A": aw / n},
        "counts": {"H": hw, "D": dw, "A": aw},
        "top_scores": [(f"{i}-{j}", c / n) for (i, j), c in cnt.most_common(8)],
        "over25": float((total >= 3).mean()), "avg_goals": float(total.mean()),
        "expected_goals": s["expected_goals"], "n": n,
    }


@st.cache_resource(show_spinner="Warming the tournament model…")
def get_wc():
    wc = WorldCup2026(get_predictor(), seed=42)
    ts = wc.teams
    for i, a in enumerate(ts):           # warm every pairing once so re-runs are fast
        for b in ts[i + 1:]:
            wc.sim.symmetric_probs(a, b)
            wc.sim.advance_prob(a, b)
    return wc


@st.cache_data(show_spinner="Simulating the tournament…")
def sim_tournament(n_sims, seed):
    return get_wc().simulate_from_groups(n_sims=n_sims, seed=seed)


# ---------------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;800;900&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

:root{
  --pitch:#0a2416; --surface:#0e3020; --surface2:#0c2a1c; --line:rgba(243,240,231,.09);
  --chalk:#F3F0E7; --dim:#89A395; --amber:#FFC23C; --brick:#E0603A; --grass:#63C98A;
}
.stApp{ background:
   radial-gradient(1100px 460px at 78% -12%, #133f29 0%, rgba(19,63,41,0) 62%),
   linear-gradient(180deg,#0a2416 0%,#07180f 100%); }
html,body,[class*="css"]{ font-family:'Inter',sans-serif; color:var(--chalk); }
.num{ font-family:'Space Mono',monospace; font-variant-numeric:tabular-nums; }

/* hide Streamlit's rainbow decoration; give the stage a measured width */
[data-testid="stDecoration"]{ display:none!important; }
[data-testid="stMainBlockContainer"], .block-container{ max-width:1160px; padding-top:2rem; }
[data-testid="stHeader"]{ background:transparent; }

/* structural eyebrow used above every section */
.eyebrow{ font-family:'Archivo',sans-serif; font-weight:800; text-transform:uppercase;
  letter-spacing:.14em; font-size:.68rem; color:var(--dim); margin:.2rem 0 .7rem; }

/* wordmark */
.brand{ display:flex; align-items:baseline; gap:.6rem; }
.brand h1{ font-family:'Archivo',sans-serif; font-weight:900; letter-spacing:-.045em;
  font-size:2.2rem; margin:0; line-height:.9; text-transform:uppercase; }
.brand .dot{ width:.55rem;height:.55rem;border-radius:50%;background:var(--amber);
  box-shadow:0 0 16px var(--amber); display:inline-block; }
.brand .sub{ color:var(--dim); font-size:.82rem; letter-spacing:.01em; }
.rule{ height:1px; background:var(--line); margin:.7rem 0 1.4rem; }

/* SIGNATURE — the scoreboard, with a faint pitch-marking motif */
.board{ position:relative; overflow:hidden;
  background:linear-gradient(180deg,#11361f,#0a2214);
  border:1px solid var(--line); border-radius:18px; padding:1.6rem 1.8rem; }
.board::before{ content:""; position:absolute; top:0; bottom:0; left:50%; width:1px;
  background:rgba(243,240,231,.05); }
.board::after{ content:""; position:absolute; left:50%; top:44%; width:168px; height:168px;
  transform:translate(-50%,-50%); border:1px solid rgba(243,240,231,.05); border-radius:50%; }
.board > *{ position:relative; z-index:1; }
.board .teams{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:1rem;}
.team-side.away{ text-align:right; }
.team-code{ font-family:'Archivo',sans-serif; font-weight:900; font-size:2.5rem;
  letter-spacing:.01em; line-height:1; }
.team-full{ color:var(--dim); font-size:.8rem; margin-top:.3rem; letter-spacing:.01em; }
.score{ font-family:'Space Mono',monospace; font-weight:700; font-size:3.4rem;
  letter-spacing:.02em; text-align:center; white-space:nowrap; padding:0 .3rem;
  color:var(--amber); text-shadow:0 0 30px rgba(255,194,60,.28);}
.score small{ display:block; font-size:.64rem; color:var(--dim); letter-spacing:.26em;
  text-transform:uppercase; margin-top:.35rem; }
.xg{ display:flex; justify-content:space-between; margin-top:.9rem; padding-top:.7rem;
  border-top:1px solid var(--line); color:var(--dim); font-size:.78rem; }
.xg .num{ color:var(--chalk); }

/* probability momentum bar */
.pbar{ display:flex; height:32px; border-radius:8px; overflow:hidden; margin:.15rem 0 .3rem; }
.pbar > div{ display:flex; align-items:center; justify-content:center; font-size:.78rem;
  font-family:'Space Mono',monospace; font-weight:700; color:#07180f; }
.seg-h{ background:var(--amber); }
.seg-d{ background:#2f4a3a; color:var(--dim)!important; }
.seg-a{ background:var(--brick); }
.pkey{ display:flex; justify-content:space-between; color:var(--dim); font-size:.7rem;
  letter-spacing:.1em; text-transform:uppercase; margin-bottom:.15rem; }

/* cards + rows */
.card{ background:var(--surface2); border:1px solid var(--line); border-radius:14px;
  padding:1.1rem 1.25rem; height:100%; }
.card h4{ margin:0 0 .85rem; font-family:'Archivo',sans-serif; font-weight:800;
  text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; color:var(--dim); }
.row{ display:flex; justify-content:space-between; align-items:center; padding:.34rem 0;
  border-bottom:1px solid var(--line); }
.row:last-child{ border-bottom:none; }
.row .k{ color:var(--dim); font-size:.88rem; } .row .v{ font-family:'Space Mono',monospace; }

/* stat chip */
.chip .lab{ font-family:'Archivo',sans-serif; font-weight:800; text-transform:uppercase;
  letter-spacing:.08em; font-size:.64rem; color:var(--dim); }
.chip .big{ font-family:'Space Mono',monospace; font-size:1.7rem; margin-top:.35rem; }
.chip .unit{ color:var(--dim); font-size:.66rem; margin-top:.1rem; }
.pill{ font-family:'Space Mono',monospace; font-size:.74rem; padding:.1rem .5rem;
  border:1px solid var(--line); border-radius:999px; color:var(--amber); }

/* odds tables */
table.odds{ width:100%; border-collapse:collapse; font-family:'Space Mono',monospace; }
table.odds td,table.odds th{ padding:.44rem .2rem; border-bottom:1px solid var(--line);
  text-align:right; font-size:.88rem; }
table.odds tr:last-child td{ border-bottom:none; }
table.odds th{ color:var(--dim); text-transform:uppercase; letter-spacing:.1em; font-size:.62rem; }
table.odds td:first-child,table.odds th:first-child{ text-align:left;
  font-family:'Inter',sans-serif; color:var(--chalk); }
.am{ color:var(--grass); }

/* progress bars (leaderboard + teams table) */
.barwrap{ background:rgba(0,0,0,.3); border-radius:6px; overflow:hidden; }
.bar{ height:8px; border-radius:6px; background:var(--amber); }
.lb .row{ padding:.46rem 0; }
.lb .barwrap{ flex:1; margin:0 .8rem; }

/* streamlit chrome */
[data-baseweb="tab-list"]{ gap:1.3rem; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"]{ font-family:'Archivo',sans-serif!important; font-weight:800;
  text-transform:uppercase; letter-spacing:.06em; font-size:.76rem; padding:.4rem 0; }
button[data-baseweb="tab"]{ color:var(--dim); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--chalk); }
[data-baseweb="tab-highlight"]{ background:var(--amber)!important; height:2px; }
section[data-testid="stSidebar"]{ background:#081a10; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(s):
    return str(s)


def scoreboard(home, away, sc, xg):
    h, a = sc.split("-")
    hc, ac = CODES.get(home, home[:3].upper()), CODES.get(away, away[:3].upper())
    return f"""
    <div class="board"><div class="teams">
      <div class="team-side"><div class="team-code">{hc}</div>
        <div class="team-full">{esc(home)}</div></div>
      <div class="score">{h}<span style="color:var(--dim)"> – </span>{a}
        <small>most likely score</small></div>
      <div class="team-side away"><div class="team-code">{ac}</div>
        <div class="team-full">{esc(away)}</div></div>
    </div>
    <div class="xg"><span>xG <span class="num">{xg['home']}</span></span>
      <span>expected goals</span><span><span class="num">{xg['away']}</span> xG</span></div>
    </div>"""


def prob_bar(H, D, A, home, away):
    def w(x):
        return max(x * 100, 6)
    return f"""
    <div class="pkey"><span>{esc(home)}</span><span>Draw</span><span>{esc(away)}</span></div>
    <div class="pbar">
      <div class="seg-h" style="width:{w(H)}%">{H*100:.0f}%</div>
      <div class="seg-d" style="width:{w(D)}%">{D*100:.0f}%</div>
      <div class="seg-a" style="width:{w(A)}%">{A*100:.0f}%</div>
    </div>"""


def odds_table(header, rows):
    body = "".join(
        f"<tr><td>{name}</td><td class='num'>{d['fair_decimal']}</td>"
        f"<td class='am'>{d['fair_american']}</td><td class='num'>{d['book_decimal']}</td></tr>"
        for name, d in rows.items())
    return f"""<table class="odds"><tr><th>{header}</th><th>Fair</th><th>US</th><th>Book</th></tr>
    {body}</table>"""


# ----------------------------------------------------------------- controls
feats = load_features()
teams = sorted(feats["team"].tolist())
tour = load_tournament()
CODES = load_codes()

with st.sidebar:
    st.markdown("<div class='brand'><span class='dot'></span>"
                "<span style='font-family:Archivo;font-weight:900;font-size:1.05rem;"
                "letter-spacing:-.03em;text-transform:uppercase'>Touchline</span></div>"
                "<div class='rule'></div><div class='eyebrow'>Match setup</div>",
                unsafe_allow_html=True)
    home = st.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = st.selectbox("Away team", teams, index=teams.index("France") if "France" in teams else 1)
    neutral = st.toggle("Neutral venue", value=True, help="World Cup matches are neutral.")
    margin = st.slider("Bookmaker margin", 0.0, 0.15, 0.05, 0.01,
                       help="The book's built-in edge (overround). 0 = fair odds.")
    if home == away:
        st.warning("Pick two different teams.")

# ----------------------------------------------------------------- header
st.markdown(
    "<div class='brand'><span class='dot'></span><h1>Touchline</h1></div>"
    "<div class='eyebrow' style='margin-top:.55rem'>World Cup 2026 · neutral-venue match "
    "model, odds &amp; tournament simulator</div><div class='rule'></div>",
    unsafe_allow_html=True)

tab_match, tab_odds, tab_mc, tab_wc, tab_teams = st.tabs(
    ["Match", "Odds board", "Monte Carlo", "World Cup 2026", "Teams"])

# ============================================================= MATCH
with tab_match:
    if home != away:
        s = score(home, away, neutral)
        rp = s["result_probs"]
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.markdown(scoreboard(home, away, s["most_likely_score"], s["expected_goals"]),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown(prob_bar(rp["H"], rp["D"], rp["A"], home, away), unsafe_allow_html=True)
        with c2:
            top = "".join(f"<div class='row'><span class='k'>{t['score']}</span>"
                          f"<span class='v'>{t['prob']*100:.1f}%</span></div>"
                          for t in s["top_scores"])
            st.markdown(f"<div class='card'><h4>Likeliest scorelines</h4>{top}</div>",
                        unsafe_allow_html=True)
        # form read
        pred = get_predictor()
        hf = pred.builder.build_features(home)
        af = pred.builder.build_features(away)
        m = s["markets"]
        hcode, acode = CODES.get(home, home[:3].upper()), CODES.get(away, away[:3].upper())
        cols = st.columns(4)
        chips = [("Over 2.5 goals", f"{m['over_2.5']*100:.0f}%", "chance"),
                 ("Both teams score", f"{m['btts_yes']*100:.0f}%", "chance"),
                 (f"{hcode} recent form", f"{hf['ppg_5']:.2f}", "points / game · last 5"),
                 (f"{acode} recent form", f"{af['ppg_5']:.2f}", "points / game · last 5")]
        for col, (k, v, u) in zip(cols, chips):
            col.markdown(f"<div class='card chip'><div class='lab'>{k}</div>"
                         f"<div class='big'>{v}</div><div class='unit'>{u}</div></div>",
                         unsafe_allow_html=True)

# ============================================================= ODDS BOARD
with tab_odds:
    if home != away:
        o = odds(home, away, neutral, margin)
        st.caption(f"Odds for **{home} vs {away}** · neutral={neutral} · "
                   f"bookmaker margin {margin*100:.0f}% · fair = no margin")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='card'><h4>Match result (1X2)</h4>"
                    f"{odds_table('Outcome', o['match_result'])}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><h4>Goals over / under 2.5</h4>"
                    f"{odds_table('Line', o['over_under_2.5'])}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><h4>Both teams to score</h4>"
                    f"{odds_table('BTTS', o['both_teams_score'])}</div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        c4.markdown(f"<div class='card'><h4>Correct score</h4>"
                    f"{odds_table('Score', o['correct_score'])}</div>", unsafe_allow_html=True)
        ah = o["asian_handicap"]
        ah_rows = {k: v for k, v in ah.items() if isinstance(v, dict)}
        c5.markdown(f"<div class='card'><h4>Asian handicap "
                    f"<span class='pill'>line {ah['line']:+g}</span></h4>"
                    f"{odds_table('Handicap', ah_rows)}"
                    f"<div class='row'><span class='k'>Push probability</span>"
                    f"<span class='v'>{ah['push_prob']*100:.1f}%</span></div></div>",
                    unsafe_allow_html=True)

# ============================================================= MONTE CARLO
with tab_mc:
    if home != away:
        hcode, acode = CODES.get(home, home[:3].upper()), CODES.get(away, away[:3].upper())
        cc1, cc2 = st.columns([2, 1])
        replays = cc1.select_slider("Replays", options=[100, 1000, 10000, 100000],
                                    value=10000, format_func=lambda n: f"{n:,}")
        seed_mc = cc2.number_input("Seed", value=42, step=1, key="mc_seed")
        r = run_match_mc(home, away, neutral, int(replays), int(seed_mc))
        pr, em, ct = r["predicted"], r["emp"], r["counts"]
        se = (0.25 / replays) ** 0.5
        maxd = max(abs(pr[k] - em[k]) for k in "HDA")

        def _bar(H, D, A):
            w = lambda x: max(x * 100, 5)
            return (f"<div class='pbar'><div class='seg-h' style='width:{w(H)}%'>{H*100:.0f}%</div>"
                    f"<div class='seg-d' style='width:{w(D)}%'>{D*100:.0f}%</div>"
                    f"<div class='seg-a' style='width:{w(A)}%'>{A*100:.0f}%</div></div>")

        st.markdown(f"<div class='eyebrow'>Replayed {replays:,} times · does the simulation "
                    f"converge on the model?</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='card'>"
            f"<div class='pkey'><span>{home}</span><span>Draw</span><span>{away}</span></div>"
            f"<div style='display:grid;grid-template-columns:82px 1fr;gap:.6rem .9rem;"
            f"align-items:center'>"
            f"<span class='eyebrow' style='margin:0'>Model</span>{_bar(pr['H'],pr['D'],pr['A'])}"
            f"<span class='eyebrow' style='margin:0'>Simulated</span>{_bar(em['H'],em['D'],em['A'])}"
            f"</div>"
            f"<div class='xg' style='margin-top:.9rem'>"
            f"<span>{hcode} won <span class='num'>{ct['H']:,}</span></span>"
            f"<span>drew <span class='num'>{ct['D']:,}</span></span>"
            f"<span><span class='num'>{ct['A']:,}</span> {acode} won</span></div></div>",
            unsafe_allow_html=True)

        c1, c2 = st.columns([1.3, 1])
        mxp = r["top_scores"][0][1]
        srows = ""
        for sc, p in r["top_scores"]:
            srows += (f"<div class='row'><span class='k' style='flex:0 0 46px'>{sc}</span>"
                      f"<span class='barwrap' style='flex:1;margin:0 .7rem'><span class='bar' "
                      f"style='display:block;width:{p/mxp*100:.0f}%'></span></span>"
                      f"<span class='v'>{p*100:.1f}%</span></div>")
        c1.markdown(f"<div class='card'><h4>Simulated scorelines</h4>{srows}</div>",
                    unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='card chip'><div class='lab'>Average goals</div>"
                        f"<div class='big'>{r['avg_goals']:.2f}</div>"
                        f"<div class='unit'>per replay</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card chip'><div class='lab'>Model vs simulated</div>"
                        f"<div class='big'>±{maxd*100:.2f}%</div>"
                        f"<div class='unit'>largest gap · error ±{se*100:.2f}%</div></div>",
                        unsafe_allow_html=True)

# ============================================================= WORLD CUP
with tab_wc:
    fg = tour["from_groups"]
    champ = fg["champion"]
    cur = tour["current_bracket"]

    with st.expander("Run your own tournament from the group stage", expanded=False):
        e1, e2, e3 = st.columns([1.3, 1, 1.2])
        wc_n = e1.select_slider("Simulations", options=[1000, 2000, 5000, 10000],
                                value=2000, format_func=lambda n: f"{n:,}")
        wc_seed = e2.number_input("Seed", value=7, step=1, key="wc_seed")
        run = e3.button("Simulate tournament", type="primary", use_container_width=True)
        if run:
            res = sim_tournament(int(wc_n), int(wc_seed))
            fg, champ = res, res["champion"]
            first = next(iter(champ.items()))
            st.success(f"Simulated {wc_n:,} tournaments (seed {wc_seed}). "
                       f"Favourite: {first[0]} at {first[1]*100:.1f}%.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"<div class='card'><h4>Title odds · from the group stage · "
                    f"{fg.get('n_sims', 10000):,} sims</h4>", unsafe_allow_html=True)
        top = list(champ.items())[:12]
        mx = top[0][1]
        rows = ""
        for t, p in top:
            rows += (f"<div class='row'><span class='k' style='flex:0 0 130px'>{t}</span>"
                     f"<span class='barwrap'><span class='bar' style='display:block;"
                     f"width:{p/mx*100:.0f}%'></span></span>"
                     f"<span class='v'>{p*100:.1f}%</span></div>")
        st.markdown(f"<div class='lb'>{rows}</div></div>", unsafe_allow_html=True)
    with right:
        pf = fg["projected_final"]
        st.markdown(f"<div class='card'><h4>Projected final</h4>"
                    f"<div style='font-family:Archivo;font-weight:800;font-size:1.15rem;"
                    f"letter-spacing:-.01em'>{pf['matchup'][0]}"
                    f" <span style='color:var(--dim);font-weight:400'>vs</span> {pf['matchup'][1]}</div>"
                    f"<div class='score' style='font-size:2rem;margin:.4rem 0'>"
                    f"{pf['projected_score'].split(' ',1)[1].rsplit(' ',1)[0]}</div>"
                    f"<div class='row'><span class='k'>Most common final</span>"
                    f"<span class='v'>{pf['probability']*100:.1f}%</span></div>"
                    f"<div class='row'><span class='k'>Expected goals</span>"
                    f"<span class='v'>{pf['expected_goals']['home']} – {pf['expected_goals']['away']}</span>"
                    f"</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h4>Where the real cup stands</h4>"
                    f"<div class='row'><span class='k'>Played through</span>"
                    f"<span class='v'>Round of 16</span></div>"
                    f"<div class='row'><span class='k'>Quarterfinalists</span>"
                    f"<span class='v'>{len(cur['quarterfinalists'])}</span></div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>"
                "<div class='eyebrow'>The real quarterfinals · model read &amp; title odds "
                "from here</div>", unsafe_allow_html=True)
    qcols = st.columns(4)
    qf = cur["quarterfinals"]
    champ_now = cur["champion"]
    for col, tie in zip(qcols, qf):
        ph, pd_, pa = tie["probs"]
        h, a = tie["home"], tie["away"]
        col.markdown(
            f"<div class='card'><div class='row'><span class='k'>{h}</span>"
            f"<span class='v'>{ph*100:.0f}%</span></div>"
            f"<div class='row'><span class='k'>{a}</span><span class='v'>{pa*100:.0f}%</span></div>"
            f"<div class='row'><span class='k' style='color:var(--amber)'>title odds</span>"
            f"<span class='v'>{max(champ_now.get(h,0),0)*100:.0f}% / {champ_now.get(a,0)*100:.0f}%</span>"
            f"</div></div>", unsafe_allow_html=True)

# ============================================================= TEAMS
with tab_teams:
    champ = tour["from_groups"]["champion"]
    df = feats[["team", "elo", "squad_avg_age", "squad_total_market_value",
                "squad_total_caps"]].copy()
    df["title"] = df["team"].map(lambda t: champ.get(t, 0) * 100)
    df["mv"] = df["squad_total_market_value"] / 1e6

    sort_key = st.radio("Sort by", ["Title odds", "Elo", "Squad value", "Youngest"],
                        horizontal=True, label_visibility="collapsed")
    col = {"Title odds": ("title", False), "Elo": ("elo", False),
           "Squad value": ("mv", False), "Youngest": ("squad_avg_age", True)}[sort_key]
    df = df.sort_values(col[0], ascending=col[1]).reset_index(drop=True)
    mx = max(df["title"].max(), 1e-9)

    rows = ""
    for i, r in df.iterrows():
        rows += (
            f"<tr><td style='color:var(--dim)'>{i+1:>2}</td>"
            f"<td style='font-family:Inter'>{r['team']}</td>"
            f"<td class='num'>{r['elo']:.0f}</td>"
            f"<td style='width:230px'><span class='barwrap' style='display:inline-flex;width:150px;"
            f"vertical-align:middle'><span class='bar' style='display:block;height:8px;"
            f"width:{r['title']/mx*100:.0f}%'></span></span> "
            f"<span class='num' style='color:var(--amber)'>{r['title']:.1f}%</span></td>"
            f"<td class='num'>€{r['mv']:.0f}m</td>"
            f"<td class='num'>{r['squad_avg_age']:.1f}</td>"
            f"<td class='num'>{int(r['squad_total_caps'])}</td></tr>")
    st.markdown(
        "<div class='card'><h4>All 48 teams · Elo, title odds, squad value</h4>"
        "<table class='odds' style='text-align:left'>"
        "<tr><th>#</th><th>Team</th><th style='text-align:right'>Elo</th>"
        "<th>Title odds</th><th style='text-align:right'>Squad value</th>"
        "<th style='text-align:right'>Avg age</th><th style='text-align:right'>Caps</th></tr>"
        f"{rows}</table></div>", unsafe_allow_html=True)
