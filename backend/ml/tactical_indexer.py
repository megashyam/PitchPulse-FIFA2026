"""
Tactical fingerprint indexer.

Builds one "pressing fingerprint" document per team per match from
StatsBomb WC open data, embeds it and stores it in the Weaviate
TacticalProfiles collection. At runtime the TacticalCard matches a live
team's descriptor against these fingerprints.

PPDA (Passes allowed Per Defensive Action) is the standard
pressing-intensity metric; lower means a more aggressive press.

    PPDA = opponent completed passes / (tackles + interceptions + fouls + blocks)

Both numerator and denominator are restricted to the pressing team's
attacking 60% of the pitch (StatsBomb x >= 40 on the 120-long pitch). The
press is also broken down by thirds, so the fingerprint captures where a
team presses.

StatsBomb pitch: 120 long × 80 wide; x runs from the team's own goal (0) to
the opponent goal (120), in each team's own attacking direction.

Thirds (by x):
    defensive third : x <  40
    middle third    : 40 <= x < 80
    attacking third : x >= 80

Run from backend/:
    set PYTHONPATH=.
    python ml/tactical_indexer.py            # index
    python ml/tactical_indexer.py --check    # report count only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from agents.weaviate_client import (
    get_weaviate_client,
    TACTICAL_PROFILES,
)
from ml.statsbomb import COMPETITION_ID, SEASON_IDS, load

log = logging.getLogger(__name__)

MAX_MATCHES = 500

# Defensive-action event types that count toward PPDA denominator.
# StatsBomb: a Duel of sub-type Tackle, plus Interception / Block / Foul Committed.
DEF_ACTION_TYPES = {"Interception", "Block", "Foul Committed"}

# Pitch geometry
PITCH_X = 120.0
PRESS_LINE = 40.0  # standard PPDA boundary: actions/passes with x >= 40
MID_LINE = 40.0
ATT_LINE = 80.0


def _zone(x: float) -> str:
    if x < MID_LINE:
        return "def"
    if x < ATT_LINE:
        return "mid"
    return "att"


def _is_tackle(ev: dict) -> bool:
    """A Duel event counts as a defensive action only if it's a tackle."""
    if (ev.get("type") or {}).get("name") != "Duel":
        return False
    dtype = (ev.get("duel") or {}).get("type") or {}
    return "Tackle" in (dtype.get("name") or "")


def _safe_ppda(passes: float, actions: float) -> float:
    """PPDA with a guard. No defensive actions → treat as very passive (cap 50)."""
    if actions <= 0:
        return 50.0
    return round(passes / actions, 2)


def build_fingerprints(
    events: list,
    home: str,
    away: str,
    competition: str,
    season: str,
    match_id: str,
) -> list[dict]:
    """Up to two fingerprint dicts (one per team), ready to embed and insert.

    For team T pressing opponent O:
        numerator    O's completed passes starting in T's attacking 60%
                     (O-frame x' sits at T-frame x = 120 - x')
        denominator  T's defensive actions at T-frame x >= 40
    """
    # Per-team accumulators
    # opp_passes_by_zone[T][zone] = opponent completed passes T allowed in that
    #   T-frame zone; def_actions_by_zone[T][zone] = T's defensive actions there.
    opp_passes = {home: defaultdict(float), away: defaultdict(float)}
    def_actions = {home: defaultdict(float), away: defaultdict(float)}

    # Also track simple team identity stats for the descriptor text.
    poss_events = {home: 0, away: 0}
    shots = {home: 0, away: 0}
    xg = {home: 0.0, away: 0.0}
    # Pressure events in the press region — not part of canonical PPDA, but a
    # strong signal of pressing *effort* (work that doesn't force a turnover).
    pressures = {home: 0, away: 0}

    def opponent(t: str) -> str:
        return away if t == home else home

    for ev in events:
        team_name = (ev.get("team") or {}).get("name", "")
        if team_name not in (home, away):
            continue
        etype = (ev.get("type") or {}).get("name", "")
        loc = ev.get("location") or []
        x = float(loc[0]) if len(loc) >= 1 else None

        poss_events[team_name] += 1

        # ── Opponent passes → numerator for the *pressing* team ──────────────
        if etype == "Pass":
            # completed pass = no outcome key (StatsBomb marks only failures)
            completed = (ev.get("pass") or {}).get("outcome") is None
            if completed and x is not None:
                presser = opponent(team_name)
                # This pass is by `team_name` building up; in the presser's
                # frame the pass sits at x_press = 120 - x.
                x_press = PITCH_X - x
                if x_press >= PRESS_LINE:
                    opp_passes[presser][_zone(x_press)] += 1.0

        elif etype == "Shot":
            shots[team_name] += 1
            sx = (ev.get("shot") or {}).get("statsbomb_xg")
            if sx is not None:
                xg[team_name] += float(sx)

        # ── Defensive actions → denominator for the acting team ─────────────
        is_def_action = etype in DEF_ACTION_TYPES or _is_tackle(ev)
        if is_def_action and x is not None and x >= PRESS_LINE:
            def_actions[team_name][_zone(x)] += 1.0

        # ── Pressure events (effort signal, not in PPDA) ────────────────────
        if etype == "Pressure" and x is not None and x >= PRESS_LINE:
            pressures[team_name] += 1

    total_events = poss_events[home] + poss_events[away]

    docs = []
    for team in (home, away):
        opp = opponent(team)

        p_mid = opp_passes[team]["mid"]
        p_att = opp_passes[team]["att"]
        a_mid = def_actions[team]["mid"]
        a_att = def_actions[team]["att"]

        # Overall PPDA over the full press region (x >= 40)
        ppda_overall = _safe_ppda(p_mid + p_att, a_mid + a_att)
        ppda_mid = _safe_ppda(p_mid, a_mid)
        ppda_att = _safe_ppda(p_att, a_att)
        # Defensive-third PPDA is not part of the standard metric; report a
        # nominal high value so the fingerprint vector still has the slot.
        ppda_def = 50.0

        possession = (
            round(poss_events[team] / total_events * 100, 1) if total_events else 50.0
        )

        # press_intensity: blends inverse-PPDA (turnover-forcing press) with
        # raw pressure volume (pressing effort). 0–1, higher = more intense.
        #   inverse-PPDA term: 4.0 PPDA ≈ 1.0, 20+ PPDA ≈ ~0.2
        #   pressure term: ~150 pressures in press region ≈ 1.0
        ppda_term = min(1.0, 8.0 / max(ppda_overall, 1.0))
        pressure_term = min(1.0, pressures[team] / 150.0)
        press_intensity = round(0.7 * ppda_term + 0.3 * pressure_term, 3)

        style = (
            "aggressive high press"
            if ppda_overall < 8
            else "selective mid-block" if ppda_overall < 13 else "passive low block"
        )
        where = "attacking third" if ppda_att <= ppda_mid else "middle third"

        content = (
            f"{competition} {season} · {team} vs {opp} · pressing fingerprint\n"
            f"Style: {style}, pressing primarily in the {where}.\n"
            f"PPDA overall {ppda_overall} (middle third {ppda_mid}, "
            f"attacking third {ppda_att}). "
            f"Lower PPDA means a more aggressive press.\n"
            f"Possession {possession:.0f}%, {shots[team]} shots, "
            f"{xg[team]:.2f} xG, {pressures[team]} pressing actions. "
            f"Press intensity index {press_intensity:.2f}."
        )

        docs.append(
            {
                "properties": {
                    "content": content,
                    "team": team,
                    "opponent": opp,
                    "match_id": match_id,
                    "competition": competition,
                    "season": season,
                    "ppda": ppda_overall,
                    "ppda_def_third": ppda_def,
                    "ppda_mid_third": ppda_mid,
                    "ppda_att_third": ppda_att,
                    "possession": possession,
                    "press_intensity": press_intensity,
                },
                # Text the embedding model sees. Keep it descriptive so the
                # runtime descriptor (also natural language) matches well.
                "embed_text": content,
            }
        )

    return docs


async def _index_all() -> int:
    """Index every fingerprint without prompting; returns the number inserted."""
    wv = get_weaviate_client()
    if not wv.ready:
        log.error("Weaviate not ready — is the Docker container up on :8080?")
        return 0

    log.info("Loading all-MiniLM-L6-v2 embedding model...")
    model = await asyncio.to_thread(SentenceTransformer, "all-MiniLM-L6-v2")
    log.info("Embedding model ready")

    all_docs: list[dict] = []

    matches = []
    for sid in SEASON_IDS:
        try:
            ms = await asyncio.to_thread(load, f"matches/{COMPETITION_ID}/{sid}.json")
            matches.extend(ms)
            log.info(f"Season {sid}: {len(ms)} matches")
        except Exception as e:
            log.warning(f"Season {sid} failed: {e}")

    log.info(f"Processing up to {MAX_MATCHES} of {len(matches)} matches...")
    for i, m in enumerate(matches[:MAX_MATCHES]):
        match_id = str(m["match_id"])
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        season = str(m.get("season", {}).get("season_name", ""))
        comp = m.get("competition", {}).get("competition_name", "WC")
        try:
            events = await asyncio.to_thread(load, f"events/{match_id}.json")
            docs = await asyncio.to_thread(
                build_fingerprints, events, home, away, comp, season, match_id
            )
            all_docs.extend(docs)
            log.info(f"  [{i+1:2d}] {home} vs {away}: {len(docs)} fingerprints")
        except Exception as e:
            log.warning(f"  [{i+1:2d}] {home} vs {away}: failed — {e}")

    log.info(f"\nTotal fingerprints to index: {len(all_docs)}")
    if not all_docs:
        log.warning("Nothing to index.")
        return 0

    log.info("Embedding and inserting into TacticalProfiles...")
    inserted = await asyncio.to_thread(_embed_and_insert, wv, model, all_docs)
    log.info(f"\nDone — {wv.get_count(TACTICAL_PROFILES)} fingerprints in Weaviate")
    return inserted


def _embed_and_insert(wv, model, all_docs: list[dict]) -> int:
    """Blocking encode + REST inserts; run in a thread from _index_all."""
    inserted = 0
    batch = 50
    for i in range(0, len(all_docs), batch):
        chunk = all_docs[i : i + batch]
        vectors = model.encode(
            [d["embed_text"] for d in chunk],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for doc, vec in zip(chunk, vectors):
            ok = wv.insert_document(
                collection=TACTICAL_PROFILES,
                properties=doc["properties"],
                vector=vec.tolist(),
            )
            inserted += int(ok)
        log.info(f"  Inserted {inserted}/{len(all_docs)}")
    return inserted


async def ensure_indexed() -> None:
    """Index at startup if TacticalProfiles is empty. Never prompts or raises."""
    try:
        wv = await asyncio.to_thread(get_weaviate_client)
        if not await asyncio.to_thread(lambda: wv.ready):
            log.info(
                "Tactical auto-index: Weaviate not ready yet, skipping this attempt"
            )
            return
        existing = await asyncio.to_thread(wv.get_count, TACTICAL_PROFILES)
        if existing > 0:
            log.info(
                f"Tactical auto-index: {existing} fingerprints already indexed, skipping"
            )
            return
        log.info(
            "Tactical auto-index: TacticalProfiles is empty — indexing now (this can take a few minutes)..."
        )
        count = await _index_all()
        log.info(f"Tactical auto-index: complete — {count} fingerprints inserted")
    except Exception:
        log.warning(
            "Tactical auto-index failed — will retry on next app restart", exc_info=True
        )


async def main(check_only: bool = False) -> None:
    wv = get_weaviate_client()
    if not wv.ready:
        log.error("Weaviate not ready — is the Docker container up on :8080?")
        return

    if check_only:
        log.info(f"TacticalProfiles document count: {wv.get_count(TACTICAL_PROFILES)}")
        return

    existing = wv.get_count(TACTICAL_PROFILES)
    if existing > 0:
        ans = (
            input(f"TacticalProfiles already has {existing} docs. Re-index? (y/N): ")
            .strip()
            .lower()
        )
        if ans != "y":
            return

    await _index_all()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Report count only")
    args = parser.parse_args()
    asyncio.run(main(check_only=args.check))
