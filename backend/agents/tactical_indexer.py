"""
Tactical fingerprint indexer.

Builds one "pressing fingerprint" document per team per match from
StatsBomb WC open data and stores it in the Weaviate TacticalProfiles
collection. At runtime the TacticalCard matches a live team to these
profiles by numeric kNN (agents/tactical_agent.py) on possession,
ppda_full, pass_accuracy and shots.

PPDA (Passes allowed Per Defensive Action) is the standard
pressing-intensity metric; lower means a more aggressive press.

    PPDA = opponent completed passes / (tackles + interceptions + fouls)

`ppda` restricts both terms to the pressing team's attacking 60% of the
pitch (x >= 48 on the 120-long pitch), broken down by thirds so the
fingerprint captures where a team presses. `ppda_full` counts both terms
anywhere, matching live box-score stats. Possession is the team's share of
attempted passes, as in Opta/ESPN.

StatsBomb pitch: 120 long × 80 wide; x runs from the team's own goal (0) to
the opponent goal (120), in each team's own attacking direction.

Thirds (by x):
    defensive third : x <  40
    middle third    : 40 <= x < 80
    attacking third : x >= 80

Run from backend/:
    set PYTHONPATH=.
    python agents/tactical_indexer.py            # index
    python agents/tactical_indexer.py --check    # report count only
    python agents/tactical_indexer.py --rebuild  # drop the collection, re-index
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter

from agents.weaviate_client import TACTICAL_PROFILES, get_weaviate_client
from ml.embedding_model import get_embed_model
from ml.statsbomb import COMPETITION_ID, SEASON_IDS, load

log = logging.getLogger(__name__)

MAX_MATCHES = 500

# defensive action types counted in the PPDA denominator
# StatsBomb: a Duel of sub-type Tackle, plus Interception / Foul Committed
DEF_ACTION_TYPES = {"Interception", "Foul Committed"}

PITCH_X = 120.0
PRESS_LINE = 48.0  # attacking 60% of 120 (Trainor)
ATT_LINE = 80.0


def _zone(x: float) -> str:
    """Third of a press-region x (always >= PRESS_LINE)."""
    return "att" if x >= ATT_LINE else "mid"


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


def _tally(events: list, home: str, away: str) -> dict[str, Counter]:
    """Per-team counts. The team's own: passes, passes_done, shots, xg,
    pressures (press region), def_actions (anywhere), def_mid / def_att
    (press region). The opponent completed passes the team faced: opp_passes
    (anywhere), opp_mid / opp_att (press region, in the team's frame)."""
    tally = {home: Counter(), away: Counter()}
    for ev in events:
        team = (ev.get("team") or {}).get("name", "")
        if team not in (home, away):
            continue
        t, opp = tally[team], tally[away if team == home else home]
        etype = (ev.get("type") or {}).get("name", "")
        loc = ev.get("location") or []
        x = float(loc[0]) if len(loc) >= 1 else None

        if etype == "Pass":
            # completed pass = no outcome key (StatsBomb marks only failures)
            completed = (ev.get("pass") or {}).get("outcome") is None
            t["passes"] += 1
            t["passes_done"] += completed
            if completed:
                # opponent passes: numerator for the pressing team
                opp["opp_passes"] += 1
                if x is not None and PITCH_X - x >= PRESS_LINE:
                    opp["opp_" + _zone(PITCH_X - x)] += 1
        elif etype == "Shot":
            t["shots"] += 1
            sx = (ev.get("shot") or {}).get("statsbomb_xg")
            if sx is not None:
                t["xg"] += float(sx)

        # defensive actions: denominator for the acting team
        is_def_action = etype in DEF_ACTION_TYPES or _is_tackle(ev)
        t["def_actions"] += is_def_action
        if is_def_action and x is not None and x >= PRESS_LINE:
            t["def_" + _zone(x)] += 1

        # pressure events (effort signal, not in PPDA)
        if etype == "Pressure" and x is not None and x >= PRESS_LINE:
            t["pressures"] += 1
    return tally


def _fingerprint(
    team: str, opp: str, t: Counter, total_passes: int,
    competition: str, season: str, match_id: str,
) -> dict:
    """TacticalProfiles properties for one team from its tally."""
    ppda_overall = _safe_ppda(t["opp_mid"] + t["opp_att"], t["def_mid"] + t["def_att"])
    ppda_mid = _safe_ppda(t["opp_mid"], t["def_mid"])
    ppda_att = _safe_ppda(t["opp_att"], t["def_att"])
    possession = round(t["passes"] / total_passes * 100, 1) if total_passes else 50.0
    ppda_full = _safe_ppda(t["opp_passes"], t["def_actions"])
    pass_accuracy = round(t["passes_done"] / t["passes"] * 100, 1) if t["passes"] else 0.0

    # press_intensity: inverse PPDA (turnover-forcing press) blended with raw
    # pressure volume; 0-1, higher = more intense
    #   inverse-PPDA term: 4.0 PPDA ≈ 1.0, 20+ PPDA ≈ ~0.2
    #   pressure term: ~150 pressures in press region ≈ 1.0
    ppda_term = min(1.0, 8.0 / max(ppda_overall, 1.0))
    pressure_term = min(1.0, t["pressures"] / 150.0)
    press_intensity = round(0.7 * ppda_term + 0.3 * pressure_term, 3)

    style = (
        "aggressive high press"
        if ppda_overall < 8
        else "selective mid-block" if ppda_overall < 13 else "passive low block"
    )
    where = "attacking third" if ppda_att <= ppda_mid else "middle third"

    # text the embedding model sees; descriptive, to match the runtime
    # descriptor (also natural language)
    content = (
        f"{competition} {season} · {team} vs {opp} · pressing fingerprint\n"
        f"Style: {style}, pressing primarily in the {where}.\n"
        f"PPDA overall {ppda_overall} (middle third {ppda_mid}, "
        f"attacking third {ppda_att}). "
        f"Lower PPDA means a more aggressive press.\n"
        f"Possession {possession:.0f}%, pass accuracy {pass_accuracy:.0f}%, "
        f"{t['shots']} shots, "
        f"{t['xg']:.2f} xG, {t['pressures']} pressing actions. "
        f"Press intensity index {press_intensity:.2f}."
    )
    return {
        "content": content,
        "team": team,
        "opponent": opp,
        "match_id": match_id,
        "competition": competition,
        "season": season,
        "ppda": ppda_overall,
        "ppda_full": ppda_full,
        "pass_accuracy": pass_accuracy,
        "shots": t["shots"],
        # defensive-third PPDA is not in the standard metric; a nominal high value fills
        # the fingerprint slot
        "ppda_def_third": 50.0,
        "ppda_mid_third": ppda_mid,
        "ppda_att_third": ppda_att,
        "possession": possession,
        "press_intensity": press_intensity,
    }


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
        denominator  T's defensive actions at T-frame x >= PRESS_LINE
    `ppda_full` uses the same terms over the whole pitch.
    """
    tally = _tally(events, home, away)
    total_passes = tally[home]["passes"] + tally[away]["passes"]
    return [
        _fingerprint(team, opp, tally[team], total_passes, competition, season, match_id)
        for team, opp in ((home, away), (away, home))
    ]


async def _index_all() -> int:
    """Index every fingerprint without prompting; returns the number inserted."""
    wv = get_weaviate_client()
    if not wv.ready:
        log.error("Weaviate not ready — is the Docker container up on :8080?")
        return 0

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
    inserted = await asyncio.to_thread(_embed_and_insert, wv, all_docs)
    log.info(f"\nDone — {wv.get_count(TACTICAL_PROFILES)} fingerprints in Weaviate")
    return inserted


def _embed_and_insert(wv, all_docs: list[dict]) -> int:
    """Blocking encode + batch insert; run in a thread from _index_all."""
    vectors = get_embed_model().encode(
        [d["content"] for d in all_docs], normalize_embeddings=True, show_progress_bar=False
    )
    return wv.insert_many(TACTICAL_PROFILES, all_docs, vectors)


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


async def main(check_only: bool = False, rebuild: bool = False) -> None:
    wv = get_weaviate_client()
    if not wv.ready:
        log.error("Weaviate not ready — is the Docker container up on :8080?")
        return

    if check_only:
        log.info(f"TacticalProfiles document count: {wv.get_count(TACTICAL_PROFILES)}")
        return

    if rebuild:
        wv.recreate(TACTICAL_PROFILES)
        await _index_all()
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
    parser.add_argument("--rebuild", action="store_true", help="Drop the collection and re-index")
    args = parser.parse_args()
    asyncio.run(main(check_only=args.check, rebuild=args.rebuild))
