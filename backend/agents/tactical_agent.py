"""
Tactical-fingerprint matcher for the match dashboard's TacticalCard.

Live stats have no per-zone PPDA, so matching works in two steps:
    1. Retrieve candidate profiles for the team's style band with a
       team-agnostic descriptor (team names would bias BM25 toward that
       team's own history).
    2. Re-rank by possession closeness and report the gap in points.

No "% match" is reported: Weaviate's fusion score is a rank signal, not a
similarity. Returns None when Weaviate is empty or unavailable.
"""

import asyncio
import logging
from typing import List, Optional

from agents.weaviate_client import get_weaviate_client, TACTICAL_PROFILES
from api.schemas.schema import MatchState, TeamStats
from ml.embedding_model import get_embed_model as _get_embed_model
from ml.executors import EMBED_EXECUTOR

log = logging.getLogger(__name__)


def _describe(stats: TeamStats) -> str:
    """Descriptor in the indexer's fingerprint register.

    Infers a pseudo-PPDA band from live possession and shot volume.
    """
    poss = stats.possession if stats.possession > 0 else 50.0
    shots = stats.shots_total
    pass_acc = stats.pass_accuracy

    if poss >= 58 and pass_acc >= 82:
        style = "aggressive high press"
        band = "low PPDA"
    elif poss >= 48:
        style = "selective mid-block"
        band = "moderate PPDA"
    else:
        style = "passive low block"
        band = "high PPDA"

    return (
        f"pressing fingerprint\n"
        f"Style: {style} ({band}).\n"
        f"Possession {poss:.0f}%, {shots} shots, pass accuracy {pass_acc:.0f}%."
    )


async def match_team(
    state: MatchState,
    side: str,  # "home" | "away"
    loop: asyncio.AbstractEventLoop,
    top_k: int = 3,
    pool: int = 12,
) -> Optional[dict]:
    """Best historical fingerprint match for one team, or None."""
    wv = get_weaviate_client()
    if not wv.ready:
        return None

    if side == "home":
        team, opp, stats = state.home_name, state.away_name, state.home_stats
    else:
        team, opp, stats = state.away_name, state.home_name, state.away_stats

    descriptor = _describe(stats)

    model = _get_embed_model()
    vec: List[float] = await loop.run_in_executor(
        EMBED_EXECUTOR,
        lambda: model.encode(descriptor, normalize_embeddings=True).tolist(),
    )

    objs = await asyncio.to_thread(
        wv.hybrid_search,
        query_vector=vec,
        query_text=descriptor,
        top_k=pool,
        collection=TACTICAL_PROFILES,
        return_objects=True,
    )
    if not objs:
        return None

    live_poss = stats.possession if stats.possession > 0 else 50.0

    def gap(o: dict) -> float:
        return abs(float(o.get("possession") or 50.0) - live_poss)

    objs = sorted(objs, key=gap)[:top_k]
    best = objs[0]

    return {
        "team": team,
        "opponent": opp,
        "live_possession": round(stats.possession, 1),
        "match": {
            "team": best.get("team"),
            "opponent": best.get("opponent"),
            "competition": best.get("competition"),
            "season": best.get("season"),
            "possession_gap_pp": round(gap(best), 1),
            "ppda": best.get("ppda"),
            "ppda_mid_third": best.get("ppda_mid_third"),
            "ppda_att_third": best.get("ppda_att_third"),
            "possession": best.get("possession"),
            "press_intensity": best.get("press_intensity"),
            "content": best.get("content"),
        },
        "alternatives": [
            {
                "team": o.get("team"),
                "season": o.get("season"),
                "ppda": o.get("ppda"),
                "possession_gap_pp": round(gap(o), 1),
            }
            for o in objs[1:]
        ],
    }
