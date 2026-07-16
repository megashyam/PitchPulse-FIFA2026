"""
Runtime tactical fingerprint retrieval for the Match dashboard.

Converts live match statistics (possession, shots, pass accuracy) into a
natural-language tactical descriptor, embeds the descriptor, and performs
vector similarity search against historical tactical profiles stored in
Weaviate.

The matched historical fingerprint provides contextual insights such as
similar playing styles, PPDA metrics, possession patterns, and pressing
profiles.

If tactical profiles are unavailable or Weaviate is not ready, the agent
returns None and the dashboard falls back to live statistical indicators.
"""

import asyncio
import logging
from typing import List, Optional

from agents.weaviate_client import get_weaviate_client, TACTICAL_PROFILES
from api.schemas.schema import MatchState, TeamStats
from ml.embedding_model import get_embed_model as _get_embed_model
from ml.executors import EMBED_EXECUTOR

log = logging.getLogger(__name__)


def _describe(team: str, opp: str, stats: TeamStats, minute: int) -> str:
    """
    Build a descriptor in the SAME register as the indexer's fingerprint text,
    so embeddings land in a comparable region of vector space.

    We infer a pseudo-PPDA band from live possession + shot volume: dominant,
    high-possession teams with shot pressure read as a high press; deep,
    low-possession teams read as a low block. This is a proxy — the whole point
    of the match is to map it onto a real historical fingerprint.
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
        f"{team} vs {opp} · pressing fingerprint\n"
        f"Style: {style} ({band}).\n"
        f"Possession {poss:.0f}%, {shots} shots, pass accuracy {pass_acc:.0f}% "
        f"at minute {minute}. Territorial control reflects pressing approach."
    )


async def match_team(
    state: MatchState,
    side: str,
    loop: asyncio.AbstractEventLoop,
    top_k: int = 3,
) -> Optional[dict]:
    """
    Returns the best historical fingerprint match for one team, or None.
    """
    wv = get_weaviate_client()
    if not wv.ready:
        return None

    if side == "home":
        team, opp, stats = state.home_name, state.away_name, state.home_stats
    else:
        team, opp, stats = state.away_name, state.home_name, state.away_stats

    minute = state.elapsed or 0
    descriptor = _describe(team, opp, stats, minute)

    model = _get_embed_model()
    vec: List[float] = await loop.run_in_executor(
        EMBED_EXECUTOR,
        lambda: model.encode(descriptor, normalize_embeddings=True).tolist(),
    )

    objs = wv.hybrid_search(
        query_vector=vec,
        query_text=descriptor,
        top_k=top_k,
        collection=TACTICAL_PROFILES,
        return_objects=True,
    )
    if not objs:
        return None

    best = objs[0]

    raw = best.get("_score")
    match_pct = round(min(1.0, max(0.0, float(raw))) * 100) if raw is not None else None

    return {
        "team": team,
        "opponent": opp,
        "live_possession": round(stats.possession, 1),
        "match": {
            "team": best.get("team"),
            "opponent": best.get("opponent"),
            "competition": best.get("competition"),
            "season": best.get("season"),
            "match_pct": match_pct,
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
                "match_pct": (
                    round(min(1.0, max(0.0, float(o["_score"]))) * 100)
                    if o.get("_score") is not None
                    else None
                ),
            }
            for o in objs[1:]
        ],
    }
