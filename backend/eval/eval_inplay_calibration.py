"""Backtest the Poisson in-play model against real World Cup outcomes.

For each historical match, reconstruct the true score and red-card state at
fixed checkpoints (0', 15', 45', 75'), feed it to `inplay_wdl` with an
Elo-warmed pre-match prior, and score the produced W/D/L distribution against
the actual full-time result. Reports log-loss and Brier per checkpoint, versus
two baselines: the static pre-match prior and the outcome base rate.

Minute 0 is a sanity check: at 0-0 kickoff the model is fitted to reproduce
the prior, so its scores must equal the pre-match prior's. After that, a
well-behaved in-play model should beat the prior by a growing margin as the
match progresses (later state = more information).

Uses StatsBomb Open Data (matches + events, disk-cached). By default every
match after the Elo warm-up half is scored; bound it with --max-matches for a
quick run. Prior, labels and Elo updates follow backtest_elo_wdl: host-only
home advantage, 90-minute results. Differences from the prior carry paired
bootstrap 95% CIs.

Run from backend/:
    PYTHONPATH=. python eval/eval_inplay_calibration.py --json inplay_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import httpx

from ml.backtest_elo_wdl import (
    ELO_START,
    _adv,
    load_matches,
    outcome_index,
    paired_bootstrap,
    per_match_brier,
    per_match_ll,
    update_elo,
)
from ml.in_play import inplay_wdl
from ml.prior_builder import elo_to_wdl
from ml.statsbomb import card_from_event, load

CHECKPOINTS = [0, 15, 45, 75]
EPS = 1e-12


def match_timeline(events: List[dict], home: str, away: str):
    """Extract (minute, is_home) goal list and red-card minutes per side."""
    goals: List[Tuple[int, bool]] = []
    reds: List[Tuple[int, bool]] = []
    for ev in events:
        team = ev.get("team", {}).get("name", "")
        if team not in (home, away):
            continue
        is_home = team == home
        minute = int(ev.get("minute", 0))
        etype = (ev.get("type") or {}).get("name", "")
        if (
            etype == "Shot"
            and (ev.get("shot", {}).get("outcome", {}) or {}).get("name") == "Goal"
        ):
            goals.append((minute, is_home))
        elif etype == "Own Goal For":
            goals.append((minute, is_home))
        if card_from_event(ev) == "red":
            reds.append((minute, is_home))
    return goals, reds


def state_at(goals, reds, minute: int) -> Tuple[int, int, int, int]:
    hs = sum(1 for m, h in goals if m <= minute and h)
    as_ = sum(1 for m, h in goals if m <= minute and not h)
    rh = sum(1 for m, h in reds if m <= minute and h)
    ra = sum(1 for m, h in reds if m <= minute and not h)
    return hs, as_, rh, ra


def log_loss(preds, actuals) -> float:
    return sum(-math.log(max(p[y], EPS)) for p, y in zip(preds, actuals)) / len(preds)


def brier(preds, actuals) -> float:
    total = 0.0
    for p, y in zip(preds, actuals):
        t = [0.0, 0.0, 0.0]
        t[y] = 1.0
        total += sum((p[k] - t[k]) ** 2 for k in range(3))
    return total / len(preds)


def run(max_matches: int | None = None) -> dict:
    matches = load_matches()
    elo: Dict[str, float] = {}

    # warm Elo on the first half (same protocol as backtest_elo_wdl)
    half = len(matches) // 2
    for m in matches[:half]:
        update_elo(
            elo,
            m["home_team"]["home_team_name"],
            m["away_team"]["away_team_name"],
            m["hs90"],
            m["as90"],
        )

    eval_matches = matches[half:]
    if max_matches is not None:
        eval_matches = eval_matches[:max_matches]

    preds = {cp: [] for cp in CHECKPOINTS}
    prior_preds: List[Tuple[float, float, float]] = []
    actuals: List[int] = []

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for m in eval_matches:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            hs90, as90 = m["hs90"], m["as90"]

            events = load(f"events/{m['match_id']}.json", client)
            goals, reds = match_timeline(events, home, away)
            pre = elo_to_wdl(
                elo.get(home, ELO_START) + _adv(home),
                elo.get(away, ELO_START) + _adv(away),
            )
            prior_preds.append(pre)
            actuals.append(outcome_index(hs90, as90))

            for cp in CHECKPOINTS:
                hs, as_, rh, ra = state_at(goals, reds, cp)
                preds[cp].append(inplay_wdl(pre, cp, hs, as_, rh, ra))

            update_elo(elo, home, away, hs90, as90)  # keep warming

    n = len(actuals)
    if n == 0:
        raise SystemExit("No matches evaluated")
    base = (
        sum(1 for a in actuals if a == 0) / n,
        sum(1 for a in actuals if a == 1) / n,
        sum(1 for a in actuals if a == 2) / n,
    )
    def minus_prior(p) -> dict:
        d_ll = per_match_ll(p, actuals) - per_match_ll(prior_preds, actuals)
        d_br = per_match_brier(p, actuals) - per_match_brier(prior_preds, actuals)
        return {
            "log_loss": round(float(d_ll.mean()), 4),
            "log_loss_ci95": paired_bootstrap(d_ll),
            "brier": round(float(d_br.mean()), 4),
            "brier_ci95": paired_bootstrap(d_br),
        }

    report = {
        "data": "StatsBomb Open Data (real FIFA World Cup matches)",
        "labels": "90-minute result (knockout scores rebuilt from period 1-2 goals)",
        "home_advantage": "host nation only, as in backtest_elo_wdl",
        "ci": "paired bootstrap over matches, 95% percentile",
        "n_matches": n,
        "baselines": {
            "pre_match_prior": {
                "log_loss": round(log_loss(prior_preds, actuals), 4),
                "brier": round(brier(prior_preds, actuals), 4),
            },
            "base_rate": {
                "log_loss": round(log_loss([base] * n, actuals), 4),
                "brier": round(brier([base] * n, actuals), 4),
            },
        },
        "in_play_by_checkpoint": {
            f"minute_{cp}": {
                "log_loss": round(log_loss(preds[cp], actuals), 4),
                "brier": round(brier(preds[cp], actuals), 4),
            }
            for cp in CHECKPOINTS
        },
        "in_play_minus_prior": {f"minute_{cp}": minus_prior(preds[cp]) for cp in CHECKPOINTS},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="cap on matches scored after the warm-up (default: all)",
    )
    ap.add_argument("--json", help="write report to this path")
    args = ap.parse_args()

    report = run(args.max_matches)

    print(f"\nIn-play calibration — {report['n_matches']} warmed-up WC matches")
    pm = report["baselines"]["pre_match_prior"]
    br = report["baselines"]["base_rate"]
    print(f"  pre-match prior : log-loss {pm['log_loss']}  brier {pm['brier']}")
    print(f"  base rate       : log-loss {br['log_loss']}  brier {br['brier']}")
    for cp, m in report["in_play_by_checkpoint"].items():
        d = report["in_play_minus_prior"][cp]
        print(f"  in-play @{cp:>10}: log-loss {m['log_loss']}  brier {m['brier']}"
              f"  (vs prior {d['log_loss']:+} {d['log_loss_ci95']})")
    print(
        "\nExpected: minute 0 equals the pre-match prior; later checkpoints "
        "should fall monotonically and undercut it."
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
