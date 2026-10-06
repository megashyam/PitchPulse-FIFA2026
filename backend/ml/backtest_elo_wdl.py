"""
Calibration backtest for the Elo → W/D/L model in ml/prior_builder.py.

Checks ``elo_to_wdl`` (symmetric ordered logit fitted by ml/fit_wdl_prior.py)
on StatsBomb open data, independent of the 2026 config Elos.

Method:
    1. Load every men's World Cup match in StatsBomb open data (2018, 2022)
       in chronological order.
    2. Keep an online Elo per team from 1500. Before each match, predict
       W/D/L with ``elo_to_wdl``; then apply a goal-difference-aware update.
    3. Score predictions with multiclass log loss, Brier (with its Murphy
       decomposition), ranked probability score and a reliability table,
       against the empirical H/D/A base rate.
    4. Report overall and on the back half, after ratings warm up.

Labels are the 90-minute result (knockout scores rebuilt from period 1-2
goals). HOME_ADV goes only to the host nation. Differences from the base
rate carry paired bootstrap 95% CIs.

Run from backend/:
    PYTHONPATH=. python ml/backtest_elo_wdl.py
    PYTHONPATH=. python ml/backtest_elo_wdl.py --json report.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import httpx
import numpy as np

from ml import elo_ratings as er
from ml.metrics import (
    brier,
    log_loss,
    murphy,
    paired_bootstrap,
    per_match_brier,
    per_match_ll,
    per_match_rps,
    rps,
)
from ml.prior_builder import elo_to_wdl
from ml.statsbomb import COMPETITION_ID, SEASON_IDS, goals_in_regulation, load

ELO_START = 1500.0
ELO_K = 40.0  # update step
HOME_ADV = 60.0  # Elo points, host nation only (venues are otherwise neutral)
HOSTS = {"Russia", "Qatar"}  # WC 2018, WC 2022
N_BOOT = 5000  # paired_bootstrap default


def load_matches() -> List[dict]:
    """Every WC 2018/2022 match, chronological, with 90-minute `hs90`/`as90`."""
    matches: List[dict] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for sid in SEASON_IDS:
            for m in load(f"matches/{COMPETITION_ID}/{sid}.json", client):
                if m.get("home_score") is None or m.get("away_score") is None:
                    continue
                stage = (m.get("competition_stage") or {}).get("name", "")
                if stage == "Group Stage":
                    m["hs90"], m["as90"] = int(m["home_score"]), int(m["away_score"])
                else:
                    events = load(f"events/{m['match_id']}.json", client)
                    m["hs90"], m["as90"] = goals_in_regulation(
                        events,
                        m["home_team"]["home_team_name"],
                        m["away_team"]["away_team_name"],
                    )
                matches.append(m)
    matches.sort(key=lambda m: (m.get("match_date", ""), m.get("kick_off", "")))
    return matches


def _adv(team: str) -> float:
    return HOME_ADV if team in HOSTS else 0.0





def update_elo(elo: Dict[str, float], home: str, away: str, hs: int, as_: int) -> None:
    ra = elo.get(home, ELO_START) + _adv(home)
    rb = elo.get(away, ELO_START) + _adv(away)
    exp_home = er.expected(ra - rb)
    if hs > as_:
        score_home = 1.0
    elif hs == as_:
        score_home = 0.5
    else:
        score_home = 0.0
    mult = er.goal_mult(hs - as_)
    delta = ELO_K * mult * (score_home - exp_home)
    elo[home] = elo.get(home, ELO_START) + delta
    elo[away] = elo.get(away, ELO_START) - delta


def outcome_index(hs: int, as_: int) -> int:
    """0 = home win, 1 = draw, 2 = away win (matches elo_to_wdl order)."""
    if hs > as_:
        return 0
    if hs == as_:
        return 1
    return 2


def base_rate(actuals: List[int]) -> Tuple[float, float, float]:
    n = len(actuals)
    c = [actuals.count(0), actuals.count(1), actuals.count(2)]
    return (c[0] / n, c[1] / n, c[2] / n)


def reliability_table(
    preds: List[Tuple[float, float, float]],
    actuals: List[int],
    n_bins: int = 10,
) -> List[dict]:
    """Predicted home-win probability buckets vs realised home-win frequency."""
    buckets: List[dict] = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "pred_sum": 0.0, "hits": 0}
        for i in range(n_bins)
    ]
    for p, y in zip(preds, actuals):
        ph = min(0.999999, max(0.0, p[0]))
        b = int(ph * n_bins)
        buckets[b]["n"] += 1
        buckets[b]["pred_sum"] += ph
        if y == 0:
            buckets[b]["hits"] += 1
    out = []
    for b in buckets:
        if b["n"] == 0:
            continue
        out.append(
            {
                "range": f"{b['lo']:.1f}-{b['hi']:.1f}",
                "n": b["n"],
                "mean_pred": round(b["pred_sum"] / b["n"], 3),
                "emp_freq": round(b["hits"] / b["n"], 3),
            }
        )
    return out


def run_backtest() -> dict:
    matches = load_matches()
    elo: Dict[str, float] = {}

    preds: List[Tuple[float, float, float]] = []
    actuals: List[int] = []

    changed = 0
    for m in matches:
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        hs, as_ = m["hs90"], m["as90"]
        changed += outcome_index(hs, as_) != outcome_index(int(m["home_score"]), int(m["away_score"]))

        ra = elo.get(home, ELO_START) + _adv(home)
        rb = elo.get(away, ELO_START) + _adv(away)
        preds.append(elo_to_wdl(ra, rb))
        actuals.append(outcome_index(hs, as_))

        update_elo(elo, home, away, hs, as_)

    n = len(preds)
    half = n // 2

    base = base_rate(actuals)
    base_preds = [base] * n

    def block(p, a) -> dict:
        return {
            "n": len(p),
            "log_loss": round(log_loss(p, a), 4),
            "brier": round(brier(p, a), 4),
            "rps": round(rps(p, a), 4),
            "murphy": murphy(p, a),
        }

    def delta(p, a) -> dict:
        """Elo minus base rate, per match; negative = Elo better."""
        b = [base] * len(a)
        dll = per_match_ll(p, a) - per_match_ll(b, a)
        dbr = per_match_brier(p, a) - per_match_brier(b, a)
        drps = per_match_rps(p, a) - per_match_rps(b, a)
        return {
            "log_loss": round(float(dll.mean()), 4),
            "log_loss_ci95": paired_bootstrap(dll),
            "brier": round(float(dbr.mean()), 4),
            "brier_ci95": paired_bootstrap(dbr),
            "rps": round(float(drps.mean()), 4),
            "rps_ci95": paired_bootstrap(drps),
        }

    report = {
        "n_matches": n,
        "seasons": SEASON_IDS,
        "labels": "90-minute result (knockout scores rebuilt from period 1-2 goals)",
        "labels_changed_by_90min": changed,
        "home_advantage": f"{HOME_ADV:.0f} Elo for the host nation only ({', '.join(sorted(HOSTS))})",
        "ci": f"paired bootstrap over matches, {N_BOOT} resamples, 95% percentile",
        "predicted_vs_realised_draw": {
            "mean_predicted": round(float(np.mean([p[1] for p in preds])), 3),
            "realised": round(actuals.count(1) / n, 3),
        },
        "outcome_base_rates": {
            "home_win": round(base[0], 3),
            "draw": round(base[1], 3),
            "away_win": round(base[2], 3),
        },
        "overall": {
            "elo": block(preds, actuals),
            "baseline": block(base_preds, actuals),
            "elo_minus_baseline": delta(preds, actuals),
        },
        "warmed_up_second_half": {
            "elo": block(preds[half:], actuals[half:]),
            "baseline": block([base] * (n - half), actuals[half:]),
            "elo_minus_baseline": delta(preds[half:], actuals[half:]),
        },
        "baseline_note": (
            "The base rate is fitted in-sample on the same matches, which favours "
            "the baseline; Elo is scored prequentially."
        ),
        "reliability_second_half": reliability_table(preds[half:], actuals[half:]),
    }
    return report


def _print(report: dict) -> None:
    print(
        f"\nElo→WDL calibration backtest — {report['n_matches']} WC matches "
        f"(seasons {report['seasons']})"
    )
    br = report["outcome_base_rates"]
    print(
        f"Base rates:  home {br['home_win']}  draw {br['draw']}  away {br['away_win']}"
    )

    for label, key in [
        ("Overall", "overall"),
        ("Second half (warmed up)", "warmed_up_second_half"),
    ]:
        e = report[key]["elo"]
        b = report[key]["baseline"]
        print(f"\n{label}  (n={e['n']})")
        print(
            f"  log loss   Elo {e['log_loss']:.4f}   baseline {b['log_loss']:.4f}   "
            f"{'below' if e['log_loss'] < b['log_loss'] else 'above'} baseline"
        )
        print(f"  Brier      Elo {e['brier']:.4f}   baseline {b['brier']:.4f}")
        print(f"  RPS        Elo {e['rps']:.4f}   baseline {b['rps']:.4f}")
        d = report[key]["elo_minus_baseline"]
        print(
            f"  Elo - baseline: log loss {d['log_loss']:+.4f} CI {d['log_loss_ci95']}, "
            f"Brier {d['brier']:+.4f} CI {d['brier_ci95']}, RPS {d['rps']:+.4f} CI {d['rps_ci95']}"
        )

    print("\nReliability (second half) — predicted home-win prob vs realised:")
    print(f"  {'bucket':>10} {'n':>4} {'mean_pred':>10} {'emp_freq':>9}")
    for row in report["reliability_second_half"]:
        print(
            f"  {row['range']:>10} {row['n']:>4} {row['mean_pred']:>10} {row['emp_freq']:>9}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the full report to this path")
    args = ap.parse_args()

    report = run_backtest()
    _print(report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
