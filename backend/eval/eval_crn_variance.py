"""Empirically verify the Common Random Numbers claim in the counterfactual
engine.

Question: for a fixed Elo perturbation, is the Monte Carlo variance of the
estimated championship-probability delta smaller when the baseline and
counterfactual simulations SHARE a seed than when they use independent seeds?

Offline: Elo priors only, no odds, no network.

Run from backend/:
    PYTHONPATH=. python eval/eval_crn_variance.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ml.tournament_sim import run_simulation
from ml.wc_2026_config import WC2026_TEAMS


def measure(team: str, elo_boost: float, n_sims: int, repeats: int) -> dict:
    deltas_crn: list[float] = []
    deltas_ind: list[float] = []
    shifts_crn: list[float] = []
    shifts_ind: list[float] = []

    def champ(res, name):
        return res.team(name).probs["champion"]

    def path_shift(before, after):
        return min(
            1.0,
            sum(
                abs(a.probs["champion"] - b.probs["champion"])
                for a, b in zip(after.teams, before.teams)
            )
            / 2.0,
        )

    for k in range(repeats):
        # --- CRN: identical seed for both legs
        seed = 10_000 + k
        base = run_simulation(n_sims=n_sims, seed=seed)
        cf = run_simulation(n_sims=n_sims, seed=seed, elo_overrides={team: elo_boost})
        deltas_crn.append(champ(cf, team) - champ(base, team))
        shifts_crn.append(path_shift(base, cf))

        # --- independent: two different seeds
        base_i = run_simulation(n_sims=n_sims, seed=20_000 + 2 * k)
        cf_i = run_simulation(
            n_sims=n_sims, seed=20_001 + 2 * k, elo_overrides={team: elo_boost}
        )
        deltas_ind.append(champ(cf_i, team) - champ(base_i, team))
        shifts_ind.append(path_shift(base_i, cf_i))

    def stats(xs):
        a = np.asarray(xs)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    s_crn, s_ind = stats(deltas_crn), stats(deltas_ind)
    var_ratio = (s_ind["std"] ** 2) / max(s_crn["std"] ** 2, 1e-18)
    return {
        "team": team,
        "elo_boost": elo_boost,
        "n_sims_per_leg": n_sims,
        "repeats": repeats,
        "delta_champion_prob": {"crn_shared_seed": s_crn, "independent_seeds": s_ind},
        "path_shift": {
            "crn_shared_seed": stats(shifts_crn),
            "independent_seeds": stats(shifts_ind),
        },
        "variance_reduction_factor": round(var_ratio, 1),
        "equivalent_extra_sims": (
            f"independent seeding needs ~{var_ratio:.0f}x more simulations "
            f"for the same delta precision"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", default=WC2026_TEAMS[0].name)
    ap.add_argument("--boost", type=float, default=30.0)
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--json", help="write report to this path")
    args = ap.parse_args()

    report = measure(args.team, args.boost, args.n_sims, args.repeats)

    c = report["delta_champion_prob"]["crn_shared_seed"]
    i = report["delta_champion_prob"]["independent_seeds"]
    print(
        f"\nCRN variance audit — {args.team} +{args.boost} Elo, "
        f"{args.n_sims} sims/leg, {args.repeats} repeats"
    )
    print(f"  Δ champion prob (CRN):         mean={c['mean']:+.5f}  std={c['std']:.5f}")
    print(f"  Δ champion prob (independent): mean={i['mean']:+.5f}  std={i['std']:.5f}")
    print(f"  variance reduction factor:     {report['variance_reduction_factor']}x")
    print(f"  → {report['equivalent_extra_sims']}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
