"""
Fit and validate the commentary xG model (ml/shot_xg.py).

    Train     ESPN summaries for --leagues in --year (80% of matches, split
              by match, seeded)
    Held-out  remaining 20%: log loss / Brier vs a constant base rate
    External  WC 2022 via ESPN: per team-match xG vs StatsBomb xG
              (Pearson r, calibration)
    WC 2026   --wc26-raw: xG sum vs goals on the other coordinate convention

Outputs ml/shot_xg_coef.json and ml/shot_xg_report.json.

Usage:
    python -m ml.fit_shot_xg --cache <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import httpx
import numpy as np

from ml import shot_xg
from ml.statsbomb import SB_BASE
from ml.team_names import canonical

SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WEB = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
REPORT_PATH = Path(__file__).with_name("shot_xg_report.json")


def _fetch_league(client: httpx.Client, cache: Path, league: str, year: str) -> list[dict]:
    d = cache / league
    d.mkdir(parents=True, exist_ok=True)
    sb = client.get(f"{SITE}/{league}/scoreboard", params={"dates": year, "limit": 1000}).json()
    out = []
    for e in sb.get("events", []):
        if not e["status"]["type"]["completed"]:
            continue
        p = d / f"{e['id']}.json"
        if not p.exists():
            r = client.get(f"{WEB}/{league}/summary", params={"event": e["id"]})
            r.raise_for_status()
            s = r.json()
            p.write_text(
                json.dumps({"commentary": s.get("commentary", []), "header": s.get("header")}),
                encoding="utf-8",
            )
            time.sleep(0.15)
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _shots(summary: dict) -> list[shot_xg.Shot]:
    return [
        s
        for c in summary.get("commentary", [])
        if (s := shot_xg.shot_from_play(c.get("play") or {})) is not None
    ]


def _design(shots: list[shot_xg.Shot]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[shot_xg.features(s)[k] for k in shot_xg.FEATURES] for s in shots])
    y = np.array([s.outcome == "goal" for s in shots], dtype=float)
    return X, y


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50):
    """Newton-Raphson with a light ridge on standardised features."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + reg * w * len(y)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg * len(y))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    coef = w[1:] / sd
    intercept = w[0] - float((w[1:] * mu / sd).sum())
    return float(intercept), coef


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _statsbomb_wc2022(client: httpx.Client, cache: Path) -> dict[tuple[str, str], float]:
    """(date, canonical team) → StatsBomb xG total (non-shootout, excl. own goals)."""
    d = cache / "statsbomb_wc2022"
    d.mkdir(parents=True, exist_ok=True)
    matches = client.get(f"{SB_BASE}/matches/43/106.json").json()
    out: dict[tuple[str, str], float] = {}
    for m in matches:
        p = d / f"{m['match_id']}.json"
        if not p.exists():
            p.write_text(client.get(f"{SB_BASE}/events/{m['match_id']}.json").text, encoding="utf-8")
        tot: dict[str, float] = {}
        for ev in json.loads(p.read_text(encoding="utf-8")):
            if (ev.get("type") or {}).get("name") != "Shot" or ev.get("period", 1) >= 5:
                continue
            t = canonical(ev["team"]["name"])
            tot[t] = tot.get(t, 0.0) + float(ev["shot"].get("statsbomb_xg") or 0.0)
        for t, v in tot.items():
            out[(m["match_date"], t)] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--year", default="2025")
    ap.add_argument("--leagues", default="eng.1,esp.1,ger.1,ita.1,fra.1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wc26-raw", help="dir of raw WC 2026 ESPN summaries (calibration check)")
    a = ap.parse_args()
    cache = Path(a.cache)
    client = httpx.Client(timeout=60)  # default UA: ESPN's CDN 403s custom ones

    matches: list[list[shot_xg.Shot]] = []
    for lg in a.leagues.split(","):
        ms = _fetch_league(client, cache, lg, a.year)
        matches += [_shots(m) for m in ms]
        print(f"{lg}: {len(ms)} matches", flush=True)

    rng = random.Random(a.seed)
    idx = list(range(len(matches)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    train = [s for i in idx[:cut] for s in matches[i]]
    test = [s for i in idx[cut:] for s in matches[i]]

    pens = [s for s in train if s.penalty]
    penalty_xg = sum(s.outcome == "goal" for s in pens) / max(1, len(pens))
    tr = [s for s in train if not s.penalty]
    Xtr, ytr = _design(tr)
    intercept, coef = _fit_logistic(Xtr, ytr)
    model = {
        "intercept": intercept,
        "coef": dict(zip(shot_xg.FEATURES, map(float, coef))),
        "penalty_xg": round(penalty_xg, 4),
        "trained_on": f"ESPN {a.leagues} {a.year}",
        "n_shots": len(tr),
        "n_penalties": len(pens),
    }
    shot_xg.COEF_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    shot_xg._COEF = None

    te = [s for s in test if not s.penalty]
    Xte, yte = _design(te)
    p = np.array([shot_xg.xg(s) for s in te])
    base = np.full_like(yte, ytr.mean())
    held_out = {
        "n_shots": len(te),
        "goals": int(yte.sum()),
        "xg_sum": round(float(p.sum()), 1),
        "logloss": round(_logloss(p, yte), 4),
        "logloss_base_rate": round(_logloss(base, yte), 4),
        "brier": round(float(((p - yte) ** 2).mean()), 4),
        "brier_base_rate": round(float(((base - yte) ** 2).mean()), 4),
    }

    # External: WC 2022 vs StatsBomb.
    wc = _fetch_league(client, cache, "fifa.world", "2022")
    sb = _statsbomb_wc2022(client, cache)
    pairs, goals, ours = [], 0, 0.0
    for m in wc:
        comp = (m.get("header") or {}).get("competitions", [{}])[0]
        date = comp.get("date", "")[:10]
        by_id = {c["id"]: canonical(c["team"]["displayName"]) for c in comp.get("competitors", [])}
        tot: dict[str, float] = {}
        for s in _shots(m):
            s.xg = shot_xg.xg(s)
            t = by_id.get(s.team_id) or canonical(s.team_name)
            tot[t] = tot.get(t, 0.0) + s.xg
            goals += s.outcome == "goal"
            ours += s.xg
        for t, v in tot.items():
            # ESPN dates are UTC kickoff; StatsBomb match_date is local.
            for dd in (date, _shift(date, -1)):
                if (dd, t) in sb:
                    pairs.append((v, sb[(dd, t)]))
                    break
    x = np.array(pairs)
    r = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if len(x) > 2 else float("nan")
    external = {
        "team_matches_paired": len(pairs),
        "pearson_r_vs_statsbomb": round(r, 3),
        "mean_abs_diff": round(float(np.abs(x[:, 0] - x[:, 1]).mean()), 3) if len(x) else None,
        "ours_total_xg": round(ours, 1),
        "statsbomb_total_xg": round(float(x[:, 1].sum()), 1) if len(x) else None,
        "actual_goals_non_og": goals,
    }
    # WC 2026 (other coordinate convention): calibration of totals.
    wc26_cal = None
    if a.wc26_raw:
        wc26 = [s for f in sorted(Path(a.wc26_raw).glob("*.json"))
                for s in _shots(json.loads(f.read_text(encoding="utf-8"))) if not s.penalty]
        wc26_cal = {
            "shots": len(wc26),
            "goals": sum(s.outcome == "goal" for s in wc26),
            "xg_sum": round(sum(shot_xg.xg(s) for s in wc26), 1),
        }
    report = {
        "model": model,
        "held_out_club": held_out,
        "external_wc2022": external,
        "wc2026_non_penalty_calibration": wc26_cal,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _shift(date: str, days: int) -> str:
    from datetime import date as _d, timedelta

    try:
        return (_d.fromisoformat(date) + timedelta(days=days)).isoformat()
    except ValueError:
        return date


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
