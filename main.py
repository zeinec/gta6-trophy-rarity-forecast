#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — make.fun entry gta6-trophy-rarity-forecast-01.

Forecasts every trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), from the house's recorded reads up to the
checkpoint. Stdlib only, offline, deterministic.

Why this shape: a trophy's rate is holders / players_tracked, and the two
sides move for different reasons, so they are forecast separately.
Exophase's page is frozen between its refreshes (hours apart), so the
history is a step function and only refresh points carry signal. New
players arrive in a flood after launch and hold almost nothing but the
intro trophy, so every rate dilutes; the few early trophies (Chapter 1,
fishing, arcade) recover as the base matures.

Method, per trophy:
  1. Collapse the history into refresh points: consecutive reads with
     identical (players_tracked, rates) are one observation.
  2. Players N(t): log N against log hours since the New Zealand unlock
     (2026-11-18T11:00Z). Quadratic inside the observed range; beyond it a
     damped power law, because a player count cannot turn down the way a
     concave quadratic would. With fewer than two refresh points, fall
     back to a launch-calibrated anchor curve (638 @ 12.8 h, 3868 @
     17.5 h, ~13500 @ 28.3 h).
  3. Holders H(t): across refreshes, dH = a*dN + b*dt*N_prev, where a is
     the new cohort's holding probability for the trophy and b the
     existing base's earn rate. Fit through the origin (least squares),
     shrunk toward a prior: a0 = last rate x earliness (read from the
     trophy text — chapter number, activity), b0 a small pooled value
     by tier. With too few transitions, use the prior outright.
  4. Median rate = H_target / N_target. Laplace fan on the logit scale
     with a fatter lower tail (a bigger refresh can only dilute further):
     widths (1.3, 0.9, 0, 0.9, 0.7), spread from the N-forecast
     uncertainty plus the holder-fit residual.
  5. When the history has at least three refresh points, the effective
     target is pulled earlier by half the median refresh gap (capped at
     6 h): the graded answer is the state at the last refresh before the
     target, not the target itself.

I/O: reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
FAN_WIDTHS = (1.3, 0.9, 0.0, 0.9, 0.7)

# New Zealand unlock, when the first players could earn trophies.
UNLOCK = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)
# Launch-calibrated player counts: (hours since unlock, players tracked).
ANCHOR_POINTS = ((12.83, 638.0), (17.5, 3868.0), (28.3, 13500.0))
# Pooled per-player-per-hour earn rate among existing players, by tier.
BASE_EARN_PRIOR = {
    "platinum": 0.000002,
    "gold": 0.000005,
    "silver": 0.000005,
    "bronze": 0.00001,
}
# Last-resort median rate (percent) when a trophy was never observed.
TIER_FALLBACK_MEDIAN = {
    "platinum": 0.10,
    "gold": 0.50,
    "silver": 0.80,
    "bronze": 1.00,
}


def parse_time(text):
    """ISO-8601 with Z or offset, with or without fractional seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_unlock(moment):
    return max((moment - UNLOCK).total_seconds() / 3600.0, 0.1)


def clamp_rate(pct):
    return min(max(pct, 0.01), 99.99)


def logit_pct(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    return 1.0 / (1.0 + math.exp(-x)) * 100.0


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; returns None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
                  ) / aug[i][i]
    return out


def polyfit(xs, ys, degree):
    """Least-squares polynomial (coefficients low to high degree)."""
    m = degree + 1
    matrix = [[sum(x ** (i + j) for x in xs) for j in range(m)]
              for i in range(m)]
    rhs = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(m)]
    coeffs = solve_linear(matrix, rhs)
    if coeffs is None:
        # Degenerate inputs: fall back to the mean.
        mean_y = sum(ys) / len(ys)
        return [mean_y] + [0.0] * degree
    return coeffs


def polyval(coeffs, x):
    total = 0.0
    for c in reversed(coeffs):
        total = total * x + c
    return total


def earliness(trophy):
    """How early a trophy is, as a factor on its last rate.

    Calibrated on the launch -> 19 Nov new-cohort profile: a fresh player
    holds the intro trophy and a few early ones and almost nothing late,
    so the factor tracks chapter number and activity, not the rate level.
    A trophy's prior new-cohort holding probability is last rate x this.
    """
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        return 0.0
    if "gone fishing" in text:
        return 1.0
    if "chapter 1" in text:
        return 0.56
    if "chapter 2" in text:
        return 0.08
    if "chapter 3" in text or "chapter 4" in text:
        return 0.01
    if "final mission" in text or "100% completion" in text:
        return 0.01
    if "fishing" in text:
        return 0.90
    if "arcade" in text:
        return 0.40
    if "breakfast run" in text:
        return 0.18
    if "discover 100 vehicles" in text:
        return 0.15
    if "unscheduled departure" in text:
        return 0.11
    if "what the hell" in text:
        return 0.10
    if "melee drive-by" in text:
        return 0.03
    if "bowling" in text or "bowl " in text:
        return 0.07
    if "wanted level" in text:
        return 0.05
    if "rhino" in text:
        return 0.04
    if "cocktail" in text:
        return 0.04
    if "combat focus" in text:
        return 0.03
    if "jewelry store" in text or "gold medal" in text:
        return 0.02
    return 0.005


def refresh_points(history, n_trophies):
    """Collapse frozen reads into refresh observations.

    Returns [(time, players_tracked, [rate or None per trophy]), ...].
    """
    points = []
    prev_key = None
    for read in history or []:
        rates = list(read.get("rates") or [])
        rates += [None] * (n_trophies - len(rates))
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        points.append((parse_time(read["at"]), read.get("players_tracked"),
                       rates[:n_trophies]))
    return points


def forecast_players(points, target, last_n):
    """Forecast players tracked at target; returns (N_target, sigma).

    Log N against log hours since unlock. A quadratic fits the observed
    range; beyond it, extrapolate as a power law whose exponent is the
    last log-log slope damped by 0.45 (a player count cannot turn down).
    Sigma grows with how far past the last observation the target sits.
    """
    if len(points) >= 2:
        # Refresh points carry rates too; the player fit needs (time, N).
        series = [(moment, players) for moment, players, _ in points]
    else:
        series = [(UNLOCK + timedelta(hours=h), n) for h, n in ANCHOR_POINTS]
    xs = [math.log(hours_since_unlock(t)) for t, _ in series]
    ys = [math.log(n) for _, n in series]
    degree = 2 if len(series) >= 3 else 1
    coeffs = polyfit(xs, ys, degree)
    x_target = math.log(hours_since_unlock(target))
    x_last, y_last = xs[-1], ys[-1]
    if x_target <= x_last:
        log_n = polyval(coeffs, x_target)
        past_h = 0.0
    else:
        # Beyond the range, grow as a power law from the last point,
        # with the most recent log-log slope (last secant) damped by
        # 0.45: the fitted quadratic is concave and would turn down if
        # extrapolated, but the player count keeps growing.
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        exponent = min(max(0.45 * slope, 0.25), 1.5)
        log_n = y_last + exponent * (x_target - x_last)
        past_h = (target - series[-1][0]).total_seconds() / 3600.0
    sigma = min(max(0.10 + 0.05 * math.sqrt(max(past_h, 0.0) / 24.0),
                    0.10), 0.60)
    # Floor at the last observed count; cap growth at 40x the larger of
    # the case's last count and the fit's last anchor (the prior path's
    # anchors run ahead of a short case history).
    cap = max(last_n, math.exp(y_last)) * 40.0
    n_target = math.exp(min(max(log_n, math.log(last_n)), math.log(cap)))
    return n_target, sigma


def holder_prior(trophy, last_rate_pct):
    """Prior (a0, b0) for the holder regression of one trophy."""
    if last_rate_pct is None:
        last_rate_pct = TIER_FALLBACK_MEDIAN.get(trophy.get("tier"), 1.0)
    a0 = (last_rate_pct / 100.0) * earliness(trophy)
    b0 = BASE_EARN_PRIOR.get(trophy.get("tier", "bronze"), 0.00001)
    return a0, b0


def fit_holders(transitions, a0, b0):
    """Least-squares dH = a*dN + b*dt*N_prev through the origin.

    Returns (a, b, rms) with the fit shrunk toward the prior: the more
    transitions observed, the more the data is trusted. Falls back to
    the prior when the system is singular.
    """
    saa = sum(d * d for d, _, _ in transitions)
    sab = sum(d * u for d, u, _ in transitions)
    sbb = sum(u * u for _, u, _ in transitions)
    ta = sum(d * h for d, _, h in transitions)
    tb = sum(u * h for _, u, h in transitions)
    det = saa * sbb - sab * sab
    if abs(det) <= 1e-12 * max(saa * sbb, 1.0):
        return a0, b0, 0.0
    a_ols = (ta * sbb - tb * sab) / det
    b_ols = (tb * saa - ta * sab) / det
    weight = len(transitions) / (len(transitions) + 1.0)
    a = max(weight * a_ols + (1.0 - weight) * a0, 0.0)
    b = max(weight * b_ols + (1.0 - weight) * b0, 0.0)
    sq_err = sum((h - (a * d + b * u)) ** 2 for d, u, h in transitions)
    rms = math.sqrt(sq_err / len(transitions))
    return a, b, rms


def forecast_case(case):
    trophies = case.get("trophies") or []
    n = len(trophies)
    points = refresh_points(case.get("history"), n)
    target = parse_time(case["target"])

    # The graded answer is the state at the last refresh before the
    # target, so pull the effective target earlier when refreshes are
    # frequent enough to say how often they land.
    if len(points) >= 3:
        gaps = sorted(
            (points[i][0] - points[i - 1][0]).total_seconds() / 3600.0
            for i in range(1, len(points)))
        mid = len(gaps) // 2
        median_gap = gaps[mid] if len(gaps) % 2 == 1 else \
            0.5 * (gaps[mid - 1] + gaps[mid])
        target_eff = target - timedelta(hours=min(0.5 * median_gap, 6.0))
    else:
        target_eff = target

    if not points:
        rows = []
        for trophy in trophies:
            median = TIER_FALLBACK_MEDIAN.get(trophy.get("tier"), 1.0)
            rows.append({"id": trophy["id"],
                         "q": fan_rows(median, 1.2)})
        rows.sort(key=lambda r: r["id"])
        return {"trophies": rows}

    last_t, last_n, _ = points[-1]
    n_target, sig_n = forecast_players(points, target_eff, last_n)

    rows = []
    for idx, trophy in enumerate(trophies):
        # This trophy's own observed (time, players, holders) series.
        series = []
        for moment, players, rates in points:
            rate = rates[idx] if idx < len(rates) else None
            if rate is None or players is None:
                continue
            series.append((moment, players, rate / 100.0 * players))
        if not series:
            median = TIER_FALLBACK_MEDIAN.get(trophy.get("tier"), 1.0)
            rows.append({"id": trophy["id"],
                         "q": fan_rows(median, 1.2)})
            continue
        base_t, base_n, base_h = series[-1]
        last_rate = base_h / base_n * 100.0 if base_n else None
        a0, b0 = holder_prior(trophy, last_rate)
        transitions = []
        for j in range(1, len(series)):
            t0, n0, h0 = series[j - 1]
            t1, n1, h1 = series[j]
            dt_h = (t1 - t0).total_seconds() / 3600.0
            transitions.append((n1 - n0, dt_h * n0, h1 - h0))
        if len(transitions) >= 2:
            a, b, rms = fit_holders(transitions, a0, b0)
            spread_h = 0.30 + min(rms / max(base_h, 1.0), 0.5)
        else:
            a, b = a0, b0
            spread_h = 0.55
        horizon_h = max((target_eff - base_t).total_seconds() / 3600.0, 0.0)
        # The base's earn rate b applies to the incumbent base that was
        # measured (base_n), not to a growing average: newcomers over
        # the horizon are already counted by the a-term, and counting
        # them in b too would double-count their earning.
        holders = base_h + a * (n_target - base_n) \
            + b * horizon_h * base_n
        holders = max(holders, 0.0)
        median = clamp_rate(holders / n_target * 100.0) if n_target > 0 \
            else last_rate or 1.0
        # A log error in players shifts every log rate by the same
        # amount, so the players uncertainty enters one-for-one.
        spread = 0.8 * math.sqrt(spread_h ** 2 + sig_n ** 2 + 0.10 ** 2)
        spread = min(max(spread, 0.25), 1.2)
        rows.append({"id": trophy["id"],
                     "q": fan_rows(median, spread)})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def fan_rows(median, spread):
    """Laplace quantiles on the logit scale; fat lower tail, 4 decimals."""
    centre = logit_pct(median)
    out = []
    for z, width in zip(Z_SCORES, FAN_WIDTHS):
        q = clamp_rate(inv_logit_pct(centre + z * width * spread))
        out.append(round(q, 4))
    for k in range(1, 5):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for in_path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(in_path) as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
