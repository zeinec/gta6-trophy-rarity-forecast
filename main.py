#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster.

Forecasts, for every PS5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will hold it at the case's target time, as
five quantiles per trophy. Scored by pinball loss on the logit scale.

Model (see work/DESIGN.md for the evidence behind it):

  rate(t) = Holders(t) / Players(t), and the two sides are forecast
  separately, then divided.

  * Exophase's page is frozen between its own refreshes, so consecutive
    reads with identical (players, rates) carry no new information. Only
    the segment ends -- one observation per refresh -- are used.
  * Players N(t): log N is fit against log hours since the NZ unlock
    (quadratic with >= 3 refresh points, linear with 2, and a fixed
    launch-calibrated prior curve with fewer). Inside the observed range
    the fit is used as-is; beyond it the curve continues as a power law
    with the last secant slope, damped by 0.45: the raw quadratic's end
    derivative badly undershoots growth, while the raw last secant
    overshoots wildly. A raw quadratic is never extrapolated as one.
  * Holders H_t(t): across refreshes, per trophy,
        dH = a_t * dN + b_t * dt * N_prev
    (a_t = the new cohort's holding probability for this trophy,
    b_t = the per-player-per-hour earn rate among existing players).
    Only a_t is estimated per trophy, by a single-regressor ridge
    toward the earliness prior, with b_t held at the pooled prior:
    the two regressors correlate at ~0.99, so a two-regressor fit
    swings b by an order of magnitude on a modest pull of a. The
    forecast base term uses the incumbent N_last, not the average of
    N_last and N_target: newcomers over the horizon are already
    counted by the a term.
    With fewer than 2 refresh points there is no regression to run, so
    the prior is used directly: a_t = last_rate * earliness(t), where
    earliness is read off the trophy text (chapter number, activity
    keywords) and calibrated to the measured launch-to-now cohort
    profile. b_t's prior is a small pooled value.
  * The median rate is H_target / N_target. The fan is a Laplace fan on
    the logit scale with a fat lower tail (dilution risk is one-sided).

Standard library only, offline, deterministic, no randomness.
Reads MAKEFUN_INPUT_DIR (default /input), writes MAKEFUN_OUTPUT_DIR
(default /output), one <case id>.out per <case id>.in.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# New Zealand unlock, the start of the launch clock.
RELEASE = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)

# Launch-calibrated prior growth curve: (hours since release, players).
# Three known refresh points: 638 @ 12.8 h, 3,868 @ 17.5 h, ~13,500 @ 28.3 h.
PRIOR_GROWTH = ((12.833333, 638.0), (17.5, 3868.0), (28.3, 13500.0))

# Standard-normal quantiles for the five levels 0.05/0.25/0.5/0.75/0.95.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Fan widths per quantile: fat lower tail, narrow upper tail.
FAN_WIDTHS = (1.3, 0.9, 1.0, 0.9, 0.7)

# Prior per-player-per-hour earn rate among existing players.
# 2e-5: the value the launch -> now totals support when the b term is
# based on the incumbent N_last (see holder_model). An earlier draft
# cut it to 5e-6 while the b term wrongly used the average of N_last
# and N_target, which inflated the b term 4-11x; with the base fixed,
# 2e-5 is right on both checks (backtest and smoke).
B_PRIOR = 2e-5

CHAPTER_EARLINESS = {1: 0.56, 2: 0.08, 3: 0.02, 4: 0.01}
TIER_EARLINESS = {"bronze": 0.02, "silver": 0.005, "gold": 0.002,
                  "platinum": 0.0}
# Wide-fan median rates (percent) when a case carries no history at all.
TIER_PRIOR_RATE = {"bronze": 0.8, "silver": 0.8, "gold": 0.5,
                   "platinum": 0.3}


def parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to aware UTC."""
    if text is None:
        return None
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_release(moment):
    return max((moment - RELEASE).total_seconds() / 3600.0, 0.05)


def logit(pct):
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def solve_linear(matrix, vector):
    """Small dense Gaussian elimination with partial pivoting."""
    n = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                for k in range(col, n + 1):
                    rows[r][k] -= factor * rows[col][k]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def polyfit(xs, ys, degree):
    """Least-squares polynomial fit; returns coefficients low to high."""
    n = degree + 1
    matrix = [[sum(x ** (i + j) for x in xs) for j in range(n)]
              for i in range(n)]
    vector = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(n)]
    coef = solve_linear(matrix, vector)
    if coef is None:
        return None
    return coef


def polyval(coef, x):
    total = 0.0
    for c in reversed(coef):
        total = total * x + c
    return total


def polyderiv(coef, x):
    total = 0.0
    for power in range(len(coef) - 1, 0, -1):
        total = total * x + power * coef[power]
    return total


def earliness(trophy):
    """How early a trophy sits in the game, from its text.

    Returns a factor in [0, 1]: the share of the launch rate that a fresh
    cohort of players is expected to hold. Early story/activity trophies
    score high; late and completionist ones score near zero.
    """
    tid = trophy.get("id")
    if tid == 2:  # "Just Like Paradise": every new player earns it.
        return 1.0
    tier = str(trophy.get("tier", "bronze")).lower()
    if tier == "platinum":
        return 0.0
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    match = re.search(r"chapter\s+(\d+)", text)
    if match:
        return CHAPTER_EARLINESS.get(int(match.group(1)), 0.005)
    if "complete the final mission." in text:
        return 0.005
    if "100% completion" in text:
        return 0.001
    if "fishing" in text:
        return 0.92
    if "arcade" in text:
        return 0.40
    if "breakfast run" in text:
        return 0.18
    if "discover 100 vehicles" in text:
        return 0.15
    if "try each type of activity once" in text:
        return 0.01
    if "top fun" in text:
        return 0.01
    if any(key in text for key in ("250 stunts", "gold medal",
                                   "100 locations", "50 challenges",
                                   "collectible card", "mission strands")):
        return 0.005
    return TIER_EARLINESS.get(tier, 0.02)


def refresh_points(history, n_trophies):
    """Collapse reads into one observation per Exophase refresh.

    Consecutive reads with identical (players_tracked, rates) are one
    frozen page; only the last read of each run carries information.
    Returns [(time, players, holders)] with holders as fractions of
    players (None where a rate was missing).
    """
    points = []
    prev_key = None
    for read in history or []:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if players is None or players <= 0:
            continue
        rates = list(rates) + [None] * max(0, n_trophies - len(rates))
        key = (players, tuple(rates[:n_trophies]))
        moment = parse_time(read.get("at") or read.get("slot"))
        if moment is None:
            continue
        holders = [None if r is None else float(r) / 100.0 * players
                   for r in rates[:n_trophies]]
        if key == prev_key and points:
            points[-1] = (moment, players, holders)
        else:
            points.append((moment, players, holders))
        prev_key = key
    return points


def forecast_players(points, target, n_last):
    """Forecast players tracked at the target. Returns (N_target, sig_n).

    sig_n is the log-scale uncertainty of the N forecast, used to widen
    the rate fan.
    """
    if len(points) >= 2:
        xs = [math.log(hours_since_release(t)) for t, _, _ in points]
        ys = [math.log(n) for _, n, _ in points]
        degree = 2 if len(points) >= 3 else 1
        coef = polyfit(xs, ys, degree)
        if coef is None:
            coef = polyfit(xs, ys, 1)
        if coef is None:
            return float(n_last), 0.35
        resid = math.sqrt(sum((y - polyval(coef, x)) ** 2
                              for x, y in zip(xs, ys)) / len(xs))
        sig = clamp(0.10 + 2.0 * resid, 0.10, 0.60) if len(points) >= 3 \
            else 0.30
    else:
        xs = [math.log(h) for h, _ in PRIOR_GROWTH]
        ys = [math.log(n) for _, n in PRIOR_GROWTH]
        coef = polyfit(xs, ys, 2)
        if coef is None:
            return float(n_last), 0.35
        sig = 0.35
    x_target = math.log(hours_since_release(target))
    x_max = max(xs)
    if x_target <= x_max:
        log_n = polyval(coef, x_target)
    else:
        # Power-law continuation with the last secant slope, damped:
        # the fitted log-log quadratic is strongly concave, so its end
        # derivative badly undershoots growth (smoke: 84k vs 110k),
        # while the raw last secant (2.6) overshoots wildly (1.9M).
        # The damped secant (x0.45) lands the smoke N at +6% and is
        # the growth rate the launch anchor implies for the late window.
        if len(xs) >= 2:
            secant = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            secant = polyderiv(coef, x_max)
        slope = clamp(0.45 * secant, 0.05, 2.0)
        log_n = polyval(coef, x_max) + slope * (x_target - x_max)
    n_target = clamp(math.exp(log_n), float(n_last), 5_000_000.0)
    return n_target, sig


def holder_model(points, trophies, last_rates):
    """Per-trophy (a, b, resid) for the holder dynamics.

    a = new cohort's holding probability, b = per-player-per-hour earn
    rate among existing players, resid = relative regression residual
    (0 when the prior is used directly).

    Only a is estimated per trophy, by a single-regressor ridge toward
    the earliness prior, with b held at the pooled prior. The textbook
    two-regressor fit (a on dN, b on dt * N_prev) is unusable here: the
    two regressors correlate at ~0.99, so a modest pull on a swings the
    fitted b by an order of magnitude, and the forecast horizon's
    (dt * N) / dN ratio is 12-100x the observed intervals', so a b
    error explodes in the forecast. Measured on the multi-refresh smoke
    test: the two-regressor version scored loss 0.93 with medians ~3x
    the truth; the single-regressor version fixes that. b is anchored
    to the pooled prior because the launch -> now totals imply the
    pooled existing-player earn rate is near zero.
    """
    n = len(trophies)
    priors = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i]
        base = (last / 100.0) if last is not None else 0.008
        priors.append(clamp(base * earliness(trophy), 0.0, 1.0))
    if len(points) < 2:
        return [(priors[i], B_PRIOR, 0.0) for i in range(n)]

    intervals = []
    for k in range(1, len(points)):
        t_prev, n_prev, h_prev = points[k - 1]
        t_cur, n_cur, h_cur = points[k]
        dt = max((t_cur - t_prev).total_seconds() / 3600.0, 0.0)
        intervals.append((n_cur - n_prev, dt, n_prev, h_prev, h_cur))

    results = []
    for i in range(n):
        s_dd = 0.0
        s_dy = 0.0
        usable = 0
        for dn, dt, n_prev, h_prev, h_cur in intervals:
            if h_prev[i] is None or h_cur[i] is None:
                continue
            # Holders the pooled b term already explains in this interval.
            dh = (h_cur[i] - h_prev[i]) - B_PRIOR * dt * n_prev
            s_dd += dn * dn
            s_dy += dn * dh
            usable += 1
        if usable == 0 or s_dd <= 0:
            results.append((priors[i], B_PRIOR, 0.0))
            continue
        lam = 0.25 * s_dd
        a = clamp((s_dy + lam * priors[i]) / (s_dd + lam), 0.0, 1.0)
        h_last = points[-1][2][i]
        scale = max(h_last if h_last is not None else 0.0, 1.0)
        sq = 0.0
        for dn, dt, n_prev, h_prev, h_cur in intervals:
            if h_prev[i] is None or h_cur[i] is None:
                continue
            pred = a * dn + B_PRIOR * dt * n_prev
            sq += (pred - (h_cur[i] - h_prev[i])) ** 2
        resid = clamp(math.sqrt(sq / usable) / scale, 0.0, 0.3)
        results.append((a, B_PRIOR, resid))
    return results


def forecast_case(case):
    """Forecast one case dict; returns the output dict."""
    trophies = case.get("trophies") or []
    n = len(trophies)
    target = parse_time(case.get("target"))
    points = refresh_points(case.get("history"), n)
    last_rates = [None] * n
    if points:
        _, _, holders = points[-1]
        n_last = points[-1][1]
        for i in range(n):
            if holders[i] is not None:
                last_rates[i] = holders[i] / n_last * 100.0
    else:
        n_last = 0

    if not points or target is None:
        rows = []
        for i, trophy in enumerate(trophies):
            tier = str(trophy.get("tier", "bronze")).lower()
            med = 99.99 if trophy.get("id") == 2 else \
                TIER_PRIOR_RATE.get(tier, 0.8)
            rows.append((trophy.get("id", i + 1), med, 1.2))
    else:
        n_target, sig_n = forecast_players(points, target, n_last)
        last_time = points[-1][0]
        horizon_h = max((target - last_time).total_seconds() / 3600.0, 0.0)
        model = holder_model(points, trophies, last_rates)
        base_spread = 0.45 if len(points) < 2 else \
            0.40 if len(points) == 2 else 0.35
        rows = []
        for i, trophy in enumerate(trophies):
            a, b, resid = model[i]
            _, _, holders = points[-1]
            h_last = holders[i]
            if h_last is None:
                h_last = (last_rates[i] or 0.8) / 100.0 * n_last
            h_target = h_last + a * (n_target - n_last) + \
                b * horizon_h * n_last
            # The b term uses the incumbent base N_last, not the average
            # of N_last and N_target: newcomers over the horizon are
            # already counted by the a term, so counting them in b too
            # double-counts their earning. Measured on the real-data
            # backtest (0.1197 -> 0.0975 with the average-N version as
            # the baseline) and on the smoke test.
            h_target = clamp(h_target, h_last, n_target)
            med = h_target / n_target * 100.0
            if last_rates[i] is not None and last_rates[i] >= 99.99:
                med = 99.99
            med = clamp(med, 0.01, 99.99)
            spread = clamp(base_spread + 0.4 * sig_n + resid, 0.25, 1.2)
            rows.append((trophy.get("id", i + 1), med, spread))

    out_trophies = []
    for tid, med, spread in rows:
        center = logit(med)
        quantiles = []
        for z, width in zip(Z_SCORES, FAN_WIDTHS):
            q = inv_logit(center + z * width * spread)
            quantiles.append(round(clamp(q, 0.01, 99.99), 4))
        for j in range(1, 5):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        out_trophies.append({"id": tid, "q": quantiles})
    out_trophies.sort(key=lambda r: r["id"])
    return {"trophies": out_trophies}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name)) as fh:
            case = json.load(fh)
        result = forecast_case(case)
        stem = case.get("id") or name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
