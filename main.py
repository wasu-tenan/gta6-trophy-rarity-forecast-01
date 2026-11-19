"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the
percent of Exophase-tracked players who will have earned it at the case's
target, as five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95).

Model (standard library only, deterministic, no network, no clock: every
time used comes from the case file itself).

Exophase prints one rate per trophy: earners / players-tracked.  Both the
numerator (players who earned the trophy) and the denominator (players
tracked) grow over launch week, and Exophase only refreshes the page's
counts every few hours, so the reads arrive as flat steps with jumps.

Two regimes, picked from the case's history:

1. Flat history (no read differs from another, e.g. the practice cases'
   launch night: 638 players tracked, every rate unchanged for hours).
   No trend can be fitted, so the forecast is a two-branch mixture:
   with probability p a refresh lands before the target and dilutes
   every rate (newly tracked players are fresher than the tracked
   base, so they hold far fewer trophies), and with probability 1-p
   no refresh lands and rates sit where they are.  p comes from a
   lognormal refresh-interval prior (median 4 h, sigma 0.4) conditioned
   on the time already spent with no refresh; the dilution factor
   comes from the jump size J (lognormal, median 3) and a new-player
   earn ratio kappa = 0.07 (a fresh syncer earns a trophy at ~7% of
   the tracked base's rate).  The five quantiles are the mixture's
   quantiles, computed on a deterministic grid.

2. Moving history (the final case: 2.5 days of reads, many refreshes).
   Per trophy, two recency-weighted least-squares fits against
   x = ln(1 + hours since release) over the distinct reads of the last
   48 h: (A) logit(rate) on x -- the net curve, dilution and
   progression together; (B) log(earners) and log(players-tracked)
   each on x, ratio at the target -- the structural split.  Their
   logits are averaged.  Earners only grow, so fit B is
   monotone-friendly; the log-time regressor damps extrapolation
   (a week out is only ~25% more log-time than 2.5 days in).  The
   median blends the combined prediction with persistence, weighted by
   fit quality, and is capped by an archetype ceiling (platinum and
   100%-completion trophies cannot plausibly exceed a few percent in
   week one; story trophies can reach the 90s).  The spread combines
   the fit residual (with extrapolation leverage), a horizon term and
   a refresh-jump allowance.

Quantiles are normal quantiles on the logit scale around the median,
rounded to 4 decimals half-up (the starter's rounding), clamped to
[0.01, 99.99] percent, hence non-decreasing by construction.

Usage: python3 main.py [INPUT_DIR] [OUTPUT_DIR]
Reads $MAKEFUN_INPUT_DIR (else /input, else ./input) and writes
$MAKEFUN_OUTPUT_DIR (else /output, else ./output).
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path; nothing to import beside us.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = "2026-11-19T00:00:00Z"

# Refresh-interval prior (hours): Exophase refreshes a game's counts
# "on its own schedule (hours apart)".  The kit records the first gap
# at ~4.5 h (page unchanged 23:48Z-04:00Z, refreshed by 04:30Z).
REFRESH_MEDIAN_H = 4.0
REFRESH_SIGMA = 0.4
# Size of a refresh's jump in players-tracked (lognormal median).
JUMP_MEDIAN = 3.0
JUMP_SIGMA = 0.5
# A newly tracked player's chance of holding a trophy, relative to the
# tracked base's rate.  From the kit's first refresh: 3,230 new players
# added ~103 non-intro trophies vs the base's 305 over 638 players.
KAPPA_NEW = 0.07

# Week-one median ceilings (percent) by trophy archetype.  These only
# bind a runaway extrapolation; every value is far above the rate the
# trend fit gives a well-behaved trophy.
CEILING = {
    "platinum": 4.0,
    "completion": 3.0,
    "grind": 30.0,
    "story_final": 60.0,
    "story": 95.0,
    "action": 65.0,
    "other": 50.0,
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 to epoch seconds; a date with no timezone means UTC."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def archetype(trophy):
    """Classify a trophy from its tier and description."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc or "attain 100%" in desc:
        return "completion"
    if "complete chapter" in desc or "final mission" in desc:
        if "final mission" in desc and "complete the final mission" in desc and "chapter" not in desc:
            return "story_final"
        return "story"
    if "complete" in desc and ("mission" in desc or "strand" in desc):
        return "story"
    grind_words = (
        "complete 250", "complete 50", "complete 8", "complete 3",
        "every spot", "every location", "all of leonida", "100 vehicles",
        "100 locations", "win 20", "gold medal in all", "earn $",
        "spend $", "purchase all", "rob 5", "at 100",
    )
    if any(w in desc for w in grind_words):
        return "grind"
    return "action" if any(
        w in desc
        for w in ("beat ", "win ", "lose a ", "rob ", "take down ",
                  "complete an ", "enjoy a ", "smash ", "follow-through",
                  "bowl ", "make an ", "register ")
    ) else "other"


def distinct_reads(history):
    """Reads whose (players_tracked, rates) differ from the previous kept read."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev:
            out.append(read)
            prev = key
    return out


# ---------------------------------------------------------------- flat path

def refresh_probability(elapsed_h, horizon_h):
    """P(a refresh lands in the next horizon_h hours) given the page has
    already sat unchanged for elapsed_h hours, refresh intervals
    lognormal(median REFRESH_MEDIAN_H, sigma REFRESH_SIGMA)."""
    if horizon_h <= 0:
        return 0.0
    mu = math.log(REFRESH_MEDIAN_H)
    sig = REFRESH_SIGMA
    p_no_refresh_yet = 1.0 - norm_cdf((math.log(max(elapsed_h, 1e-6)) - mu) / sig)
    p_no_refresh_by_target = 1.0 - norm_cdf(
        (math.log(elapsed_h + horizon_h) - mu) / sig)
    if p_no_refresh_yet <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - p_no_refresh_by_target / p_no_refresh_yet))


def mixture_quantiles(p_refresh, rate):
    """Quantiles of the two-branch flat-history mixture for one trophy.

    Branch A (prob 1-p): no refresh, rate ~ rate * 1.01 (tiny play).
    Branch B (prob p): a refresh lands; players-tracked jumps by J and
    the rate becomes rate * (1 + kappa*(J-1)) / J.  Each branch carries
    a small lognormal jitter (sigma 0.12 / 0.25).  The 100% trophy is
    handled by the caller.
    """
    if rate >= 99.99:
        return [90.0, 96.0, 99.0, 99.9, 100.0]
    rate = max(rate, 0.01)  # a trophy with no earners yet still needs a centre
    # Deterministic grid over the jump size (lognormal quantiles).
    jump_factors = []
    for k in range(21):
        u = (k + 0.5) / 21.0
        z = _norm_ppf(u)
        j = math.exp(math.log(JUMP_MEDIAN) + JUMP_SIGMA * z)
        jump_factors.append((1.0 + KAPPA_NEW * (j - 1.0)) / j)
    center_a = math.log(rate * 1.01)
    centers_b = [math.log(max(rate * f, 0.01)) for f in jump_factors]

    def mixture_cdf(log_rate):
        # Branch A: lognormal around log(rate*1.01), sigma 0.12.
        pa = norm_cdf((log_rate - center_a) / 0.12)
        # Branch B: average over jump grid, each lognormal sigma 0.25.
        pb = 0.0
        for c in centers_b:
            pb += norm_cdf((log_rate - c) / 0.25)
        pb /= len(centers_b)
        return (1.0 - p_refresh) * pa + p_refresh * pb

    # Invert the mixture CDF by bisection (monotone in log-rate).
    lo = math.log(max(rate * 0.02, 0.01))
    hi = math.log(min(rate * 1.6, 99.99))
    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(28):
            m = 0.5 * (a + b)
            if mixture_cdf(m) < tau:
                a = m
            else:
                b = m
        out.append(round4(clamp_pct(math.exp(0.5 * (a + b)))))
    # Monotone after rounding.
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _norm_ppf(u):
    """Acklam's inverse normal CDF approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow:
        q = math.sqrt(-2 * math.log(u))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if u > phigh:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = u - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------- trend path

def _wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_std, n)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, len(xs))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(len(xs) - 2, 1)
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw
    return (a, b, math.sqrt(var * len(xs) / dof), len(xs))


def trend_forecast(case, trophy_idx, trophy, last_rate, last_n):
    """Median + spread (logit units) from the moving-history fit.

    Two fits are combined (mean of their logits):
    A. logit(rate) on ln(1 + hours since release) -- the net curve,
       dilution and progression together.
    B. log(earners) and log(players-tracked) each on ln(1 + hours),
       ratio at the target -- the structural split.
    (A linear-time fit was tried and rejected: it overshoots badly --
    progression curves saturate, so linear time extrapolates the steep
    phase far past the target; synthetic-final loss 0.75-0.84 vs 0.38-0.54
    for the log-time pair.)
    Both use recency-weighted least squares over the distinct reads of
    the last 48 h.  The median blends the combined prediction with
    persistence, weighted by fit quality.

    Returns (median_pct, spread) or None when no fit is possible.
    """
    history = case["history"]
    at = parse_time(case["at"])
    release = parse_time(case.get("game", {}).get("release_date", RELEASE)
                         if isinstance(case.get("game"), dict) else RELEASE)
    horizon_h = case["horizon_minutes"] / 60.0
    target_h = (at - release) / 3600.0 + horizon_h

    # Reads in the last 48 h before the checkpoint, distinct only.
    window = [r for r in distinct_reads(history)
              if (at - parse_time(r["at"])) <= 48 * 3600]
    if len({parse_time(r["at"]) for r in window}) < 4:
        window = distinct_reads(history)
    if len({parse_time(r["at"]) for r in window}) < 3:
        return None

    def x_of(read):
        h = max((parse_time(read["at"]) - release) / 3600.0, 0.0)
        return math.log1p(h)

    xs, ys_r, ys_e, ys_n, ws = [], [], [], [], []
    for read in window:
        rates = read.get("rates") or []
        if trophy_idx >= len(rates) or rates[trophy_idx] is None:
            continue
        n = read.get("players_tracked") or 0
        r = rates[trophy_idx]
        if n <= 0 or r is None:
            continue
        e = max(r / 100.0 * n, 0.5)  # earners, half-count floor at zero
        xs.append(x_of(read))
        ys_r.append(logit(r))
        ys_e.append(math.log(e))
        ys_n.append(math.log(n))
        age_h = (at - parse_time(read["at"])) / 3600.0
        ws.append(0.5 ** (age_h / 24.0))
    if len(set(xs)) < 3:
        return None

    x_t = math.log1p(max(target_h, 0.0))
    preds = []
    r2s = []
    resids = []
    fit_r = _wls(xs, ys_r, ws)
    if fit_r is not None:
        ar, br, sr, _ = fit_r
        preds.append(ar + br * x_t)
        r2s.append(_r2(ys_r, fit_r, xs))
        resids.append(sr)
    fit_e = _wls(xs, ys_e, ws)
    fit_n = _wls(xs, ys_n, ws)
    if fit_e is not None and fit_n is not None:
        ae, be, se, _ = fit_e
        an, bn, sn, _ = fit_n
        log_e = max(ae + be * x_t,
                    math.log(max(last_rate / 100.0 * last_n, 0.5)))
        log_n = max(an + bn * x_t, math.log(max(last_n, 1)))
        rate_b = clamp_pct(100.0 * math.exp(log_e - log_n))
        preds.append(logit(rate_b))
        r2s.append(_r2(ys_e, fit_e, xs))
        resids.append(math.sqrt(se * se + sn * sn))
    if not preds:
        return None
    pred_logit = sum(preds) / len(preds)

    # Blend with persistence by fit quality.
    r2 = max(r2s) if r2s else 0.0
    w = 0.45 + 0.08 * min(len(set(xs)), 10) + 0.35 * max(0.0, min(r2, 1.0))
    w = max(0.45, min(0.90, w))
    med_logit = w * pred_logit + (1.0 - w) * logit(last_rate)
    median = inv_logit(med_logit)
    # Archetype ceiling.
    median = min(median, CEILING[archetype(trophy)])
    if last_rate >= 99.99:
        median = 99.0

    # Spread: base horizon term + fit residual with leverage + jump term.
    x_bar = sum(w_ * x_ for x_, w_ in zip(xs, ws)) / sum(ws)
    sxx = sum(w_ * (x_ - x_bar) ** 2 for x_, w_ in zip(xs, ws))
    leverage = (x_t - x_bar) ** 2 / sxx if sxx > 1e-12 else 1.0
    s_fit = max(resids) * math.sqrt(1.0 + leverage) if resids else 0.5
    s_base = 0.35 + 0.12 * math.sqrt(case["horizon_minutes"] / 1440.0)
    s_jump = 0.30
    spread = math.sqrt(s_base ** 2 + s_fit ** 2 + s_jump ** 2)
    spread = max(0.40, min(1.6, spread))
    return (median, spread)


def _r2(ys, fit, xs):
    a, b, _, _ = fit
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = _var(ys) * len(ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _var(ys):
    m = sum(ys) / len(ys)
    return sum((y - m) ** 2 for y in ys) / len(ys)


# ------------------------------------------------------------------- driver

def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    if not case.get("history"):
        raise ValueError("case has no history")
    history = case["history"]
    last = history[-1]
    last_rates = last["rates"]
    last_n = last.get("players_tracked") or 0
    horizon_h = case["horizon_minutes"] / 60.0

    distinct = distinct_reads(history)
    flat = len(distinct) <= 1
    if not flat:
        # Flat in rates but N moving (or vice versa) still counts as moving.
        flat = all(
            (r.get("players_tracked"), tuple(r.get("rates") or ())) ==
            (distinct[0].get("players_tracked"), tuple(distinct[0].get("rates") or ()))
            for r in distinct
        )

    rows = []
    if flat:
        first_at = parse_time(history[0]["at"])
        elapsed_h = max((parse_time(case["at"]) - first_at) / 3600.0, 0.0)
        p = refresh_probability(elapsed_h, horizon_h)
        for i, trophy in enumerate(case["trophies"]):
            rate = last_rates[i] if last_rates[i] is not None else 0.0
            rows.append({"id": trophy["id"], "q": mixture_quantiles(p, rate)})
        return {"trophies": rows}

    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if last_rates[i] is not None else 0.0
        if rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [90.0, 96.0, 99.0, 99.9, 100.0]})
            continue
        got = trend_forecast(case, i, trophy, rate, last_n)
        if got is None:
            # No usable fit: persistence with a random-walk spread.
            spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
            spread = max(0.35, min(1.6, spread))
            q = [round4(inv_logit(logit(rate) + z * spread)) for z in Z]
            rows.append({"id": trophy["id"], "q": q})
            continue
        median, spread = got
        q = [round4(clamp_pct(inv_logit(logit(median) + z * spread))) for z in Z]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) >= 1:
        in_dir = Path(args[0])
    if len(args) >= 2:
        out_dir = Path(args[1])
    if not in_dir.is_dir():
        in_dir = Path("./input")
    if not out_dir.is_dir() and str(out_dir) in ("/output", "./output"):
        out_dir = Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (out_dir / ("%s.out" % path.stem)).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
