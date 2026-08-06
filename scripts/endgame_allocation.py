"""Price every way of spending the remaining submission slots, and stress the plan.

Three objections a strict reviewer would raise against "spend tomorrow on w0.62
replicates", answered with numbers rather than intuition:

  1. Selection. We have run many single-draw probes. Under the null, the best-looking
     one is roughly -1.4 sigma purely by selection, so w0.62's -0.93 may be *less*
     extreme than chance predicts, not more.
  2. Power. Detecting a -0.05 shift needs a stated power, not just a standard error.
  3. Decorrelation. If the public-to-private offset is shared across configs in one
     family, both final slots cross the threshold together and the second slot buys
     almost nothing — which would overturn the current slot-2 rationale.

It also checks whether the draw distribution is heavier-tailed than Gaussian, since
best-of-N is worth more than the Gaussian estimate if it is.

Usage:
    uv run python scripts/endgame_allocation.py
"""

from __future__ import annotations

import math
import re
import statistics as st
from collections import defaultdict

COMPETITION = "rogii-wellbore-geology-prediction"
THRESHOLD = 6.000
OFFSET_SD = 0.6          # public->private offset uncertainty, mid of the 0.4-0.8 range
SEED_SD_PRIVATE = 0.055  # per-run seed noise, taken from the public replicate spread

# Asked-for page size. The endpoint returns everything up to this and never paginates,
# so a result of exactly PAGE_SIZE means we were truncated and must not be trusted
# (see scripts/submission_status.py, which was hardened against the same failure).
PAGE_SIZE = 1000


def phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def Phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def kernel_of(sub) -> str:
    m = re.search(r"/code/[^/]+/([^?]+)", str(getattr(sub, "url", "") or ""))
    return m.group(1) if m else "<unknown>"


def expected_gain(best: float, mu: float, sd: float) -> float:
    """E[(best - X)+] for one more draw: what best-of-N still buys."""
    a = (best - mu) / sd
    return sd * (a * Phi(a) + phi(a))


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    all_subs = list(api.competition_submissions(COMPETITION, page_size=PAGE_SIZE) or [])
    if len(all_subs) >= PAGE_SIZE:
        raise SystemExit(
            f"ABORT: got exactly {len(all_subs)} submissions == page_size. The list is "
            f"truncated and page_number does not work on this endpoint. Raise PAGE_SIZE."
        )
    subs = [s for s in all_subs if str(getattr(s, "public_score", "") or "").strip()]

    by = defaultdict(list)
    for s in subs:
        by[kernel_of(s)].append(float(s.public_score))

    ref = "rogii-gs145-probe"
    base = by[ref]
    mu, sd = st.mean(base), st.stdev(base)
    print(f"reference {ref}: n={len(base)} mean={mu:.4f} sd={sd:.4f} best={min(base):.3f}")

    # --- objection 1: selection across single-draw probes -------------------
    singles = {k: v[0] for k, v in by.items() if len(v) == 1 and k != ref}
    print(f"\n1) SELECTION — {len(singles)} single-draw probes")
    zs = sorted((v - mu) / sd for v in singles.values())
    print(f"   observed z of the best single probe: {zs[0]:+.2f}")
    # E[min of n standard normals], Blom's approximation
    n = len(singles)
    e_min = -_expected_min_normal(n)
    print(f"   E[min z] under the null with n={n}: {e_min:+.2f}")
    print(f"   -> the best probe is {'NOT ' if zs[0] > e_min else ''}more extreme than "
          f"selection alone predicts")
    for k, v in sorted(singles.items(), key=lambda kv: kv[1])[:4]:
        print(f"      {v:.3f}  z={(v - mu) / sd:+.2f}  {k}")

    # --- objection 2: power -------------------------------------------------
    print("\n2) POWER — draws of a candidate config needed to detect a mean shift")
    print(f"   {'shift':>7}  {'n for 50%':>10}  {'n for 80%':>10}")
    for shift in (0.03, 0.05, 0.08, 0.12):
        cells = []
        for zpow in (0.0, 0.84):          # 50% and 80% power, one-sided alpha=0.05
            # need SE_diff <= shift / (1.645 + zpow), with SE_diff^2 = sd^2/n + sd^2/13
            budget = (shift / (1.645 + zpow)) ** 2 - sd**2 / len(base)
            cells.append(str(math.ceil(sd**2 / budget)) if budget > 0 else ">100")
        print(f"   {shift:7.2f}  {cells[0]:>10}  {cells[1]:>10}")

    # --- objection 3: does a second slot help? ------------------------------
    print("\n3) DECORRELATION — P(private < 6.000)")
    t = THRESHOLD - mu
    one = Phi(t / math.hypot(OFFSET_SD, SEED_SD_PRIVATE))
    # two submissions share the offset delta, differ only by seed noise
    two = _p_either(t, OFFSET_SD, SEED_SD_PRIVATE)
    print(f"   offset sd {OFFSET_SD}, seed sd {SEED_SD_PRIVATE}, gap to threshold {t:+.3f}")
    print(f"   one submission            : {one*100:5.2f}%")
    print(f"   two (shared offset)       : {two*100:5.2f}%   (+{(two-one)*100:.2f} pts)")
    for r in (0.0, 0.5, 0.9):
        shift = ((min(base) - mu) / sd) * r * SEED_SD_PRIVATE
        p = Phi((t - shift) / math.hypot(OFFSET_SD, SEED_SD_PRIVATE))
        print(f"   picking the best public draw, rho(pub,priv seed)={r:.1f}: "
              f"{p*100:5.2f}%  ({(p-one)*100:+.2f} pts)")

    # --- best-of-N saturation and tail shape --------------------------------
    print("\n4) BEST-OF-N — what one more draw is worth")
    for k in (ref, "rogii-iaztec-gs13-fork-probe"):
        v = by[k]
        if len(v) < 2:
            continue
        m2, s2 = st.mean(v), st.stdev(v)
        g = expected_gain(min(v), m2, s2)
        print(f"   {k}: n={len(v)} E[gain]={g:.5f} ft  "
              f"P(improve)={Phi((min(v)-m2)/s2)*100:.1f}%")
    z = sorted((x - mu) / sd for x in base)
    skew = sum(x**3 for x in z) / len(z)
    tail = "left tail heavier — best-of-N worth more" if skew < -0.3 else "no heavy left tail"
    print(f"   GS1.45 draw skew={skew:+.2f} ({tail})")
    return 0


def _expected_min_normal(n: int) -> float:
    """Blom: E[min of n standard normals] ~ -Phi^-1((n - 0.375)/(n + 0.25))."""
    p = (n - 0.375) / (n + 0.25)
    # inverse normal via bisection, plenty accurate here
    lo, hi = 0.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _p_either(t: float, off_sd: float, seed_sd: float, n: int = 40001) -> float:
    """P(min of two seed draws below t) integrating over the shared offset."""
    lo, hi = -5 * off_sd, 5 * off_sd
    step = (hi - lo) / (n - 1)
    total = 0.0
    for i in range(n):
        d = lo + i * step
        w = phi(d / off_sd) / off_sd
        p1 = Phi((t - d) / seed_sd)
        total += w * (1 - (1 - p1) ** 2) * step
    return total


if __name__ == "__main__":
    raise SystemExit(main())
