
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    if not ms:
        raise AgreementError("Empty Sequence")

    for t in ms:
        if not _finite(t, "time") or t < 0:
            raise AgreementError()
    
    n = len(ms)
    x_mean = sum(ms) / n
    worst_case = max(ms)

    percs = {}
    for q in LATENCY_PERCENTILES:
        rank = _percentile(ms, q)
        s = sorted(ms)
        idx = (n - 1) * q/100
        lo = int(idx)
        hi = min(lo + 1, n-1)

        percs[q] = s[lo] + (s[hi] - s[lo]) * (idx - lo)

    jitter = percs[95] - percs[50]

    return{
        "n": n,
        "mean": x_mean,
        "worst": worst_case,
        "p50": percs[50],
        "p95": percs[95],
        "p99": percs[99],
        "jitter": jitter
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    _finite(base_ms, "")
    _finite(other_ms, "")

    if base_ms <= 0.0 or other_ms <= 0.0:
        raise AgreementError("negative time")

    factor = base_ms / other_ms
    faster = True if factor > 1.0 else False 
    saved_ms = base_ms - other_ms

    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved_ms
    }

def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    results_a = all(_finite(f, " ") for f in a)
    results_b = all(_finite(f, " ") for f in b)

    if not results_a or not results_b:
        raise AgreementError("Not finite")
    
    if len(a) != len(b):
        raise AgreementError("Lengths do not match")
    
    best_i = 0
    best = abs(a[0] - b[0])

    for i in range(1, len(a)):
        best_i = i if abs(a[i] - b[i]) > best else best_i
        best = abs(a[i] - b[i]) if abs(a[i] - b[i]) > best else best

    return {
        "max_abs": best,
        "index": best_i,
        "a": a[best_i],
        "b": b[best_i],
        "n": len(a)
    }

def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    if len(a) != len(b) or floor <= 0:
        raise AgreementError("lengths dont match or floor less than 0")
    
    results_a = all(_finite(f, " ") for f in a)
    results_b = all(_finite(f, " ") for f in b)

    if not results_a or not results_b:
        raise AgreementError("Not finite")
    
    floored = 0
    denom_min = 100000
    best = 0
    best_i = 0

    for i in range(len(a)):
        d = abs(b[i])
        denom_min = min(denom_min, d)
        if d < floor:
            floored = floored + 1
        
        r = abs(a[i] - b[i]) / max(d, floor)

        if r > best:
            best_i = i
            best = r
        
    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": len(a)
    }

def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    results_a = all(a for a in rows_a)
    results_b = all(b for b in rows_b)

    if not results_a or not results_b:
        raise AgreementError("zero values")
    
    if len(rows_a) != len(rows_b):
        raise AgreementError("lengths dont match")
    
    agree = 0
    first_disa = None

    for k in range(len(rows_a)):
        row_a = rows_a[k]
        row_b = rows_b[k]

        if len(row_a) == 0 or len(row_b) == 0:
            raise AgreementError("rows must be non empty")

        if len(row_a) != len(row_b):
            raise AgreementError("rows must have equal length")

        for value in row_a:
            _finite(value, f"rows_a[{k}]")
        
        for value in row_b:
            _finite(value, f"rows_b[{k}]")
        
        ia = 0
        ib = 0

        for c in range(1, len(row_a)):
            if row_a[c] > row_a[ia]:
                ia = c
        
        for c in range(1, len(row_b)):
            if row_b[c] > row_b[ib]:
                ib = c

        if ia == ib:
            agree += 1
        elif first_disa is None:
            first_disa = k
    
    n = len(rows_a)
    disagree = n - agree
    rate = agree/n
    return {
        "n": n,
        "agree": agree,
        "disagree": disagree,
        "rate": rate,
        "first_disagreement": first_disa
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    _finite(tol, "tol")

    if tol < 0:
        raise AgreementError("tolerance must be greater than 0")

    if len(rows) == 0:
        raise AgreementError("rows must be non empty")
    
    margins = []

    for k, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError("each row must have at least 2 values")

        for r in row:
            _finite(r, f"rows[{k}]")
        
        sorted_row = sorted(row, reverse=True)

        margin = sorted_row[0] - sorted_row[1]
        margins.append(margin)
    
    min_magrin = margins[0]
    arg_min = 0

    for i in range(1, len(margins)):
        if margins[i] < min_magrin:
            min_magrin = margins[i]
            arg_min = i
    
    mean = sum(margins) / len(margins)
    median = _percentile(margins, 50)

    below_tol = 0

    for margin in margins:
        if margin <tol:
            below_tol += 1
    
    safe = below_tol == 0

    return{
        "n": len(rows),
        "min": min_magrin,
        "argmin": arg_min,
        "median": median,
        "mean": mean,
        "below_tol": below_tol,
        "safe": safe
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()