#!/usr/bin/env python3
"""Parse .report.md files and categorize shapes by T/B ratio (Tensile / hipblaslt-bench).

Also extracts Tensile wall-clock time from .tensile.log files and
summarises per-model timing with smallest/largest batch size breakdowns.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent / "tunning_results" / "bf16"


def _parse_tensile_time(log_path):
    """Extract Tensile total elapsed seconds from the log's final summary line."""
    if not log_path.is_file():
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(-min(size, 8192), 2)
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"Finish Analysing data to .+ in ([\d.]+)s", tail)
    if m:
        return float(m.group(1))
    try:
        st = log_path.stat()
        birth = st.st_ctime
        modify = st.st_mtime
        if modify > birth > 0:
            return modify - birth
    except Exception:
        pass
    return None


def parse_report(path):
    text = path.read_text()

    def _field(label):
        m = re.search(rf"\|\s*{label}\s*\|\s*(.+?)\s*\|", text)
        return m.group(1).strip() if m else None

    model = _field("Model")
    layer = _field("Layer")
    phase = _field("Phase")
    mbs = _field("MBS")

    tensile_row = re.search(r"\*\*Tensile tuned\*\*\s*\|\s*([\d.]+)", text)
    api_row = re.search(r"\*\*API bench.*?\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    bench_row = re.search(r"\*\*hipblaslt-bench.*?\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    tensile_tflops = float(tensile_row.group(1)) if tensile_row else None
    api_tflops = float(api_row.group(1)) if api_row else None
    api_time_us = float(api_row.group(2)) if api_row else None
    bench_tflops = float(bench_row.group(1)) if bench_row else None
    bench_time_us = float(bench_row.group(2)) if bench_row else None

    m_val = _field("M")
    n_val = _field("N")
    k_val = _field("K")

    log_name = path.name.replace(".report.md", ".tensile.log")
    log_path = path.parent / log_name
    tensile_secs = _parse_tensile_time(log_path)

    tb_ratio = None
    if tensile_tflops and bench_tflops and bench_tflops > 0:
        tb_ratio = tensile_tflops / bench_tflops * 100

    return {
        "file": path.name,
        "model": model, "layer": layer, "phase": phase, "mbs": mbs,
        "M": m_val, "N": n_val, "K": k_val,
        "tensile": tensile_tflops, "api": api_tflops, "bench": bench_tflops,
        "api_time_us": api_time_us, "bench_time_us": bench_time_us,
        "tb_ratio": tb_ratio, "tensile_secs": tensile_secs,
    }


BUCKETS = [
    ("T/B \u2264 95%  (Tensile worse than stock)", lambda r: r <= 95),
    ("95% < T/B \u2264 100% (Tensile \u2248 stock)", lambda r: 95 < r <= 100),
    ("100% < T/B \u2264 105% (Tensile slightly better)", lambda r: 100 < r <= 105),
    ("105% < T/B \u2264 115% (Tensile better)", lambda r: 105 < r <= 115),
    ("T/B > 115% (Tensile much better)", lambda r: r > 115),
]


def _fmt_time(secs):
    if secs is None:
        return "N/A"
    mins = secs / 60
    if mins >= 60:
        return f"{mins / 60:.1f}h"
    return f"{mins:.1f}m"


def main():
    reports = sorted(REPORT_DIR.glob("*.report.md"))
    if not reports:
        print("No .report.md files found.")
        return

    entries = []
    no_tb = []
    for r in reports:
        try:
            e = parse_report(r)
        except Exception as exc:
            print(f"WARN: failed to parse {r.name}: {exc}", file=sys.stderr)
            continue
        if e["tb_ratio"] is not None:
            entries.append(e)
        else:
            no_tb.append(e)

    all_entries = entries + no_tb

    # --- Per-bucket breakdown ---
    bucketed = defaultdict(list)
    for e in entries:
        for label, pred in BUCKETS:
            if pred(e["tb_ratio"]):
                bucketed[label].append(e)
                break

    print(f"{'='*80}")
    print(f"  T/B Ratio Report (Tensile / hipblaslt-bench)")
    print(f"  {len(entries)} shapes with T/B, {len(no_tb)} without")
    print(f"{'='*80}\n")

    # --- Distribution bar chart ---
    if entries:
        BAR_WIDTH = 40
        max_count = max(len(bucketed.get(label, [])) for label, _ in BUCKETS)
        max_count = max(max_count, 1)
        short_labels = ["\u226495%", "95-100%", "100-105%", "105-115%", ">115%"]
        print("  T/B Distribution:")
        print()
        for (label, _), short in zip(BUCKETS, short_labels):
            count = len(bucketed.get(label, []))
            bar_len = int(count / max_count * BAR_WIDTH)
            bar = "\u2588" * bar_len
            pct = count / len(entries) * 100 if entries else 0
            print(f"  {short:>8} \u2502{bar:<{BAR_WIDTH}} {count:>4} ({pct:4.1f}%)")
        print(f"  {'':>8} \u2514{'\u2500'*BAR_WIDTH}")
        print()

    for label, pred in BUCKETS:
        items = bucketed.get(label, [])
        print(f"  {label}:  {len(items)} shapes")
        # Skip per-shape details for the 95-100% bucket (expected range, too many to list)
        if items and not (95 < items[0]["tb_ratio"] <= 100):
            by_model = defaultdict(list)
            for e in items:
                by_model[e["model"]].append(e)
            for model in sorted(by_model):
                shapes = by_model[model]
                print(f"    {model} ({len(shapes)}):")
                for s in sorted(shapes, key=lambda x: x["tb_ratio"]):
                    bench_ms = f"{s['bench_time_us']/1000:.2f}ms" if s.get("bench_time_us") else "N/A"
                    print(f"      T/B={s['tb_ratio']:5.1f}%  "
                          f"mbs={s['mbs']:>2}  {s['layer']:<20} [{s['phase']}]  "
                          f"M={s['M']} N={s['N']} K={s['K']}  "
                          f"T={s['tensile']:.0f} B={s['bench']:.0f}  "
                          f"({bench_ms})")
        print()

    # --- Summary stats ---
    ratios = [e["tb_ratio"] for e in entries]
    if not ratios:
        print(f"{'\u2500'*80}")
        print("  No shapes with T/B ratio to summarize.")
    else:
        print(f"{'\u2500'*80}")
        print(f"  T/B:  min={min(ratios):.1f}%  max={max(ratios):.1f}%  "
              f"mean={sum(ratios)/len(ratios):.1f}%  "
              f"median={sorted(ratios)[len(ratios)//2]:.1f}%")
        print()

        # --- Per-model summary ---
        by_model_tb = defaultdict(list)
        for e in entries:
            by_model_tb[e["model"]].append(e["tb_ratio"])

        print(f"  {'Model':<25} {'Count':>5}  {'Min':>6}  {'Mean':>6}  {'Median':>6}  {'Max':>6}")
        print(f"  {'\u2500'*25} {'\u2500'*5}  {'\u2500'*6}  {'\u2500'*6}  {'\u2500'*6}  {'\u2500'*6}")
        for model in sorted(by_model_tb):
            rs = sorted(by_model_tb[model])
            n = len(rs)
            print(f"  {model:<25} {n:>5}  {min(rs):>5.1f}%  "
                  f"{sum(rs)/n:>5.1f}%  {rs[n//2]:>5.1f}%  {max(rs):>5.1f}%")

    # --- Per-model timing ---
    print(f"\n{'\u2500'*80}")
    print("  Tensile Time per Model")
    print(f"{'\u2500'*80}\n")

    by_model_time = defaultdict(list)
    for e in all_entries:
        by_model_time[e["model"]].append(e)

    for model in sorted(by_model_time):
        shapes = by_model_time[model]
        timed = [e for e in shapes if e["tensile_secs"] is not None]
        if not timed:
            print(f"  {model}: no timing data")
            print()
            continue
        total_secs = sum(e["tensile_secs"] for e in timed)

        by_mbs = defaultdict(list)
        for e in timed:
            by_mbs[int(e["mbs"])].append(e["tensile_secs"])

        sorted_mbs = sorted(by_mbs.keys())
        print(f"  {model}")
        print(f"    Total: {len(timed)} shapes, {_fmt_time(total_secs)}  "
              f"(avg {_fmt_time(total_secs / len(timed))} / shape)")

        if sorted_mbs:
            for mbs in sorted_mbs:
                times = by_mbs[mbs]
                mbs_total = sum(times)
                print(f"    BS={mbs:<3}  {len(times):>3} shapes  "
                      f"total={_fmt_time(mbs_total):>8}  "
                      f"avg={_fmt_time(mbs_total / len(times)):>6} / shape  "
                      f"min={_fmt_time(min(times)):>6}  max={_fmt_time(max(times)):>6}")

            if len(sorted_mbs) >= 2:
                lo, hi = sorted_mbs[0], sorted_mbs[-1]
                lo_total = sum(by_mbs[lo])
                hi_total = sum(by_mbs[hi])
                print(f"    \u2192 Smallest BS ({lo}): {_fmt_time(lo_total)}  |  "
                      f"Largest BS ({hi}): {_fmt_time(hi_total)}")
        print()

    # --- N/A shapes ---
    if no_tb:
        print(f"{'\u2500'*80}")
        print(f"  Shapes without T/B ratio ({len(no_tb)}):")
        for e in no_tb:
            t = f"T={e['tensile']:.0f}" if e["tensile"] else "T=N/A"
            b = f"B={e['bench']:.0f}" if e["bench"] else "B=N/A"
            a = f"A={e['api']:.0f}" if e["api"] else "A=N/A"
            print(f"    {e['model']:<25} mbs={e['mbs']:>2}  {e['layer']:<20} "
                  f"[{e['phase']}]  {t} {a} {b}")


if __name__ == "__main__":
    main()
