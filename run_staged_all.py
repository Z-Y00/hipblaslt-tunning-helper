#!/usr/bin/env python3
"""Run staged search on all shapes from config.py.

Iterates through every unique (M, N, K, transA, transB) shape and runs
the 5-stage search on each. Skips shapes that already have a report.md.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from config import gen_all_shapes
from staged_search import _run_shape, STAGE_RESULTS_DIR


def _parse_devices(spec):
    """Parse device spec like '1,2,3' or '1-4' or '1,3-5' into a list of ints."""
    devices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            devices.extend(range(int(lo), int(hi) + 1))
        else:
            devices.append(int(part))
    return sorted(set(devices))


def _worker(gpu_id, shape_queue, results, total):
    """Worker: pull shapes from queue and run staged search on assigned GPU."""
    while True:
        item = shape_queue.get()
        if item is None:
            shape_queue.task_done()
            break
        idx, s = item
        M, N, K = s["M"], s["N"], s["K"]
        trans_a, trans_b = s["trans_a"], s["trans_b"]
        ta = "T" if trans_a else "N"
        tb = "T" if trans_b else "N"
        shape_id = f"{ta}{tb}_M{M}_N{N}_K{K}"
        label = f"{s.get('model','?')} {s.get('layer','?')} mbs{s.get('mbs','?')} {s.get('phase','?')}"

        print(f"\n[{idx}/{total}] GPU{gpu_id} {shape_id}  ({label})")

        t0 = time.time()
        try:
            _run_shape(M, N, K, trans_a, trans_b, gpu_id)
            results["done"] += 1
        except Exception as e:
            results["failed"] += 1
            print(f"  GPU{gpu_id} ERROR: {e}")
        elapsed = time.time() - t0
        print(f"  GPU{gpu_id} shape time: {elapsed/60:.1f} min")
        shape_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1",
                        help="GPU(s) to use: single id, comma-separated, or range. "
                             "E.g. '1' or '1,3,5' or '2-7' (default: 1)")
    parser.add_argument("--include-bwd", action="store_true", default=False,
                        help="Include backward pass shapes (default: fwd only)")
    parser.add_argument("--quick-sweep", action="store_true",
                        help="Only BS=1 per model")
    args = parser.parse_args()

    devices = _parse_devices(args.device)

    shapes = gen_all_shapes(include_bwd=args.include_bwd, quick_sweep=args.quick_sweep)

    seen = set()
    unique_shapes = []
    for s in shapes:
        key = (s["M"], s["N"], s["K"], s["trans_a"], s["trans_b"])
        if key not in seen:
            seen.add(key)
            unique_shapes.append(s)

    print(f"=== Staged Search: {len(unique_shapes)} unique shapes ===")
    print(f"  Devices: GPU {devices}")
    print(f"  Workers: {len(devices)}")
    print(f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Filter out shapes that already have reports
    pending = []
    skipped = 0
    for i, s in enumerate(unique_shapes):
        M, N, K = s["M"], s["N"], s["K"]
        trans_a, trans_b = s["trans_a"], s["trans_b"]
        ta = "T" if trans_a else "N"
        tb = "T" if trans_b else "N"
        shape_id = f"{ta}{tb}_M{M}_N{N}_K{K}"
        report = STAGE_RESULTS_DIR / shape_id / "report.md"
        if report.exists():
            skipped += 1
        else:
            pending.append((i + 1, s))

    print(f"  Pending: {len(pending)}  Skipped: {skipped} (already done)")

    results = {"done": 0, "failed": 0}
    t_total = time.time()

    if len(devices) == 1:
        # Single GPU — simple sequential loop
        for idx, s in pending:
            M, N, K = s["M"], s["N"], s["K"]
            trans_a, trans_b = s["trans_a"], s["trans_b"]
            ta = "T" if trans_a else "N"
            tb = "T" if trans_b else "N"
            shape_id = f"{ta}{tb}_M{M}_N{N}_K{K}"
            label = f"{s.get('model','?')} {s.get('layer','?')} mbs{s.get('mbs','?')} {s.get('phase','?')}"

            print(f"\n[{idx}/{len(unique_shapes)}] {shape_id}  ({label})")
            t0 = time.time()
            try:
                _run_shape(M, N, K, trans_a, trans_b, devices[0])
                results["done"] += 1
            except Exception as e:
                results["failed"] += 1
                print(f"  ERROR: {e}")
            print(f"  Shape time: {(time.time() - t0)/60:.1f} min")
    else:
        # Multi-GPU — one worker per GPU, shapes dispatched via queue
        q = Queue()
        for item in pending:
            q.put(item)
        for _ in devices:
            q.put(None)

        total = len(unique_shapes)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(_worker, gpu, q, results, total) for gpu in devices]
            for f in futures:
                f.result()

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  Done: {results['done']}  Skipped: {skipped}  Failed: {results['failed']}")
    print(f"  Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
