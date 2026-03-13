#!/usr/bin/env python3
"""
Tensile tuning + hipblaslt-bench comparison for BF16 dense GEMM shapes.

Shapes are derived from dense LLM architectures (Llama, Qwen, Mistral).
For each shape the script:
  1. Generates a per-shape Tensile YAML config from the template
  2. Runs Tensile to compile & benchmark kernels
  3. Runs hipblaslt-bench as a baseline comparison
  4. Produces a per-shape markdown report and a summary CSV

Usage:
  python3 run_shapes.py --list
  python3 run_shapes.py --run
  python3 run_shapes.py --run --filter "Llama-3.1-8B" --max-shapes 2
  python3 run_shapes.py --gen-only
  python3 run_shapes.py --compare-only
  python3 run_shapes.py --run --parallel 8
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import gen_all_shapes

WORKSPACE = Path(__file__).resolve().parent
TEMPLATE_YAML = WORKSPACE / "templates" / "bf16_gemm_gfx950.yaml"
OUTPUT_DIR = WORKSPACE / "tunning_results"
HIPBLASLT_BENCH = "/opt/rocm/bin/hipblaslt-bench"

# Tensile working directory — defaults to the hipblaslt submodule's tensilelite
TENSILE_WD = Path(os.environ.get(
    "TENSILE_WD",
    str(WORKSPACE / "hipblaslt" / "tensilelite"),
))

# ---------------------------------------------------------------------------
# Rotating buffer auto-sizing
# ---------------------------------------------------------------------------

_GFX950_LLC_MB = 256
_MIN_ROTATIONS = 3
_MAX_ROTATING_MB = 5120


def compute_rotating_buffer_mb(M, N, K, elem_bytes=2):
    tensor_set = (M * K + K * N + 2 * M * N) * elem_bytes
    tensor_set_mb = tensor_set / (1024 * 1024)
    needed_mb = max(tensor_set_mb * _MIN_ROTATIONS, _GFX950_LLC_MB * 2)
    return min(int(math.ceil(needed_mb)), _MAX_ROTATING_MB)


# ---------------------------------------------------------------------------
# MI4 → MI9 expansion
# ---------------------------------------------------------------------------

_MI_MAX_MT = 256
_MI_MAX_WAVETILE = 8
_MI_WAVEGROUP_COMBOS = [(2, 2), (4, 1), (1, 4)]


def _expand_mi4_for_shape(mi4_list, M, N, max_mt=_MI_MAX_MT, max_wt=_MI_MAX_WAVETILE):
    seen = set()
    result = []
    for mi4 in mi4_list:
        mi_M, mi_N, mi_K, mi_B = mi4
        max_bm = int(math.log2(mi_B)) if mi_B > 1 else 0
        for bm in range(max_bm + 1):
            bm_val = 2 ** bm
            for wm, wn in _MI_WAVEGROUP_COMBOS:
                for tt0 in range(1, max_wt + 1):
                    mt0 = mi_M * bm_val * tt0 * wm
                    if mt0 > max_mt or mt0 > M:
                        break
                    for tt1 in range(1, max_wt + 1):
                        mt1 = mi_N * tt1 * wn
                        if mt1 > max_mt or mt1 > N:
                            break
                        key = (mi_M, mi_N, mi_K, mi_B, bm_val, tt0, tt1, wm, wn)
                        if key not in seen:
                            seen.add(key)
                            result.append(list(key))
    return result


def _expand_mi_in_header(header, M, N):
    lines = header.split("\n")
    mi_start_idx = None
    mi_first_entry = None
    mi_last_entry = None
    mi4_list = []
    scanning_mi = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("- MatrixInstruction:"):
            mi_start_idx = i
            scanning_mi = True
            continue
        if scanning_mi:
            if stripped.startswith("- ["):
                m = re.search(r"\[([^\]]+)\]", stripped)
                if m:
                    vals = [int(x.strip()) for x in m.group(1).split(",")]
                    mi4_list.append(vals)
                    if mi_first_entry is None:
                        mi_first_entry = i
                    mi_last_entry = i
            elif stripped.startswith("#"):
                continue
            elif stripped:
                scanning_mi = False
                break

    if not mi4_list:
        return header, 0
    if len(mi4_list[0]) == 9:
        return header, len(mi4_list)
    if len(mi4_list[0]) != 4:
        return header, len(mi4_list)

    mi9_list = _expand_mi4_for_shape(mi4_list, M, N)
    entry_indent = lines[mi_first_entry][: len(lines[mi_first_entry]) - len(lines[mi_first_entry].lstrip())]
    new_mi_lines = []
    for mi9 in mi9_list:
        new_mi_lines.append(
            f"{entry_indent}- [{mi9[0]}, {mi9[1]},{mi9[2]}, {mi9[3]},  "
            f"{mi9[4]},   {mi9[5]}, {mi9[6]},  {mi9[7]},{mi9[8]} ]"
        )
    new_lines = lines[:mi_first_entry] + new_mi_lines + lines[mi_last_entry + 1:]
    return "\n".join(new_lines), len(mi9_list)


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def read_template(template_path=None):
    path = template_path or TEMPLATE_YAML
    with open(path) as f:
        lines = f.readlines()
    header_lines = []
    footer_lines = []
    found_ps = False
    skip_exact = False
    for line in lines:
        if "ProblemSizes:" in line:
            found_ps = True
            header_lines.append(line)
            skip_exact = True
            continue
        if skip_exact:
            if line.strip().startswith("- Exact:"):
                continue
            skip_exact = False
        if not found_ps:
            header_lines.append(line)
        else:
            footer_lines.append(line)
    return "".join(header_lines), "".join(footer_lines)


def gen_yaml(shape, header, footer, out_path):
    M, N, K = shape["M"], shape["N"], shape["K"]

    expanded_header, mi_count = _expand_mi_in_header(header, M, N)

    rot_mb = compute_rotating_buffer_mb(M, N, K)
    expanded_header = re.sub(
        r"RotatingBufferSize:\s*\d+",
        f"RotatingBufferSize: {rot_mb}",
        expanded_header)

    exact_line = f"          - Exact: [{M}, {N}, 1, {K}]"
    content = expanded_header + exact_line + "\n" + footer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    return out_path, mi_count


# ---------------------------------------------------------------------------
# Run Tensile
# ---------------------------------------------------------------------------

def run_tensile(yaml_path, case_dir, device=None):
    if device is not None:
        text = yaml_path.read_text()
        text = re.sub(r"Device:\s*\d+", f"Device: {device}", text)
        yaml_path.write_text(text)

    env = os.environ.copy()
    tensile_wd = TENSILE_WD
    rocisa_lib = str(tensile_wd / "build_tmp" / "tensilelite" / "rocisa" / "lib")
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + rocisa_lib
    env["TENSILELITE_ENABLE_AUTOBUILD"] = "ON"
    if device is not None:
        env["HIP_VISIBLE_DEVICES"] = str(device)
        env["CUDA_VISIBLE_DEVICES"] = str(device)

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    tensile_bin = tensile_wd / "Tensile" / "bin" / "Tensile"
    cmd = [str(tensile_bin), str(yaml_path), str(case_dir)]

    log_path = case_dir.parent / f"{case_dir.name}.tensile.log"
    print(f"  Tensile: running ...  (log: {log_path.name})")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, env=env, cwd=str(tensile_wd),
                                stdout=log_f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            elapsed = time.time() - t0
            hint = _tail_hint(log_path)
            print(f"\r  Tensile: {elapsed:6.0f}s  {hint[:60]:<60}", end="", flush=True)
            time.sleep(5)

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\r  Tensile: {status} in {elapsed:.0f}s{' '*40}")
    if not ok:
        print(f"    See full log: {log_path}")
    return ok


def _tail_hint(path, n=5):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(-read_size, 2)
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    lines = [l.strip() for l in tail.split("\n")
             if l.strip() and not l.strip().startswith("Tensile::WARNING")]
    for line in reversed(lines[-n:]):
        if any(k in line for k in ["Finding", "Generating", "Writing", "Assembling",
                                    "Compiling", "buildSource", "client", "BenchmarkData",
                                    "Solution", "Winner", "numSolutions", "exit"]):
            return line
    return lines[-1] if lines else ""


def parse_tensile_csv(case_dir):
    csv_files = list(case_dir.rglob("*CSVWinner.csv"))
    if not csv_files:
        return None
    csv_path = csv_files[0]
    best_gflops = 0.0
    best_time = 0.0
    best_name = ""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gf = float(row.get(" WinnerGFlops", 0) or row.get("WinnerGFlops", 0))
            except (ValueError, TypeError):
                gf = 0.0
            if gf > best_gflops:
                best_gflops = gf
                best_time = float(row.get(" WinnerTimeUS", 0) or row.get("WinnerTimeUS", 0))
                best_name = (row.get(" WinnerName", "") or row.get("WinnerName", "")).strip()
    if best_gflops == 0:
        return None
    return {
        "gflops": best_gflops,
        "tflops": best_gflops / 1000.0,
        "time_us": best_time,
        "winner": best_name,
    }


# ---------------------------------------------------------------------------
# hipblaslt-bench comparison
# ---------------------------------------------------------------------------

def run_hipblaslt_bench(M, N, K, iters=50, log_dir=None, device=None):
    if not Path(HIPBLASLT_BENCH).exists():
        print(f"  hipblaslt-bench: SKIPPED (binary not found)")
        return None

    rot_mb = compute_rotating_buffer_mb(M, N, K)
    cmd = [
        HIPBLASLT_BENCH,
        "-m", str(M), "-n", str(N), "-k", str(K),
        "--batch_count", "1",
        "--precision", "bf16_r",
        "--compute_type", "f32_r",
        "--transA", "T", "--transB", "N",
        "-i", str(iters),
        "-j", "30",
        "--rotating", str(rot_mb),
        "--print_kernel_info",
    ]
    bench_env = None
    if device is not None:
        bench_env = os.environ.copy()
        bench_env["HIP_VISIBLE_DEVICES"] = str(device)
        bench_env["CUDA_VISIBLE_DEVICES"] = str(device)
    print(f"  hipblaslt-bench: running ...", end="", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                env=bench_env)
        output = result.stdout + result.stderr

        if log_dir:
            log_path = Path(log_dir) / f"hipblaslt_bench_M{M}_N{N}_K{K}.log"
            with open(log_path, "w") as f:
                f.write(output)

        parsed = _parse_hipblaslt_output(output)
        if parsed["tflops"] is not None:
            print(f"\r  hipblaslt-bench: {parsed['tflops']:.1f} TFLOPS{' '*20}")
        else:
            print(f"\r  hipblaslt-bench: FAILED to parse output{' '*10}")
        return parsed
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"\r  hipblaslt-bench: ERROR ({e}){' '*20}")
        return None


def _parse_hipblaslt_output(output):
    result = {"tflops": None, "kernel_name": None, "time_us": None}
    lines = [l.strip() for l in output.split("\n") if l.strip()]

    header_idx = None
    for i, line in enumerate(lines):
        if "hipblaslt-Gflops" in line:
            header_idx = i
            break
    if header_idx is not None:
        headers = [h.strip() for h in lines[header_idx].split(",")]
        headers[0] = headers[0].split(":")[-1].strip()

        def _col(name):
            try:
                return headers.index(name)
            except ValueError:
                return None

        gflops_col = _col("hipblaslt-Gflops")
        us_col = _col("us") or _col("hipblaslt-us")

        if header_idx + 1 < len(lines):
            data = lines[header_idx + 1].split(",")
            if gflops_col is not None and len(data) > gflops_col:
                try:
                    result["tflops"] = float(data[gflops_col].strip()) / 1000.0
                except ValueError:
                    pass
            if us_col is not None and len(data) > us_col:
                try:
                    result["time_us"] = float(data[us_col].strip())
                except ValueError:
                    pass

    for line in lines:
        m = re.search(r'(Cijk_\S+)', line)
        if m:
            result["kernel_name"] = m.group(1)
            break
    return result


# ---------------------------------------------------------------------------
# Shape ID / reporting
# ---------------------------------------------------------------------------

def shape_id(s):
    return f"{s['model']}_{s['layer']}_mbs{s['mbs']}_M{s['M']}_N{s['N']}_K{s['K']}"


def _write_shape_report(row, path):
    t_tflops = row.get("tensile_tflops")
    t_us = row.get("tensile_time_us")
    t_kernel = row.get("tensile_winner")
    b_tflops = row.get("bench_tflops")
    b_us = row.get("bench_time_us")
    b_kernel = row.get("bench_kernel")

    def _v(val, fmt=".2f", suffix=""):
        return f"{val:{fmt}}{suffix}" if val is not None else "N/A"

    def _ratio(val):
        if val is not None and b_tflops is not None and b_tflops > 0:
            return f"{val / b_tflops:.1%}"
        return "N/A"

    lines = [
        f"# {row['model']} — {row['layer']}",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {row['model']} |",
        f"| Layer | {row['layer']} |",
        f"| MBS | {row['mbs']} |",
        f"| M | {row['M']} |",
        f"| N | {row['N']} |",
        f"| K | {row['K']} |",
        f"| FLOPs | {2 * row['M'] * row['N'] * row['K']:.3e} |",
        "",
        "## Results",
        "",
        "| Method | TFLOPS | Time (us) | vs hipblaslt-bench |",
        "|--------|-------:|----------:|-------------------:|",
        f"| **Tensile tuned** | {_v(t_tflops)} | {_v(t_us)} | {_ratio(t_tflops)} |",
        f"| **hipblaslt-bench** | {_v(b_tflops)} | {_v(b_us)} | — |",
        "",
        "## Kernels",
        "",
        f"- **Tensile winner**: `{t_kernel or 'N/A'}`",
        f"- **hipblaslt-bench**: `{b_kernel or 'N/A'}`",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _thread_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _process_one_shape(shape, idx, total, header, footer, out_dir, args, device):
    sid = shape_id(shape)
    case_dir = out_dir / sid
    yaml_path = out_dir / f"{sid}.yaml"

    tag = f"[{idx+1}/{total}] [GPU{device}]"
    _thread_print(f"\n{tag} {shape['model']} {shape['layer']}  "
                  f"MBS={shape['mbs']} M={shape['M']} N={shape['N']} K={shape['K']}")
    _thread_print(f"{'─'*60}")

    if not args.compare_only:
        _, mi_count = gen_yaml(shape, header, footer, yaml_path)
        rot_mb = compute_rotating_buffer_mb(shape["M"], shape["N"], shape["K"])
        _thread_print(f"  {tag} YAML: {yaml_path.name}  "
                      f"({mi_count} MI configs, RotBuf={rot_mb} MB)")

    tensile_result = None
    if args.run and not args.skip_tensile:
        ok = run_tensile(yaml_path, case_dir, device=device)
        if ok:
            tensile_result = parse_tensile_csv(case_dir)
            if tensile_result:
                _thread_print(f"  {tag} Tensile winner: {tensile_result['tflops']:.1f} TFLOPS  "
                              f"({tensile_result['time_us']:.1f} µs)")
            else:
                _thread_print(f"  {tag} Tensile: no valid winner found")
    elif args.compare_only:
        tensile_result = parse_tensile_csv(case_dir)

    bench_result = None
    if args.run or args.compare_only:
        bench_result = run_hipblaslt_bench(
            shape["M"], shape["N"], shape["K"],
            log_dir=out_dir, device=device,
        )

    row = {
        "model": shape["model"],
        "layer": shape["layer"],
        "mbs": shape["mbs"],
        "M": shape["M"],
        "N": shape["N"],
        "K": shape["K"],
        "tensile_tflops": tensile_result["tflops"] if tensile_result else None,
        "tensile_time_us": tensile_result["time_us"] if tensile_result else None,
        "tensile_winner": tensile_result["winner"] if tensile_result else None,
        "bench_tflops": bench_result["tflops"] if bench_result else None,
        "bench_time_us": bench_result["time_us"] if bench_result else None,
        "bench_kernel": bench_result["kernel_name"] if bench_result else None,
    }

    _write_shape_report(row, out_dir / f"{sid}.report.md")

    t = row["tensile_tflops"]
    b = row["bench_tflops"]
    parts = []
    if t is not None:
        parts.append(f"Tensile={t:.1f}")
    if b is not None:
        parts.append(f"Bench={b:.1f}")
    ratio = ""
    if t and b and b > 0:
        ratio = f"  (T/B={t/b:.0%})"
    _thread_print(f"  >> {' | '.join(parts)}{ratio}")

    return row


def print_report(results):
    print(f"\n{'='*110}")
    print("COMPARISON REPORT: Tensile tuned BF16 GEMM vs hipblaslt-bench")
    print(f"{'='*110}")
    hdr = (f"{'Model':<18} {'Layer':<14} {'MBS':>4} {'M':>6} {'N':>6} {'K':>6}  "
           f"{'Tensile':>10} {'Bench':>10} {'T/B':>8}")
    print(hdr)
    print("-" * 110)
    for r in results:
        t = r["tensile_tflops"]
        b = r["bench_tflops"]
        t_s = f"{t:.2f}" if t else "N/A"
        b_s = f"{b:.2f}" if b else "N/A"
        tb = f"{t/b:.2%}" if (t and b and b > 0) else "N/A"
        print(f"{r['model']:<18} {r['layer']:<14} {r['mbs']:>4} {r['M']:>6} "
              f"{r['N']:>6} {r['K']:>6}  {t_s:>10} {b_s:>10} {tb:>8}")
    print(f"{'='*110}")


def save_report_csv(results, path):
    if not results:
        return
    keys = ["model", "layer", "mbs", "M", "N", "K",
            "tensile_tflops", "tensile_time_us", "tensile_winner",
            "bench_tflops", "bench_time_us", "bench_kernel"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nReport saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="hipBLASLt BF16 GEMM Tensile tuning for dense LLM shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 run_shapes.py --list
              python3 run_shapes.py --run --filter "Llama-3.1-8B" --max-shapes 2
              python3 run_shapes.py --gen-only
              python3 run_shapes.py --compare-only
              python3 run_shapes.py --run --parallel 8
        """),
    )
    parser.add_argument("--list", action="store_true", help="List all shapes and exit")
    parser.add_argument("--run", action="store_true", help="Generate + run Tensile + compare")
    parser.add_argument("--gen-only", action="store_true", help="Only generate YAML configs")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run comparison on existing results")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter shapes by model name substring")
    parser.add_argument("--max-shapes", type=int, default=None,
                        help="Limit number of shapes to process")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--device", type=int, default=None,
                        help="Override GPU device index (single-GPU mode)")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Run N shapes in parallel on GPUs 0..N-1")
    parser.add_argument("--gpu-list", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g. '0,2,4,6')")
    parser.add_argument("--template", type=str, default=None,
                        help="Override YAML template path")
    parser.add_argument("--skip-tensile", action="store_true",
                        help="Skip Tensile runs (only run hipblaslt-bench)")

    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    shapes = gen_all_shapes(model_filter=args.filter)
    if args.max_shapes:
        shapes = shapes[:args.max_shapes]

    if args.list:
        print(f"{'#':>4}  {'Model':<18}  {'Layer':<14}  {'MBS':>4}  {'M':>6}  {'N':>6}  {'K':>6}")
        print("-" * 75)
        for i, s in enumerate(shapes):
            print(f"{i+1:>4}  {s['model']:<18}  {s['layer']:<14}  "
                  f"{s['mbs']:>4}  {s['M']:>6}  {s['N']:>6}  {s['K']:>6}")
        print(f"\nTotal shapes: {len(shapes)}")
        return

    if not (args.run or args.gen_only or args.compare_only):
        parser.print_help()
        return

    template_path = Path(args.template) if args.template else None
    header, footer = read_template(template_path)

    if args.gpu_list:
        gpu_ids = [int(x.strip()) for x in args.gpu_list.split(",")]
    elif args.device is not None:
        gpu_ids = [args.device]
    else:
        gpu_ids = list(range(args.parallel))

    n_gpus = len(gpu_ids)
    if n_gpus > 1:
        print(f"Parallel mode: {n_gpus} GPUs {gpu_ids}")

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(shapes)

    if n_gpus <= 1:
        device = gpu_ids[0] if gpu_ids else 0
        results = []
        for i, shape in enumerate(shapes):
            row = _process_one_shape(shape, i, total, header, footer,
                                     out_dir, args, device)
            results.append(row)
    else:
        results = [None] * total
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {}
            for i, shape in enumerate(shapes):
                device = gpu_ids[i % n_gpus]
                fut = pool.submit(_process_one_shape, shape, i, total,
                                  header, footer, out_dir, args, device)
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    _thread_print(f"ERROR processing shape {idx}: {e}")
                    results[idx] = {
                        "model": shapes[idx]["model"],
                        "layer": shapes[idx]["layer"],
                        "mbs": shapes[idx]["mbs"],
                        **{k: shapes[idx][k] for k in ("M", "N", "K")},
                        "tensile_tflops": None, "tensile_time_us": None,
                        "tensile_winner": None, "bench_tflops": None,
                        "bench_time_us": None, "bench_kernel": None,
                    }

    valid = [r for r in results if r is not None]
    if valid and (args.run or args.compare_only):
        print_report(valid)
        save_report_csv(valid, out_dir / "comparison_report.csv")


if __name__ == "__main__":
    main()
