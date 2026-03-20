#!/usr/bin/env python3
"""
Tensile tuning + hipblaslt-bench comparison for dense GEMM shapes.

Supports BF16 and FP8 (OCP E4M3, mixed E4M3/E5M2) data types.

Shapes are derived from dense LLM architectures (Llama, Qwen, Mistral).
For each shape the script:
  1. Generates a per-shape Tensile YAML config from the template
  2. Runs Tensile to compile & benchmark kernels
  3. Runs hipblaslt-bench as a baseline comparison
  4. Produces a per-shape markdown report and a summary CSV

Usage:
  python3 run_shapes.py --list
  python3 run_shapes.py --run
  python3 run_shapes.py --run --dtype f8
  python3 run_shapes.py --run --dtype f8b8
  python3 run_shapes.py --run --filter "Llama-3.1-8B" --max-shapes 2
  python3 run_shapes.py --gen-only
  python3 run_shapes.py --compare-only
  python3 run_shapes.py --run --parallel 8
  python3 run_shapes.py --run --origami-top-n 0   # disable origami pruning
  python3 run_shapes.py --run --fwd-only
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
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from pathlib import Path
from typing import Optional

from config import gen_all_shapes

WORKSPACE = Path(__file__).resolve().parent
TEMPLATES = {
    "bf16": WORKSPACE / "templates" / "bf16_gemm_gfx950.yaml",
    "f8":   WORKSPACE / "templates" / "f8_gemm_gfx950.yaml",
    "f8b8": WORKSPACE / "templates" / "f8_gemm_gfx950.yaml",
}
OUTPUT_DIR = WORKSPACE / "tunning_results"
HIPBLASLT_BENCH = "/opt/rocm/bin/hipblaslt-bench"
API_BENCH = str(WORKSPACE / "test_hipblaslt_api")

# Tensile DataType codes per dtype flag
_TENSILE_DTYPE = {"bf16": "b", "f8": "F8", "f8b8": "F8B8"}
_TENSILE_DEST_DTYPE = {"bf16": "b", "f8": "b", "f8b8": "b"}

_BENCH_PRECISION = {
    "bf16": {"precision": "bf16_r"},
    "f8":   {"a_type": "f8_r", "b_type": "f8_r",
             "c_type": "bf16_r", "d_type": "bf16_r"},
    "f8b8": {"a_type": "f8_r", "b_type": "bf8_r",
             "c_type": "bf16_r", "d_type": "bf16_r"},
}

_FP8_DTYPES = {"f8", "f8b8"}

# Tensile working directory — prefer the build-tree Tensile (same version as
# TensileCreateLibrary used during hipBLASLt build) so that tuned solutions
# have compatible parameters.  Falls back to the local submodule.
_BUILD_TENSILE = WORKSPACE / "tmp_rebuild" / "rocm-libraries" / "projects" / "hipblaslt" / "tensilelite"
_LOCAL_TENSILE = WORKSPACE / "hipblaslt" / "tensilelite"
_DEFAULT_TENSILE = str(_BUILD_TENSILE if _BUILD_TENSILE.exists() else _LOCAL_TENSILE)
TENSILE_WD = Path(os.environ.get("TENSILE_WD", _DEFAULT_TENSILE))

# ---------------------------------------------------------------------------
# Rotating buffer auto-sizing
# ---------------------------------------------------------------------------

_GFX950_LLC_MB = 256
_MIN_ROTATIONS = 5
_MAX_ROTATING_MB = 8192
_FP8_ROTATION_SCALE = 5
_TENSILE_OVERHEAD_FACTOR = 1.25  # Tensile client reserves ~15-17% for alignment/metadata


def compute_rotating_buffer_mb(M, N, K, dtype="bf16"):
    in_bytes = 1 if dtype in _FP8_DTYPES else 2
    out_bytes = 2  # DestDataType is always BF16
    scale = _FP8_ROTATION_SCALE if dtype in _FP8_DTYPES else 1
    tensor_set = (M * K + K * N) * in_bytes + 2 * M * N * out_bytes
    tensor_set_mb = tensor_set / (1024 * 1024)
    if tensor_set_mb >= _GFX950_LLC_MB * 2:
        return 0
    needed_mb = max(tensor_set_mb * _MIN_ROTATIONS * scale, _GFX950_LLC_MB * 2)
    needed_mb *= _TENSILE_OVERHEAD_FACTOR
    return min(int(math.ceil(needed_mb)), _MAX_ROTATING_MB)


# ---------------------------------------------------------------------------
# MI4 → MI9 expansion
# ---------------------------------------------------------------------------

_MI_MAX_MT = 256
_MI_MAX_WAVETILE = 16
_MI_WAVEGROUP_COMBOS = [(2, 2), (4, 1), (1, 4)]

# ---------------------------------------------------------------------------
# Origami analytical tile ranking (optional, reduces search space)
# ---------------------------------------------------------------------------
_origami_available = False
_ORIGAMI_PKG = str(WORKSPACE / "origami" / "shared" / "origami" / "python")
if Path(_ORIGAMI_PKG).is_dir() and _ORIGAMI_PKG not in sys.path:
    sys.path.insert(0, _ORIGAMI_PKG)
try:
    import origami as _origami
    _origami_available = True
except ImportError:
    pass

_origami_hw = None


def _get_origami_hw():
    global _origami_hw
    if _origami_hw is None:
        _origami_hw = _origami.get_hardware_for_device(0)
    return _origami_hw


def _trans_label(trans):
    """Convert boolean transpose to single-char label (T or N)."""
    return "T" if trans else "N"


def _trans_code(trans_a, trans_b):
    """Return 3-char hipBLASLt layout code for A, B, C (C is always N)."""
    return _trans_label(trans_a) + _trans_label(trans_b) + "N"


def _origami_dtype(dtype="bf16"):
    """Map our dtype flag to origami data_type_t values for (a, b, output).

    For mixed FP8 (f8b8), we set per-operand types.  The mi_dtype is set
    to a_dtype since the MFMA instruction type follows A's format.
    """
    if not _origami_available:
        return None, None, None
    dt = _origami.data_type_t
    if dtype == "f8":
        return dt.Float8, dt.Float8, dt.BFloat16
    elif dtype == "f8b8":
        return dt.Float8, dt.BFloat8, dt.BFloat16
    return dt.BFloat16, dt.BFloat16, dt.BFloat16


def _origami_filter_mi9(mi9_list, M, N, K, depth_u_values, top_n=30,
                         trans_a=True, trans_b=False, dtype="bf16"):
    """Use Origami analytical model to rank MI9 configs and keep only the top N.

    For each MI9 entry, compute the macro tile (MT_M, MT_N) and cross with
    every DepthU value to build origami configs.  Rank them all, then return
    only the MI9 entries whose tiles appear in the top-N results.
    """
    if not _origami_available:
        return mi9_list

    hw = _get_origami_hw()
    ab_dt, b_dt, out_dt = _origami_dtype(dtype)

    prob = _origami.problem_t()
    prob.size = _origami.dim3_t(M, N, K)
    prob.batch = 1
    prob.a_transpose = _origami.transpose_t.T if trans_a else _origami.transpose_t.N
    prob.b_transpose = _origami.transpose_t.T if trans_b else _origami.transpose_t.N
    prob.a_dtype = ab_dt
    prob.b_dtype = b_dt
    prob.c_dtype = out_dt
    prob.d_dtype = out_dt
    prob.mi_dtype = ab_dt
    prob.a_mx_block_size = 0
    prob.b_mx_block_size = 0

    configs = []
    config_to_mi9_idx = []

    for idx, mi9 in enumerate(mi9_list):
        mi_M, mi_N, mi_K, mi_B, bm_val, tt0, tt1, wm, wn = mi9
        mt_m = mi_M * bm_val * tt0 * wm
        mt_n = mi_N * tt1 * wn
        for du in depth_u_values:
            c = _origami.config_t()
            c.mt = _origami.dim3_t(mt_m, mt_n, du)
            c.mi = _origami.dim3_t(mi_M, mi_N, mi_K)
            c.occupancy = 1
            configs.append(c)
            config_to_mi9_idx.append(idx)

    if not configs:
        return mi9_list

    ranked = _origami.select_topk_configs(prob, hw, configs, top_n)

    surviving_tiles = set()
    for r in ranked:
        mt = r.config.mt
        mi = r.config.mi
        surviving_tiles.add((mt.m, mt.n, mi.m, mi.n, mi.k))

    kept = []
    kept_set = set()
    for idx, mi9 in enumerate(mi9_list):
        mi_M, mi_N, mi_K, mi_B, bm_val, tt0, tt1, wm, wn = mi9
        mt_m = mi_M * bm_val * tt0 * wm
        mt_n = mi_N * tt1 * wn
        tile_key = (mt_m, mt_n, mi_M, mi_N, mi_K)
        if tile_key in surviving_tiles:
            k = tuple(mi9)
            if k not in kept_set:
                kept_set.add(k)
                kept.append(mi9)

    return kept if kept else mi9_list


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


def _parse_depth_u(header):
    """Extract DepthU values from the YAML header, e.g. '- DepthU: [32,64]'."""
    m = re.search(r"-\s*DepthU:\s*\[([^\]]+)\]", header)
    if m:
        return [int(x.strip()) for x in m.group(1).split(",")]
    return [64]


def _expand_mi_in_header(header, M, N, K=0, origami_top_n=0,
                          trans_a=True, trans_b=False, dtype="bf16"):
    """Find MatrixInstruction entries in *header*.  If they are 4-element,
    expand to 9-element entries sized for the given problem (M × N).

    When *origami_top_n* > 0 and origami is available, the expanded MI9
    list is pruned to only those tiles that origami ranks in the top N.
    """
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

    if origami_top_n > 0 and _origami_available and K > 0:
        depth_u_values = _parse_depth_u(header)
        before = len(mi9_list)
        mi9_list = _origami_filter_mi9(
            mi9_list, M, N, K, depth_u_values, top_n=origami_top_n,
            trans_a=trans_a, trans_b=trans_b, dtype=dtype,
        )
        if before != len(mi9_list):
            print(f"    [origami] pruned MI9: {before} -> {len(mi9_list)}  "
                  f"(top {origami_top_n} tiles)")

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
    path = template_path or TEMPLATES["bf16"]
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


def gen_yaml(shape, header, footer, out_path, origami_top_n=0, dtype="bf16"):
    # Swap M↔N: PyTorch row-major (M,K)@(N,K).T uses BLAS col-major m=N, n=M
    M, N, K = shape["N"], shape["M"], shape["K"]
    trans_a = bool(shape.get("trans_a", True))
    trans_b = bool(shape.get("trans_b", False))

    expanded_header, mi_count = _expand_mi_in_header(
        header, M, N, K=K, origami_top_n=origami_top_n,
        trans_a=trans_a, trans_b=trans_b, dtype=dtype,
    )

    rot_mb = compute_rotating_buffer_mb(M, N, K, dtype=dtype)
    expanded_header = re.sub(
        r"RotatingBufferSize:\s*\d+",
        f"RotatingBufferSize: {rot_mb}",
        expanded_header)

    tensile_dt = _TENSILE_DTYPE[dtype]
    dest_dt = _TENSILE_DEST_DTYPE[dtype]
    expanded_header = re.sub(
        r"(DataType:\s*)\S+", rf"\g<1>{tensile_dt}", expanded_header, count=1)
    expanded_header = re.sub(
        r"(DestDataType:\s*)\S+", rf"\g<1>{dest_dt}", expanded_header, count=1)

    expanded_header = re.sub(
        r"(TransposeA:\s*)[01]",
        rf"\g<1>{1 if trans_a else 0}",
        expanded_header, count=1)
    expanded_header = re.sub(
        r"(TransposeB:\s*)[01]",
        rf"\g<1>{1 if trans_b else 0}",
        expanded_header, count=1)

    is_batched = bool(re.search(r"Batched:\s*[Tt]rue", expanded_header))
    if is_batched:
        exact_line = f"          - Exact: [{M}, {N}, 1, {K}]"
    else:
        exact_line = f"          - Exact: [{M}, {N}, {K}]"
    content = expanded_header + exact_line + "\n" + footer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    return out_path, mi_count


# ---------------------------------------------------------------------------
# Run Tensile
# ---------------------------------------------------------------------------


def _find_rocisa_lib(tensile_wd: Path) -> str:
    """Locate the rocisa shared library directory for *tensile_wd*.

    When using the build-tree Tensile the pre-built rocisa lives under the
    hipBLASLt build directory rather than a local build_tmp.
    """
    candidates = [
        # init_build.sh puts rocisa here (invoke build-client)
        tensile_wd / "build_tmp" / "tensilelite" / "rocisa" / "lib",
        # hipblaslt cmake build puts rocisa here
        tensile_wd.parent / "build" / "tensilelite" / "rocisa" / "lib",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Fallback — let TENSILELITE_ENABLE_AUTOBUILD handle it
    return str(candidates[-1])


def run_tensile(yaml_path, case_dir, device=None):
    if device is not None:
        text = yaml_path.read_text()
        # HIP_VISIBLE_DEVICES remaps the physical GPU to index 0
        text = re.sub(r"Device:\s*\d+", "Device: 0", text)
        yaml_path.write_text(text)

    env = os.environ.copy()
    tensile_wd = TENSILE_WD
    rocisa_lib = _find_rocisa_lib(tensile_wd)
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
            time.sleep(30)

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
    archive = case_dir.parent / (case_dir.name + ".tar.zst")
    if not case_dir.is_dir() and archive.is_file():
        for pattern in ["CSVWinner.csv", "00_Final.csv"]:
            try:
                proc = subprocess.run(
                    f"zstd -dc {archive} | tar tf - | grep {pattern}",
                    shell=True, capture_output=True, text=True, timeout=30,
                )
                csv_member = proc.stdout.strip().split("\n")[0] if proc.stdout.strip() else ""
                if csv_member:
                    extract = subprocess.run(
                        f"zstd -dc {archive} | tar xf - -O {csv_member}",
                        shell=True, capture_output=True, text=True, timeout=60,
                    )
                    if extract.returncode == 0 and extract.stdout:
                        import io
                        return _parse_csv_content(io.StringIO(extract.stdout))
            except Exception:
                pass
        return None

    csv_files = list(case_dir.rglob("*CSVWinner.csv"))
    if not csv_files:
        csv_files = list(case_dir.rglob("00_Final.csv"))
    if not csv_files:
        return None
    with open(csv_files[0]) as f:
        return _parse_csv_content(f)


def _parse_csv_content(f):
    """Parse a CSVWinner file object and return the best result."""
    best_gflops = 0.0
    best_time = 0.0
    best_name = ""
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
# Post-tuning cleanup — remove rebuildable artifacts, compress the rest
# ---------------------------------------------------------------------------

def _cleanup_shape_dir(case_dir: Path, compress: bool = True):
    """Remove rebuildable build artifacts and optionally compress what remains.

    Deletes .o (object files), .s (assembly), .co (code objects), and
    .hsaco (GPU binaries) — all regenerated by re-running Tensile.
    Then compresses the remaining directory into a .tar.zst archive
    and removes the uncompressed directory.

    Keeps: CSVWinner.csv/yaml, TensileLibrary.yaml, generated source headers.
    """
    if not case_dir.is_dir():
        return

    removed = 0
    for ext in ("*.o", "*.s", "*.co", "*.hsaco"):
        for f in case_dir.rglob(ext):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass

    # Remove empty directories left behind
    for dirpath, dirnames, filenames in os.walk(case_dir, topdown=False):
        p = Path(dirpath)
        if p != case_dir and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass

    if not compress:
        if removed:
            print(f"    [cleanup] removed {removed} rebuildable files")
        return

    archive = case_dir.parent / (case_dir.name + ".tar.zst")
    try:
        result = subprocess.run(
            ["tar", "cf", "-", "-C", str(case_dir.parent), case_dir.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
        if result.returncode == 0:
            zstd = subprocess.run(
                ["zstd", "-3", "--rm", "-o", str(archive)],
                input=result.stdout, capture_output=True, timeout=600,
            )
            if zstd.returncode == 0:
                shutil.rmtree(case_dir, ignore_errors=True)
                sz = archive.stat().st_size
                print(f"    [cleanup] compressed to {archive.name} "
                      f"({sz / (1024*1024):.1f} MB)")
                return
        print(f"    [cleanup] compression failed, kept uncompressed dir")
    except Exception as e:
        print(f"    [cleanup] compression error: {e}")


# ---------------------------------------------------------------------------
# hipblaslt-bench comparison
# ---------------------------------------------------------------------------

def run_hipblaslt_bench(M, N, K, trans_a=True, trans_b=False,
                        iters=50, device=None, dtype="bf16"):
    if not Path(HIPBLASLT_BENCH).exists():
        print(f"  hipblaslt-bench: SKIPPED (binary not found)")
        return None

    rot_mb = compute_rotating_buffer_mb(M, N, K, dtype=dtype)
    cmd = [
        HIPBLASLT_BENCH,
        "-m", str(M), "-n", str(N), "-k", str(K),
    ]
    prec = _BENCH_PRECISION[dtype]
    if "precision" in prec:
        cmd += ["--precision", prec["precision"]]
    if "a_type" in prec:
        cmd += ["--a_type", prec["a_type"], "--b_type", prec["b_type"]]
    if "c_type" in prec:
        cmd += ["--c_type", prec["c_type"], "--d_type", prec["d_type"]]
    if dtype in _FP8_DTYPES:
        cmd += ["--compute_type", "f32_r", "--scaleA", "1", "--scaleB", "1"]
    else:
        cmd += ["--compute_type", "f32_r"]
    cmd += [
        "--transA", _trans_label(trans_a),
        "--transB", _trans_label(trans_b),
        "-i", str(iters),
        "-j", "30",
        "--rotating", str(rot_mb),
        "--use_gpu_timer",
        "--flush",
        "--initialization", "trig_float",
        "--print_kernel_info",
    ]
    bench_env = None
    if device is not None:
        bench_env = os.environ.copy()
        bench_env["HIP_VISIBLE_DEVICES"] = str(device)
        bench_env["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd_str = " ".join(cmd)
    print(f"  hipblaslt-bench: running ...", end="", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                env=bench_env)
        output = result.stdout + result.stderr
        parsed = _parse_hipblaslt_output(output)
        parsed["raw_output"] = output.strip()
        parsed["cmd"] = cmd_str
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
# API bench (Tensile-aligned timing via hipblasLtMatmul)
# ---------------------------------------------------------------------------

def run_api_bench(M, N, K, trans_a=True, trans_b=False,
                  device=None, warmup=30, iters=50, rotating_mb=None,
                  dtype="bf16"):
    """Run test_hipblaslt_api in single-shape CSV mode.

    Uses Tensile-aligned timing (single event pair, total/N average,
    random [-3,3] init).  When rotating_mb is None (default), the
    rotating buffer size is computed automatically via
    compute_rotating_buffer_mb() to match Tensile's cold-cache behavior.
    Pass rotating_mb=0 to force warm L2.
    """
    if not Path(API_BENCH).exists():
        print(f"  api-bench: SKIPPED (binary not found at {API_BENCH})")
        return None

    if rotating_mb is None:
        rotating_mb = compute_rotating_buffer_mb(M, N, K, dtype=dtype)

    ta = "T" if trans_a else "N"
    tb = "T" if trans_b else "N"
    cmd = [
        API_BENCH,
        "-m", str(M), "-n", str(N), "-k", str(K),
        "--transA", ta, "--transB", tb,
        "--warmup", str(warmup), "--iters", str(iters),
        "--rotating", str(rotating_mb),
        "--csv",
    ]
    env = None
    if device is not None:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(device)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        cmd += ["--device", "0"]

    cache_label = "cold" if rotating_mb > 0 else "warm"
    print(f"  api-bench: running (rotating={rotating_mb} MB, {cache_label} L2) ...",
          end="", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120, env=env)
        line = result.stdout.strip()
        if result.returncode != 0 or not line:
            err = result.stderr.strip()[:120]
            print(f"\r  api-bench: FAILED (exit {result.returncode}) {err}")
            return None
        parts = line.split(",", 2)
        avg_us = float(parts[0])
        tflops = float(parts[1])
        kernel = parts[2] if len(parts) > 2 else ""
        print(f"\r  api-bench: {tflops:.1f} TFLOPS  ({avg_us:.1f} µs, "
              f"rotating={rotating_mb} MB){' '*10}")
        return {"tflops": tflops, "time_us": avg_us, "kernel_name": kernel}
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"\r  api-bench: ERROR ({e}){' '*20}")
        return None


# ---------------------------------------------------------------------------
# Shape ID / reporting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernel name parsing & diff
# ---------------------------------------------------------------------------

def _parse_kernel_params(kernel_name):
    """Parse a Tensile kernel name into an ordered dict of parameter=value pairs.

    Handles: KEY<digits>, KEYn<digits> (n = negative), bare-alpha tokens,
    and multi-part values like WG32_8_1 and MIWT8_8.
    """
    if not kernel_name:
        return {}
    prefix_end = kernel_name.find("_MT")
    if prefix_end == -1:
        return {}
    tag_section = kernel_name[prefix_end + 1:]
    tokens = tag_section.split("_")
    params = {}
    last_key = None
    for tok in tokens:
        m = re.match(r'^([A-Za-z]+?)(n?\d.*)$', tok)
        if m:
            last_key = m.group(1)
            params[last_key] = m.group(2)
        elif re.match(r'^\d+$', tok) and last_key:
            params[last_key] += "_" + tok
        else:
            last_key = tok
            params[tok] = ""
    return params


def _kernel_param_diff(tensile_kernel, bench_kernel):
    """Return (shared_diffs, tensile_only, bench_only) for parameters that differ.

    shared_diffs: [(param, t_val, b_val)] - params in both but different values
    tensile_only: [(param, val)] - params only in Tensile kernel
    bench_only:   [(param, val)] - params only in bench kernel
    """
    tp = _parse_kernel_params(tensile_kernel)
    bp = _parse_kernel_params(bench_kernel)
    shared_diffs = []
    tensile_only = []
    bench_only = []
    all_keys = list(dict.fromkeys(list(tp.keys()) + list(bp.keys())))
    for k in all_keys:
        tv = tp.get(k)
        bv = bp.get(k)
        if tv == bv:
            continue
        if tv is not None and bv is not None:
            shared_diffs.append((k, tv, bv))
        elif tv is not None:
            tensile_only.append((k, tv))
        else:
            bench_only.append((k, bv))
    return shared_diffs, tensile_only, bench_only


def shape_id(s):
    code = _trans_code(s.get("trans_a", True), s.get("trans_b", False))
    phase = s.get("phase", "fwd")
    return (f"{s['model']}_{s['layer']}_mbs{s['mbs']}"
            f"_{phase}_{code}_M{s['M']}_N{s['N']}_K{s['K']}")


def _write_shape_report(row, path):
    t_tflops = row.get("tensile_tflops")
    t_us = row.get("tensile_time_us")
    t_kernel = row.get("tensile_winner")
    b_tflops = row.get("bench_tflops")
    b_us = row.get("bench_time_us")
    b_kernel = row.get("bench_kernel")
    a_tflops = row.get("api_tflops")
    a_us = row.get("api_time_us")
    a_kernel = row.get("api_kernel")

    def _v(val, fmt=".2f", suffix=""):
        return f"{val:{fmt}}{suffix}" if val is not None else "N/A"

    def _ratio(val, base):
        if val is not None and base is not None and base > 0:
            return f"{val / base:.1%}"
        return "N/A"

    phase = row.get("phase", "fwd")
    lines = [
        f"# {row['model']} — {row['layer']} [{phase}]",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {row['model']} |",
        f"| Layer | {row['layer']} |",
        f"| Phase | {phase} |",
        f"| MBS | {row['mbs']} |",
        f"| Trans (ABC) | {row.get('trans', 'TNN')} |",
        f"| M | {row['M']} |",
        f"| N | {row['N']} |",
        f"| K | {row['K']} |",
        f"| FLOPs | {2 * row['M'] * row['N'] * row['K']:.3e} |",
        "",
        "## Results",
        "",
        "| Method | TFLOPS | Time (us) | vs Bench | vs API |",
        "|--------|-------:|----------:|---------:|-------:|",
        f"| **Tensile tuned** | {_v(t_tflops)} | {_v(t_us)} "
        f"| {_ratio(t_tflops, b_tflops)} | {_ratio(t_tflops, a_tflops)} |",
        f"| **API bench (installed)** | {_v(a_tflops)} | {_v(a_us)} "
        f"| {_ratio(a_tflops, b_tflops)} | — |",
        f"| **hipblaslt-bench (stock)** | {_v(b_tflops)} | {_v(b_us)} | — | — |",
        "",
    ]
    if t_tflops and a_tflops and a_tflops > 0:
        lines.append(f"> **Tuned/API ratio (gate metric):** {t_tflops/a_tflops:.2%}")
        lines.append("")
    elif t_tflops and b_tflops and b_tflops > 0:
        lines.append(f"> **Tuned/Bench ratio (fallback):** {t_tflops/b_tflops:.2%}")
        lines.append("")

    lines += [
        "## Kernels",
        "",
        f"- **Tensile winner**: `{t_kernel or 'N/A'}`",
        f"- **API bench (installed)**: `{a_kernel or 'N/A'}`",
        f"- **hipblaslt-bench**: `{b_kernel or 'N/A'}`",
        "",
    ]

    bench_cmd = row.get("bench_cmd")
    bench_raw = row.get("bench_raw")
    if bench_cmd:
        lines += [
            "## hipblaslt-bench",
            "",
            "```bash",
            bench_cmd,
            "```",
            "",
        ]
        if bench_raw:
            lines += [
                "<details><summary>Raw output</summary>",
                "",
                "```",
                bench_raw,
                "```",
                "",
                "</details>",
                "",
            ]

    shared_diffs, t_only, b_only = _kernel_param_diff(t_kernel, b_kernel)
    if shared_diffs or t_only or b_only:
        lines += [
            "## Parameter Differences (Tensile vs hipblaslt-bench)",
            "",
        ]
        if shared_diffs:
            lines += [
                "| Parameter | Tensile | hipblaslt-bench |",
                "|-----------|---------|-----------------|",
            ]
            for param, tv, bv in shared_diffs:
                lines.append(f"| {param} | {tv} | {bv} |")
            lines.append("")
        if t_only:
            lines.append("**Tensile-only params:** "
                         + ", ".join(f"`{p}={v}`" if v else f"`{p}`"
                                     for p, v in t_only))
            lines.append("")
        if b_only:
            lines.append("**hipblaslt-bench-only params:** "
                         + ", ".join(f"`{p}={v}`" if v else f"`{p}`"
                                     for p, v in b_only))
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Resume: parse existing report.md to skip already-completed shapes
# ---------------------------------------------------------------------------

def _parse_existing_report(report_path: Path) -> Optional[dict]:
    """Read an existing .report.md and extract result fields.

    Returns a row dict if the report contains a valid Tensile TFLOPS value
    (indicating tuning completed), or None if missing / incomplete.
    """
    if not report_path.is_file():
        return None
    try:
        text = report_path.read_text()
    except OSError:
        return None

    row = {}

    # Parse parameter table
    for key, pattern in [
        ("model", r"\| Model \| (.+?) \|"),
        ("layer", r"\| Layer \| (.+?) \|"),
        ("phase", r"\| Phase \| (.+?) \|"),
        ("mbs", r"\| MBS \| (\d+) \|"),
        ("trans", r"\| Trans \(ABC\) \| (\w+) \|"),
        ("M", r"\| M \| (\d+) \|"),
        ("N", r"\| N \| (\d+) \|"),
        ("K", r"\| K \| (\d+) \|"),
    ]:
        m = re.search(pattern, text)
        if m:
            val = m.group(1).strip()
            row[key] = int(val) if key in ("mbs", "M", "N", "K") else val

    # Parse results table — look for Tensile tuned row
    # Format: | **Tensile tuned** | 1733.63 | 554.95 | ...
    m = re.search(r"\*\*Tensile tuned\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if m:
        row["tensile_tflops"] = float(m.group(1))
        row["tensile_time_us"] = float(m.group(2))
    else:
        row["tensile_tflops"] = None
        row["tensile_time_us"] = None

    m = re.search(r"\*\*API bench.*?\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if m:
        row["api_tflops"] = float(m.group(1))
        row["api_time_us"] = float(m.group(2))
    else:
        row["api_tflops"] = None
        row["api_time_us"] = None

    m = re.search(r"\*\*hipblaslt-bench.*?\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if m:
        row["bench_tflops"] = float(m.group(1))
        row["bench_time_us"] = float(m.group(2))
    else:
        row["bench_tflops"] = None
        row["bench_time_us"] = None

    # Parse kernel names
    m = re.search(r"\*\*Tensile winner\*\*:\s*`([^`]+)`", text)
    row["tensile_winner"] = m.group(1) if m and m.group(1) != "N/A" else None

    m = re.search(r"\*\*API bench.*?\*\*:\s*`([^`]+)`", text)
    row["api_kernel"] = m.group(1) if m and m.group(1) != "N/A" else None

    m = re.search(r"\*\*hipblaslt-bench\*\*:\s*`([^`]+)`", text)
    row["bench_kernel"] = m.group(1) if m and m.group(1) != "N/A" else None

    # Parse bench command
    m = re.search(r"## hipblaslt-bench\n+```bash\n(.+?)\n```", text, re.DOTALL)
    if not m:
        m = re.search(r"## Reproduce hipblaslt-bench\n+```bash\n(.+?)\n```", text, re.DOTALL)
    row["bench_cmd"] = m.group(1).strip() if m else None
    row["bench_raw"] = None

    # Consider valid if Tensile TFLOPS is present
    if row.get("tensile_tflops") is not None:
        return row
    return None


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
    dtype = getattr(args, "dtype", "bf16") or "bf16"

    tag = f"[{idx+1}/{total}] [GPU{device}]"
    code = _trans_code(shape.get("trans_a", True), shape.get("trans_b", False))
    phase = shape.get("phase", "fwd")

    # Resume: skip shapes with valid existing results
    if not getattr(args, "force", False):
        report_path = out_dir / f"{sid}.report.md"
        cached = _parse_existing_report(report_path)
        if cached is not None:
            t = cached.get("tensile_tflops")
            a = cached.get("api_tflops")
            b = cached.get("bench_tflops")
            parts = []
            if t is not None:
                parts.append(f"Tensile={t:.1f}")
            if a is not None:
                parts.append(f"API={a:.1f}")
            if b is not None:
                parts.append(f"Bench={b:.1f}")
            _thread_print(f"\n{tag} {shape['model']} {shape['layer']} [{phase}]  "
                          f"MBS={shape['mbs']} M={shape['M']} N={shape['N']} K={shape['K']} "
                          f"({code}) [{dtype}]  [SKIP — cached]")
            _thread_print(f"  >> {' | '.join(parts)}")
            return cached

    _thread_print(f"\n{tag} {shape['model']} {shape['layer']} [{phase}]  "
                  f"MBS={shape['mbs']} M={shape['M']} N={shape['N']} K={shape['K']} "
                  f"({code}) [{dtype}]")
    _thread_print(f"{'─'*60}")

    if not args.compare_only:
        origami_n = getattr(args, "origami_top_n", 0) or 0
        _, mi_count = gen_yaml(shape, header, footer, yaml_path,
                               origami_top_n=origami_n, dtype=dtype)
        rot_mb = compute_rotating_buffer_mb(shape["M"], shape["N"], shape["K"],
                                            dtype=dtype)
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
    api_result = None
    if args.run or args.compare_only:
        # hipblaslt-bench expects BLAS col-major dims: m=N, n=M
        bench_result = run_hipblaslt_bench(
            shape["N"], shape["M"], shape["K"],
            trans_a=shape.get("trans_a", True),
            trans_b=shape.get("trans_b", False),
            device=device, dtype=dtype,
        )
        # API bench: same col-major dims, Tensile-aligned timing + rotating
        if dtype == "bf16":
            api_result = run_api_bench(
                shape["N"], shape["M"], shape["K"],
                trans_a=shape.get("trans_a", True),
                trans_b=shape.get("trans_b", False),
                device=device, dtype=dtype,
            )

    row = {
        "model": shape["model"],
        "layer": shape["layer"],
        "phase": shape.get("phase", "fwd"),
        "mbs": shape["mbs"],
        "trans": _trans_code(shape.get("trans_a", True), shape.get("trans_b", False)),
        "M": shape["M"],
        "N": shape["N"],
        "K": shape["K"],
        "tensile_tflops": tensile_result["tflops"] if tensile_result else None,
        "tensile_time_us": tensile_result["time_us"] if tensile_result else None,
        "tensile_winner": tensile_result["winner"] if tensile_result else None,
        "bench_tflops": bench_result["tflops"] if bench_result else None,
        "bench_time_us": bench_result["time_us"] if bench_result else None,
        "bench_kernel": bench_result["kernel_name"] if bench_result else None,
        "bench_cmd": bench_result.get("cmd") if bench_result else None,
        "bench_raw": bench_result.get("raw_output") if bench_result else None,
        "api_tflops": api_result["tflops"] if api_result else None,
        "api_time_us": api_result["time_us"] if api_result else None,
        "api_kernel": api_result["kernel_name"] if api_result else None,
    }

    _write_shape_report(row, out_dir / f"{sid}.report.md")

    t = row["tensile_tflops"]
    b = row["bench_tflops"]
    a = row["api_tflops"]
    parts = []
    if t is not None:
        parts.append(f"Tensile={t:.1f}")
    if a is not None:
        parts.append(f"API={a:.1f}")
    if b is not None:
        parts.append(f"Bench={b:.1f}")
    ratio = ""
    if t and a and a > 0:
        ratio = f"  (T/A={t/a:.0%})"
    elif t and b and b > 0:
        ratio = f"  (T/B={t/b:.0%})"
    _thread_print(f"  >> {' | '.join(parts)}{ratio}")

    tw = row.get("tensile_winner")
    ak = row.get("api_kernel")
    bk = row.get("bench_kernel")
    if tw:
        _thread_print(f"  >> Tensile kernel: {tw}")
    if ak:
        _thread_print(f"  >> API kernel:     {ak}")
    if bk:
        _thread_print(f"  >> Bench kernel:   {bk}")

    if not getattr(args, "no_cleanup", False) and case_dir.is_dir():
        compress = not getattr(args, "no_compress", False)
        _cleanup_shape_dir(case_dir, compress=compress)

    return row


def print_report(results, dtype="bf16"):
    dtype_label = dtype.upper()
    print(f"\n{'='*160}")
    print(f"COMPARISON REPORT: Tensile tuned {dtype_label} GEMM vs stock baseline")
    print(f"{'='*160}")
    hdr = (f"{'Model':<18} {'Layer':<14} {'Phase':<6} {'MBS':>4} {'Trans':>5} "
           f"{'M':>6} {'N':>6} {'K':>6}  "
           f"{'Tensile':>10} {'API':>10} {'Bench':>10} {'T/A':>8} {'T/B':>8}")
    print(hdr)
    print("-" * 140)
    for r in results:
        t = r["tensile_tflops"]
        a = r.get("api_tflops")
        b = r["bench_tflops"]
        t_s = f"{t:.2f}" if t else "N/A"
        a_s = f"{a:.2f}" if a else "N/A"
        b_s = f"{b:.2f}" if b else "N/A"
        ta_ratio = f"{t/a:.2%}" if (t and a and a > 0) else "N/A"
        tb_ratio = f"{t/b:.2%}" if (t and b and b > 0) else "N/A"
        print(f"{r['model']:<18} {r['layer']:<14} {r.get('phase', 'fwd'):<6} {r['mbs']:>4} "
              f"{r.get('trans', 'TNN'):>5} "
              f"{r['M']:>6} {r['N']:>6} {r['K']:>6}  "
              f"{t_s:>10} {a_s:>10} {b_s:>10} {ta_ratio:>8} {tb_ratio:>8}")
    print(f"{'='*140}")
    print(f"  T/A = Tensile tuned / API bench (same-methodology gate metric)")
    print(f"  T/B = Tensile tuned / hipblaslt-bench stock")


def save_report_csv(results, path):
    if not results:
        return
    keys = ["model", "layer", "phase", "mbs", "trans", "M", "N", "K",
            "tensile_tflops", "tensile_time_us", "tensile_winner",
            "api_tflops", "api_time_us", "api_kernel",
            "bench_tflops", "bench_time_us", "bench_kernel"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nReport saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="hipBLASLt GEMM Tensile tuning for dense LLM shapes "
                    "(BF16 / FP8 OCP / FP8 mixed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 run_shapes.py --list
              python3 run_shapes.py --run --filter "Llama-3.1-8B" --max-shapes 2
              python3 run_shapes.py --run --dtype f8
              python3 run_shapes.py --run --dtype f8b8
              python3 run_shapes.py --gen-only
              python3 run_shapes.py --compare-only
              python3 run_shapes.py --run --parallel 8
              python3 run_shapes.py --run --origami-top-n 0   # disable origami
              python3 run_shapes.py --run --fwd-only
        """),
    )
    parser.add_argument("--list", action="store_true", help="List all shapes and exit")
    parser.add_argument("--run", action="store_true", help="Generate + run Tensile + compare")
    parser.add_argument("--gen-only", action="store_true", help="Only generate YAML configs")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run comparison on existing results")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "f8", "f8b8"],
                        help="Data type: bf16, f8 (OCP E4M3), f8b8 (mixed "
                             "E4M3 A / E5M2 B). Default: bf16")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter shapes by model name substring")
    parser.add_argument("--filter-layer", type=str, default=None,
                        help="Filter shapes by layer name substring (e.g. 'lm_head')")
    parser.add_argument("--filter-mbs", type=str, default=None,
                        help="Comma-separated MBS values to include (e.g. '4' or '1,4,8')")
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
    parser.add_argument("--origami-top-n", type=int, default=30, metavar="N",
                        help="Use Origami analytical model to prune MI configs "
                             "to the top N tiles per shape (default 30, 0 = disabled)")
    parser.add_argument("--fwd-only", action="store_true",
                        help="Only include forward GEMMs (TN); skip backward "
                             "grad_a (TT) and grad_b (NT)")
    parser.add_argument("--force", action="store_true",
                        help="Re-tune all shapes even if valid results exist")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip post-tuning cleanup (keep all build artifacts)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Delete .o/.co/.hsaco but don't compress the shape dir")

    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve() / args.dtype

    shapes = gen_all_shapes(model_filter=args.filter,
                            include_bwd=not args.fwd_only)
    if args.filter_layer:
        shapes = [s for s in shapes if args.filter_layer.lower() in s["layer"].lower()]
    if args.filter_mbs:
        mbs_set = {int(x.strip()) for x in args.filter_mbs.split(",")}
        shapes = [s for s in shapes if s["mbs"] in mbs_set]
    if args.max_shapes:
        shapes = shapes[:args.max_shapes]

    if args.list:
        print(f"{'#':>4}  {'Model':<18}  {'Layer':<14}  {'Phase':<6}  {'MBS':>4}  "
              f"{'Trans':>5}  {'M':>6}  {'N':>6}  {'K':>6}")
        print("-" * 86)
        for i, s in enumerate(shapes):
            code = _trans_code(s.get("trans_a", True), s.get("trans_b", False))
            phase = s.get("phase", "fwd")
            print(f"{i+1:>4}  {s['model']:<18}  {s['layer']:<14}  {phase:<6}  "
                  f"{s['mbs']:>4}  {code:>5}  "
                  f"{s['M']:>6}  {s['N']:>6}  {s['K']:>6}")
        print(f"\nTotal shapes: {len(shapes)}")
        return

    if not (args.run or args.gen_only or args.compare_only):
        parser.print_help()
        return

    print(f"Tensile working directory: {TENSILE_WD}")
    print(f"  rocisa lib: {_find_rocisa_lib(TENSILE_WD)}")

    if args.template:
        template_path = Path(args.template)
    else:
        template_path = TEMPLATES.get(args.dtype, TEMPLATES["bf16"])
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
        shape_queue = Queue()
        for i, shape in enumerate(shapes):
            shape_queue.put((i, shape))

        def _gpu_worker(gpu_id):
            while True:
                try:
                    i, shape = shape_queue.get_nowait()
                except Exception:
                    break
                try:
                    results[i] = _process_one_shape(
                        shape, i, total, header, footer,
                        out_dir, args, gpu_id)
                except Exception as e:
                    _thread_print(f"ERROR processing shape {i}: {e}")
                    results[i] = {
                        "model": shapes[i]["model"],
                        "layer": shapes[i]["layer"],
                        "phase": shapes[i].get("phase", "fwd"),
                        "mbs": shapes[i]["mbs"],
                        "trans": _trans_code(shapes[i].get("trans_a", True),
                                             shapes[i].get("trans_b", False)),
                        **{k: shapes[i][k] for k in ("M", "N", "K")},
                        "tensile_tflops": None, "tensile_time_us": None,
                        "tensile_winner": None, "bench_tflops": None,
                        "bench_time_us": None, "bench_kernel": None,
                    }

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(_gpu_worker, gid) for gid in gpu_ids]
            for fut in futures:
                fut.result()

    valid = [r for r in results if r is not None]
    if valid and (args.run or args.compare_only):
        print_report(valid, dtype=args.dtype)
        save_report_csv(valid, out_dir / "comparison_report.csv")


if __name__ == "__main__":
    main()
