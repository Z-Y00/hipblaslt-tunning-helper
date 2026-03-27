#!/usr/bin/env python3
"""Staged GEMM kernel search — finds kernels beating stock hipBLASLt.

Divides the parameter space into 5 stages, each narrowing the search:
  1. Tile Selection (MI9 tiles via Origami)
  2. Memory System (TLDS, DTL, 1LDS, CLR)
  3. Fine Tuning (DepthU, StaggerU, StaggerUStride)
  4. Execution Model (StreamK, WGMXCC, WGM, SKXCCM)
  5. Cache Coherency (NTA, NTB, NTC, NTD)

Each stage benchmarks ~10-144 solutions and propagates the top-3 winners
to the next stage. Total: ~346 solutions in ~37 min per shape.

Usage:
  python3 staged_search.py --m 8192 --n 128256 --k 32768 --transA N --transB T --device 1
  python3 staged_search.py --from-config --device 1
  python3 staged_search.py --m 8192 --n 128256 --k 32768 --resume stage3 --device 1
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from run_shapes import (
    TENSILE_WD, _find_rocisa_lib,
    compute_rotating_buffer_mb, parse_tensile_csv,
    run_hipblaslt_bench, run_api_bench,
    _expand_mi4_for_shape, _origami_filter_mi9,
    _origami_available,
)

WORKSPACE = Path(__file__).resolve().parent
STAGE_RESULTS_DIR = WORKSPACE / "staged_results"
TOP_N = 3


# ---------------------------------------------------------------------------
# YAML generation for each stage
# ---------------------------------------------------------------------------


def generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, mi9_list=None,
                        out_path=None, origami_top_n=0):
    """Generate a Tensile YAML using run_shapes.gen_yaml with overridden ForkParameters.

    Uses the existing template and gen_yaml() pipeline, then patches individual
    ForkParameter values via regex replacement on the generated YAML.
    """
    from run_shapes import read_template, gen_yaml as _gen_yaml

    header, footer = read_template()

    # Extract MI9 list from fork_params if provided there
    if mi9_list is None and "MatrixInstruction" in fork_params:
        mi9_list = fork_params["MatrixInstruction"]

    # Use gen_yaml to produce a working baseline YAML (handles MI expansion, transpose, etc.)
    shape = {"M": N, "N": M, "K": K, "trans_a": trans_a, "trans_b": trans_b}
    if out_path is None:
        out_path = Path("/tmp/staged_search_tmp.yaml")

    origami_n = 10 if mi9_list is None else 0
    _gen_yaml(shape, header, footer, out_path, origami_top_n=origami_n, dtype="bf16")

    # Now patch the YAML with stage-specific ForkParameters
    content = out_path.read_text()

    # If MI9 list provided, replace MatrixInstruction entries
    if mi9_list is not None:
        # Remove existing MI entries and replace
        lines = content.split("\n")
        new_lines = []
        skip_mi = False
        for line in lines:
            if "MatrixInstruction:" in line and "ForkParameters" not in line:
                new_lines.append(f"        - MatrixInstruction:")
                for mi in mi9_list:
                    new_lines.append(f"          - {mi}")
                skip_mi = True
                continue
            if skip_mi:
                stripped = line.strip()
                if stripped.startswith("- [") or stripped.startswith("# -"):
                    continue
                skip_mi = False
            new_lines.append(line)
        content = "\n".join(new_lines)

    # Remove the second bias pass — keep BiasDataTypeList for codegen but benchmark only once
    content = re.sub(
        r"BiasTypeArgs:\s*\[.*?\]",
        "BiasTypeArgs: ['S']",
        content,
    )

    # Override specific ForkParameter values
    for key, vals in fork_params.items():
        if key == "MatrixInstruction":
            continue
        val_str = str(vals) if isinstance(vals, list) else str([vals])
        # Match the ForkParameter line (with possible inline comments)
        pattern = rf"(        - {key}:\s*)\[.*?\](.*)"
        if re.search(pattern, content):
            content = re.sub(pattern, rf"\g<1>{val_str}\g<2>", content)
        else:
            # Parameter not in template — insert before BenchmarkJoinParameters
            # Find the line that has BenchmarkJoinParameters
            lines = content.split("\n")
            for idx, line in enumerate(lines):
                if "BenchmarkJoinParameters:" in line:
                    lines.insert(idx, f"        - {key}: {val_str}")
                    break
            content = "\n".join(lines)

    out_path.write_text(content)
    return content


# ---------------------------------------------------------------------------
# Run Tensile and parse all solutions (not just winner)
# ---------------------------------------------------------------------------

def run_stage(yaml_path, case_dir, device, stage_name):
    """Run Tensile for a stage and return list of (tflops, solution_name) tuples."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    if device is not None:
        text = yaml_path.read_text()
        text = re.sub(r"Device:\s*\d+", "Device: 0", text)
        yaml_path.write_text(text)

    import subprocess
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
    print(f"  Running Tensile ...  (log: {log_path.name})")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, env=env, cwd=str(tensile_wd),
                                stdout=log_f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            elapsed = time.time() - t0
            print(f"\r  Tensile: {elapsed:6.0f}s", end="", flush=True)
            time.sleep(10)

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\r  Tensile: {status} in {elapsed:.0f}s{' '*40}")

    # Parse all solutions from the log (not just winner)
    solutions = _parse_all_solutions(log_path)
    winner = parse_tensile_csv(case_dir)

    if winner:
        print(f"  Winner: {winner['tflops']:.1f} TFLOPS ({winner['time_us']:.1f} us)")
    print(f"  Total solutions benchmarked: {len(solutions)}")

    return solutions, winner


def _parse_all_solutions(log_path):
    """Parse all PASSED solutions from a Tensile log, return sorted by TFLOPS."""
    solutions = []
    try:
        with open(log_path) as f:
            for line in f:
                if "PASSED" not in line:
                    continue
                m = re.search(r"PASSED,([\d.e+]+),([\d.e+]+)", line)
                if not m:
                    continue
                time_us = float(m.group(1))
                gflops = float(m.group(2))
                if gflops < 1000:
                    continue
                name_m = re.search(r"(Cijk_\S+?),PASSED", line)
                name = name_m.group(1) if name_m else "unknown"
                solutions.append({
                    "tflops": gflops / 1000,
                    "time_us": time_us,
                    "name": name,
                })
    except Exception:
        pass
    solutions.sort(key=lambda x: x["tflops"], reverse=True)
    return solutions


def _extract_param(name, param_abbrev):
    """Extract a parameter value from a kernel name string."""
    m = re.search(rf"{param_abbrev}(\d+)", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage1_tile_selection(M, N, K, trans_a, trans_b, device, out_dir):
    """Stage 1: Find the best MI9 tile configurations."""
    print("\n" + "#"*60)
    print("# STAGE 1: Tile Selection")
    print("#"*60)

    # Generate MI9 tiles via Origami
    mi4_list = [[32, 32, 16, 1], [16, 16, 32, 1]]
    mi9_list = _expand_mi4_for_shape(mi4_list, M, N)

    if _origami_available and K > 0:
        depth_u_values = [64]
        before = len(mi9_list)
        mi9_list = _origami_filter_mi9(
            mi9_list, M, N, K, depth_u_values, top_n=10,
            trans_a=trans_a, trans_b=trans_b, dtype="bf16",
        )
        print(f"  Origami pruned: {before} -> {len(mi9_list)} MI9 tiles")

    fork_params = {
        "MatrixInstruction": mi9_list,
        "DepthU": [64],
        "StreamK": [3],
        "SourceSwap": [1],
        "TransposeLDS": [0, 1],
        "DirectToLds": [0, 1],
        "ClusterLocalRead": [0],
        "NonTemporalD": [0],
        "NonTemporalB": [0],
        "StaggerU": [0],
        "StaggerUStride": [256],
        "WorkGroupMappingXCC": [1],
        "WorkGroupMappingXCCGroup": [-1],
        "1LDSBuffer": [-1],
        "GlobalSplitU": [0],
        "GlobalSplitUAlgorithm": ["MultipleBuffer"],
        "PreloadKernArgs": [True],
        "LdsBlockSizePerPadA": [-1],
        "LdsBlockSizePerPadB": [-1],
        "LdsPadA": [-1],
        "LdsPadB": [-1],
        "LocalReadVectorWidth": [-1],
        "ScheduleIterAlg": [3],
        "PrefetchGlobalRead": [2],
        "PrefetchLocalRead": [1],
    }

    yaml_path = out_dir / "stage1.yaml"
    case_dir = out_dir / "stage1"
    generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, out_path=yaml_path)

    solutions, winner = run_stage(yaml_path, case_dir, device, "Stage 1: Tile Selection")

    # Extract top-N unique tile configs
    seen_tiles = set()
    top_tiles = []
    for sol in solutions:
        mt = re.search(r"MT(\d+x\d+x\d+)", sol["name"])
        mi = re.search(r"MI(\d+x\d+x\d+)", sol["name"])
        miwt = re.search(r"MIWT(\d+_\d+)", sol["name"])
        wg = re.search(r"WG(\d+_\d+_\d+)", sol["name"])
        tile_key = (mt.group(1) if mt else "", mi.group(1) if mi else "",
                    miwt.group(1) if miwt else "", wg.group(1) if wg else "")
        if tile_key not in seen_tiles and tile_key != ("", "", "", ""):
            seen_tiles.add(tile_key)
            top_tiles.append({"tile_key": tile_key, "tflops": sol["tflops"], "name": sol["name"]})
            if len(top_tiles) >= TOP_N:
                break

    print(f"\n  Top {len(top_tiles)} tiles:")
    for i, t in enumerate(top_tiles):
        print(f"    {i+1}. {t['tflops']:.1f} TFLOPS  MT{t['tile_key'][0]}_MI{t['tile_key'][1]}")

    # Save stage results
    _save_stage_results(out_dir / "stage1_results.json", solutions[:20], top_tiles)
    return top_tiles


def stage2_memory_system(M, N, K, trans_a, trans_b, device, out_dir, top_tiles):
    """Stage 2: Sweep memory system params for each top tile."""
    print("\n" + "#"*60)
    print("# STAGE 2: Memory System")
    print("#"*60)

    best_combos = []
    for i, tile in enumerate(top_tiles):
        # Find the MI9 from the tile's kernel name
        mi9 = _extract_mi9_from_name(tile["name"])
        if mi9 is None:
            print(f"  Skipping tile {i+1}: could not extract MI9")
            continue

        fork_params = {
            "MatrixInstruction": [mi9],
            "DepthU": [64],
            "StreamK": [3],
            "SourceSwap": [1],
            "TransposeLDS": [0, 1],
            "DirectToLds": [0, 1],
            "1LDSBuffer": [0, 1],
            "ClusterLocalRead": [0, 1],
            "NonTemporalD": [0],
            "NonTemporalB": [0],
            "StaggerU": [0],
            "StaggerUStride": [256],
            "WorkGroupMappingXCC": [1],
            "WorkGroupMappingXCCGroup": [-1],
            "GlobalSplitU": [0],
            "GlobalSplitUAlgorithm": ["MultipleBuffer"],
            "PreloadKernArgs": [True],
            "LdsBlockSizePerPadA": [-1],
            "LdsBlockSizePerPadB": [-1],
            "LdsPadA": [-1],
            "LdsPadB": [-1],
            "LocalReadVectorWidth": [-1],
            "ScheduleIterAlg": [3],
            "PrefetchGlobalRead": [2],
            "PrefetchLocalRead": [1],
        }

        yaml_path = out_dir / f"stage2_tile{i}.yaml"
        case_dir = out_dir / f"stage2_tile{i}"
        generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, out_path=yaml_path)

        solutions, winner = run_stage(yaml_path, case_dir, device,
                                       f"Stage 2: Memory System (tile {i+1}/{len(top_tiles)})")

        if solutions:
            best = solutions[0]
            best_combos.append({"tile": tile, "mem_winner": best, "name": best["name"]})
            print(f"    Best: {best['tflops']:.1f} TFLOPS")

    print(f"\n  Top {len(best_combos)} tile+mem combos:")
    for i, c in enumerate(best_combos):
        print(f"    {i+1}. {c['mem_winner']['tflops']:.1f} TFLOPS")

    _save_stage_results(out_dir / "stage2_results.json", [], best_combos)
    return best_combos[:TOP_N]


def stage3_fine_tuning(M, N, K, trans_a, trans_b, device, out_dir, combos):
    """Stage 3: Sweep DepthU, StaggerU, StaggerUStride."""
    print("\n" + "#"*60)
    print("# STAGE 3: Fine Tuning")
    print("#"*60)

    best_combos = []
    for i, combo in enumerate(combos):
        mi9 = _extract_mi9_from_name(combo["name"])
        base_params = _extract_params_from_name(combo["name"])
        if mi9 is None:
            continue

        fork_params = {
            "MatrixInstruction": [mi9],
            "DepthU": [32, 64],
            "StreamK": [3],
            "SourceSwap": [1],
            "TransposeLDS": [base_params.get("TLDS", 1)],
            "DirectToLds": [base_params.get("DTL", 1)],
            "1LDSBuffer": [base_params.get("1LDS", 0)],
            "ClusterLocalRead": [base_params.get("CLR", 0)],
            "StaggerU": [0, 8, 16],
            "StaggerUStride": [0, 64, 128, 256],
            "NonTemporalD": [0],
            "NonTemporalB": [0],
            "WorkGroupMappingXCC": [1],
            "WorkGroupMappingXCCGroup": [-1],
            "GlobalSplitU": [0],
            "GlobalSplitUAlgorithm": ["MultipleBuffer"],
            "PreloadKernArgs": [True],
            "LdsBlockSizePerPadA": [-1],
            "LdsBlockSizePerPadB": [-1],
            "LdsPadA": [-1],
            "LdsPadB": [-1],
            "LocalReadVectorWidth": [-1],
            "ScheduleIterAlg": [3],
            "PrefetchGlobalRead": [2],
            "PrefetchLocalRead": [1],
        }

        yaml_path = out_dir / f"stage3_combo{i}.yaml"
        case_dir = out_dir / f"stage3_combo{i}"
        generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, out_path=yaml_path)

        solutions, winner = run_stage(yaml_path, case_dir, device,
                                       f"Stage 3: Fine Tuning (combo {i+1}/{len(combos)})")

        if solutions:
            best = solutions[0]
            best_combos.append({"prev": combo, "winner": best, "name": best["name"]})
            print(f"    Best: {best['tflops']:.1f} TFLOPS")

    _save_stage_results(out_dir / "stage3_results.json", [], best_combos)
    return best_combos[:TOP_N]


def stage4_execution_model(M, N, K, trans_a, trans_b, device, out_dir, combos):
    """Stage 4: Sweep StreamK, WGMXCC, WGM."""
    print("\n" + "#"*60)
    print("# STAGE 4: Execution Model")
    print("#"*60)

    best_combos = []
    for i, combo in enumerate(combos):
        mi9 = _extract_mi9_from_name(combo["name"])
        base_params = _extract_params_from_name(combo["name"])
        if mi9 is None:
            continue

        fork_params = {
            "MatrixInstruction": [mi9],
            "DepthU": [base_params.get("DU", 64)],
            "StreamK": [0, 3],
            "SourceSwap": [1],
            "TransposeLDS": [base_params.get("TLDS", 1)],
            "DirectToLds": [base_params.get("DTL", 1)],
            "1LDSBuffer": [base_params.get("1LDS", 0)],
            "ClusterLocalRead": [base_params.get("CLR", 0)],
            "StaggerU": [base_params.get("SU", 0)],
            "StaggerUStride": [base_params.get("SUS", 256)],
            "WorkGroupMappingXCC": [1, 2, 4, 8],
            "WorkGroupMappingXCCGroup": [-1],
            "NonTemporalD": [0],
            "NonTemporalB": [0],
            "GlobalSplitU": [0],
            "GlobalSplitUAlgorithm": ["MultipleBuffer"],
            "PreloadKernArgs": [True],
            "LdsBlockSizePerPadA": [-1],
            "LdsBlockSizePerPadB": [-1],
            "LdsPadA": [-1],
            "LdsPadB": [-1],
            "LocalReadVectorWidth": [-1],
            "ScheduleIterAlg": [3],
            "PrefetchGlobalRead": [2],
            "PrefetchLocalRead": [1],
        }

        yaml_path = out_dir / f"stage4_combo{i}.yaml"
        case_dir = out_dir / f"stage4_combo{i}"
        generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, out_path=yaml_path)

        solutions, winner = run_stage(yaml_path, case_dir, device,
                                       f"Stage 4: Execution Model (combo {i+1}/{len(combos)})")

        if solutions:
            best = solutions[0]
            best_combos.append({"prev": combo, "winner": best, "name": best["name"]})
            print(f"    Best: {best['tflops']:.1f} TFLOPS")

    _save_stage_results(out_dir / "stage4_results.json", [], best_combos)
    return best_combos[:TOP_N]


def stage5_cache_coherency(M, N, K, trans_a, trans_b, device, out_dir, combos):
    """Stage 5: Sweep NonTemporal for A, B, C, D."""
    print("\n" + "#"*60)
    print("# STAGE 5: Cache Coherency")
    print("#"*60)

    best_combos = []
    for i, combo in enumerate(combos):
        mi9 = _extract_mi9_from_name(combo["name"])
        base_params = _extract_params_from_name(combo["name"])
        if mi9 is None:
            continue

        fork_params = {
            "MatrixInstruction": [mi9],
            "DepthU": [base_params.get("DU", 64)],
            "StreamK": [base_params.get("SK", 3)],
            "SourceSwap": [1],
            "TransposeLDS": [base_params.get("TLDS", 1)],
            "DirectToLds": [base_params.get("DTL", 1)],
            "1LDSBuffer": [base_params.get("1LDS", 0)],
            "ClusterLocalRead": [base_params.get("CLR", 0)],
            "StaggerU": [base_params.get("SU", 0)],
            "StaggerUStride": [base_params.get("SUS", 256)],
            "WorkGroupMappingXCC": [base_params.get("WGMXCC", 1)],
            "WorkGroupMappingXCCGroup": [-1],
            "NonTemporalA": [0, 1, 4],
            "NonTemporalB": [0, 1, 4],
            "NonTemporalC": [0, 4],
            "NonTemporalD": [0, 4],
            "GlobalSplitU": [0],
            "GlobalSplitUAlgorithm": ["MultipleBuffer"],
            "PreloadKernArgs": [True],
            "LdsBlockSizePerPadA": [-1],
            "LdsBlockSizePerPadB": [-1],
            "LdsPadA": [-1],
            "LdsPadB": [-1],
            "LocalReadVectorWidth": [-1],
            "ScheduleIterAlg": [3],
            "PrefetchGlobalRead": [2],
            "PrefetchLocalRead": [1],
        }

        yaml_path = out_dir / f"stage5_combo{i}.yaml"
        case_dir = out_dir / f"stage5_combo{i}"
        generate_stage_yaml(M, N, K, trans_a, trans_b, fork_params, out_path=yaml_path)

        solutions, winner = run_stage(yaml_path, case_dir, device,
                                       f"Stage 5: Cache Coherency (combo {i+1}/{len(combos)})")

        if solutions:
            best = solutions[0]
            best_combos.append({"prev": combo, "winner": best, "name": best["name"]})
            print(f"    Best: {best['tflops']:.1f} TFLOPS")

    _save_stage_results(out_dir / "stage5_results.json", [], best_combos)
    return best_combos[:TOP_N]


# ---------------------------------------------------------------------------
# Helper: extract MI9 and params from kernel name
# ---------------------------------------------------------------------------

def _extract_mi9_from_name(name):
    """Extract MI9 list from a Tensile kernel name.

    MI9 format: [mi_M, mi_N, mi_K, mi_B, bm, tt0, tt1, wm, wn]
    Kernel name contains: MI{M}x{N}x{B}, MIWT{tt0}_{tt1}, WG{x}_{y}_{z}, MT{mtM}x{mtN}x{DU}
    Reconstruct: mi_K = 16 for MI32, 32 for MI16.
    wm = mtM / (mi_M * tt0), wn = mtN / (mi_N * tt1).
    """
    mt = re.search(r"MT(\d+)x(\d+)x(\d+)", name)
    mi = re.search(r"MI(\d+)x(\d+)x(\d+)", name)
    miwt = re.search(r"MIWT(\d+)_(\d+)", name)
    if not mi or not miwt or not mt:
        return None
    mi_m, mi_n = int(mi.group(1)), int(mi.group(2))
    tt0, tt1 = int(miwt.group(1)), int(miwt.group(2))
    mt_m, mt_n = int(mt.group(1)), int(mt.group(2))

    mi_k = 16 if mi_m == 32 else 32
    wm = mt_m // (mi_m * tt0) if (mi_m * tt0) > 0 else 1
    wn = mt_n // (mi_n * tt1) if (mi_n * tt1) > 0 else 1
    return [mi_m, mi_n, mi_k, 1, 1, tt0, tt1, wm, wn]


def _extract_params_from_name(name):
    """Extract key parameters from a kernel name."""
    params = {}
    patterns = {
        "TLDS": r"TLDS(\d+)",
        "DTL": r"DTLA(\d+)",
        "CLR": r"CLR(\d+)",
        "1LDS": r"1LDS(\d+)",  # Note: 1LDSBuffer appears as LDSB in some names
        "SU": r"_SU(\d+)_",
        "SUS": r"SUS(\d+)",
        "SK": r"SK(\d+)",
        "WGMXCC": r"WGMXCC(\d+)",
        "DU": r"MT\d+x\d+x(\d+)",
        "NTD": r"NTD(\d+)",
        "NTB": r"NTB(\d+)",
        "NTA": r"NTA(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, name)
        if m:
            params[key] = int(m.group(1))
    # LDSB for 1LDSBuffer
    m = re.search(r"LDSB(\d+)", name)
    if m and "1LDS" not in params:
        params["1LDS"] = int(m.group(1))
    return params


def _save_stage_results(path, solutions, combos):
    """Save stage results to JSON."""
    data = {
        "top_solutions": solutions[:20] if solutions else [],
        "combos": [{"tflops": c.get("winner", c.get("mem_winner", {})).get("tflops", 0),
                     "name": c.get("name", "")} for c in combos],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def generate_report(M, N, K, trans_a, trans_b, device, out_dir, final_combos):
    """Generate final comparison report."""
    print("\n" + "#"*60)
    print("# FINAL REPORT")
    print("#"*60)

    if not final_combos:
        print("  No valid results from staged search.")
        return

    best = final_combos[0]
    best_tflops = best["winner"]["tflops"]
    best_name = best["name"]

    # Run hipblaslt-bench for comparison
    bench = run_hipblaslt_bench(M, N, K, trans_a=trans_a, trans_b=trans_b, device=device)
    bench_tflops = bench["tflops"] if bench and bench.get("tflops") else None

    # Run API bench
    api = None
    try:
        api = run_api_bench(M, N, K, trans_a=trans_a, trans_b=trans_b, device=device)
    except Exception:
        pass
    api_tflops = api["tflops"] if api and api.get("tflops") else None

    print(f"\n  Staged Search Winner: {best_tflops:.1f} TFLOPS")
    if bench_tflops:
        print(f"  hipblaslt-bench:      {bench_tflops:.1f} TFLOPS")
        print(f"  T/B ratio:            {best_tflops/bench_tflops*100:.1f}%")
    if api_tflops:
        print(f"  API bench:            {api_tflops:.1f} TFLOPS")

    # Write report
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(f"# Staged Search Report\n\n")
        f.write(f"## Shape\n")
        f.write(f"M={M}, N={N}, K={K}, TransA={'T' if trans_a else 'N'}, TransB={'T' if trans_b else 'N'}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | TFLOPS | vs Bench |\n")
        f.write(f"|--------|-------:|---------:|\n")
        f.write(f"| **Staged winner** | {best_tflops:.1f} | ")
        if bench_tflops:
            f.write(f"{best_tflops/bench_tflops*100:.1f}%")
        f.write(f" |\n")
        if api_tflops:
            f.write(f"| API bench | {api_tflops:.1f} | ")
            if bench_tflops:
                f.write(f"{api_tflops/bench_tflops*100:.1f}%")
            f.write(f" |\n")
        if bench_tflops:
            f.write(f"| hipblaslt-bench | {bench_tflops:.1f} | 100% |\n")
        f.write(f"\n## Winner Kernel\n\n`{best_name}`\n")

    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, required=False, help="BLAS M dimension")
    parser.add_argument("--n", type=int, required=False, help="BLAS N dimension")
    parser.add_argument("--k", type=int, required=False, help="BLAS K dimension")
    parser.add_argument("--transA", type=str, default="N", choices=["N", "T"])
    parser.add_argument("--transB", type=str, default="N", choices=["N", "T"])
    parser.add_argument("--device", type=int, default=1, help="GPU device (default: 1, avoids GPU 0)")
    parser.add_argument("--from-config", action="store_true",
                        help="Run on RegressionTestCases from config.py")
    parser.add_argument("--resume", type=str, default=None,
                        choices=["stage2", "stage3", "stage4", "stage5"],
                        help="Resume from a specific stage (requires prior stage results)")
    parser.add_argument("--top-n", type=int, default=3, help="Top N winners per stage")
    args = parser.parse_args()

    global TOP_N
    TOP_N = args.top_n

    if args.from_config:
        from config import RegressionTestCases
        if not RegressionTestCases:
            print("No RegressionTestCases defined in config.py")
            return
        for case in RegressionTestCases:
            _run_shape(case["M"], case["N"], case["K"],
                       case.get("trans_a", True), case.get("trans_b", False),
                       args.device, args.resume)
    elif args.m and args.n and args.k:
        trans_a = args.transA == "T"
        trans_b = args.transB == "T"
        _run_shape(args.m, args.n, args.k, trans_a, trans_b, args.device, args.resume)
    else:
        parser.print_help()
        sys.exit(1)


def _run_shape(M, N, K, trans_a, trans_b, device, resume=None):
    ta = "T" if trans_a else "N"
    tb = "T" if trans_b else "N"
    shape_id = f"{ta}{tb}_M{M}_N{N}_K{K}"
    out_dir = STAGE_RESULTS_DIR / shape_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Staged Search: M={M} N={N} K={K} Trans={ta}{tb}")
    print(f"# Output: {out_dir}")
    print(f"# Device: GPU {device}")
    print(f"{'#'*60}")

    start_stage = 1
    if resume:
        start_stage = int(resume.replace("stage", ""))

    # Stage 1
    if start_stage <= 1:
        top_tiles = stage1_tile_selection(M, N, K, trans_a, trans_b, device, out_dir)
    else:
        top_tiles = _load_stage_combos(out_dir / "stage1_results.json")
        print(f"  Loaded {len(top_tiles)} tiles from stage 1")

    if not top_tiles:
        print("  No valid tiles found. Aborting.")
        return

    # Stage 2
    if start_stage <= 2:
        mem_combos = stage2_memory_system(M, N, K, trans_a, trans_b, device, out_dir, top_tiles)
    else:
        mem_combos = _load_stage_combos(out_dir / "stage2_results.json")
        print(f"  Loaded {len(mem_combos)} combos from stage 2")

    if not mem_combos:
        print("  No valid memory combos. Aborting.")
        return

    # Stage 3
    if start_stage <= 3:
        fine_combos = stage3_fine_tuning(M, N, K, trans_a, trans_b, device, out_dir, mem_combos)
    else:
        fine_combos = _load_stage_combos(out_dir / "stage3_results.json")
        print(f"  Loaded {len(fine_combos)} combos from stage 3")

    if not fine_combos:
        print("  No valid fine-tune combos. Aborting.")
        return

    # Stage 4
    if start_stage <= 4:
        exec_combos = stage4_execution_model(M, N, K, trans_a, trans_b, device, out_dir, fine_combos)
    else:
        exec_combos = _load_stage_combos(out_dir / "stage4_results.json")
        print(f"  Loaded {len(exec_combos)} combos from stage 4")

    if not exec_combos:
        print("  No valid execution combos. Aborting.")
        return

    # Stage 5
    cache_combos = stage5_cache_coherency(M, N, K, trans_a, trans_b, device, out_dir, exec_combos)

    if not cache_combos:
        print("  No valid cache combos. Using stage 4 winner.")
        cache_combos = exec_combos

    # Final report
    generate_report(M, N, K, trans_a, trans_b, device, out_dir, cache_combos)


def _load_stage_combos(path):
    """Load combos from a saved stage result JSON."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    combos = []
    for c in data.get("combos", []):
        combos.append({"name": c["name"], "winner": {"tflops": c["tflops"], "name": c["name"]}})
    return combos


if __name__ == "__main__":
    main()
