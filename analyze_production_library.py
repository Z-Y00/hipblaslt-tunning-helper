#!/usr/bin/env python3
"""Analyze the production hipBLASLt library's kernel parameters for gfx950.

Scans the Equality YAML files in the Tensile Logic directory and reports
the unique values for each tuning-relevant field, along with counts.
Estimates the total combinatorial search space.

Usage:
  python3 analyze_production_library.py
  python3 analyze_production_library.py --logic-dir /path/to/Logic/asm_full/gfx950/Equality
  python3 analyze_production_library.py --logic-dir tmp_rebuild/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality

  python3 analyze_production_library.py --file tmp_rebuild/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality/gfx950_Cijk_Ailk_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml
 # 48,448,880,366,321,664
  python3 analyze_production_library.py --file tmp_rebuild/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/aquavanjaram_Cijk_Ailk_Bjlk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml
 # 112,140,288
  python3 analyze_production_library.py --file tmp_rebuild/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/aquavanjaram_Cijk_Ailk_Bjlk_BBS_BH_UserArgs.yaml
 #  6,742,112,993,280
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_LOGIC_DIR = (
    WORKSPACE / "tmp_rebuild" / "rocm-libraries" / "projects" / "hipblaslt"
    / "library" / "src" / "amd_detail" / "rocblaslt" / "src" / "Tensile"
    / "Logic" / "asm_full" / "gfx950" / "Equality"
)

FIELDS = [
    "MIBlock", "DepthU", "TransposeLDS",
    "DirectToLdsA", "DirectToLdsB",
    "NonTemporalA", "NonTemporalB", "NonTemporalD",
    "StaggerU", "StaggerUStride", "StaggerUMapping",
    "WorkGroupMappingXCC", "WorkGroupMappingXCCGroup",
    "1LDSBuffer", "GlobalSplitU", "GlobalSplitUAlgorithm",
    "StreamK", "SourceSwap",
    "PreloadKernArgs", "ClusterLocalRead",
    "PrefetchGlobalRead", "PrefetchLocalRead", "ScheduleIterAlg",
    "LdsBlockSizePerPadA", "LdsBlockSizePerPadB",
    "LdsPadA", "LdsPadB",
    "LocalReadVectorWidth",
    "GlobalReadVectorWidthA", "GlobalReadVectorWidthB",
    "VectorWidthA", "VectorWidthB",
    "StoreVectorWidth", "StoreRemapVectorWidth",
    "InnerUnroll",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logic-dir", type=str, default=str(DEFAULT_LOGIC_DIR),
                        help="Path to Equality YAML directory")
    parser.add_argument("--file", type=str, default=None,
                        help="Analyze a single YAML file instead of the whole directory")
    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            print(f"ERROR: {args.file} not found")
            return
        yaml_files = [args.file]
        print(f"Analyzing: {os.path.basename(args.file)}")
    else:
        logic_dir = args.logic_dir
        if not os.path.isdir(logic_dir):
            print(f"ERROR: {logic_dir} not found")
            return
        yaml_files = [os.path.join(logic_dir, f) for f in sorted(os.listdir(logic_dir)) if f.endswith(".yaml")]

    values = defaultdict(lambda: defaultdict(int))
    total_kernels = 0

    for fpath in yaml_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                for field in FIELDS:
                    m = re.match(rf"{field}:\s*(.+)", line)
                    if m:
                        values[field][m.group(1).strip()] += 1
                if line.startswith("SolutionIndex:"):
                    total_kernels += 1

    print(f"Total kernels in production library: {total_kernels}")
    print()
    print(f"{'Field':<30} {'Unique':>6}  Values (count)")
    print("=" * 100)

    search_space = 1
    for field in FIELDS:
        if field not in values:
            continue
        vals = values[field]
        sorted_vals = sorted(vals.items(), key=lambda x: -x[1])
        n_unique = len(sorted_vals)
        search_space *= n_unique

        total = sum(vals.values())
        top = sorted_vals[:6]
        val_str = "  ".join(f"{v}({c*100/total:.0f}%)" for v, c in top)
        if len(sorted_vals) > 6:
            val_str += f"  ...+{len(sorted_vals) - 6} more"
        print(f"{field:<30} {n_unique:>6}  {val_str}")

    print()
    print(f"Estimated combinatorial search space: {search_space:,}")
    print(f"(product of unique values per field — many combos are invalid)")


if __name__ == "__main__":
    main()
