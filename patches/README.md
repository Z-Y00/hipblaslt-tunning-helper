# Tensile Patches

Patches for the Tensile client and Python infrastructure in `tmp_rebuild/rocm-libraries`.

## Patch files

| File | Target repo | Description |
|------|-------------|-------------|
| `0001-cosmo-tensile-tuning-fixes.patch` | `hipblaslt` submodule | Original format-patch from `hipblaslt` commit `7e2615bb5` |
| `0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch` | `tmp_rebuild/rocm-libraries` | Path-remapped version for the monorepo (applied diff) |

## What the patch fixes

1. **int32 overflow in rotating buffer size** — `Rotating.cpp` used `int` for buffer size calculation causing overflow on large shapes (e.g. `369131520 * 6 = -2080178176`). Changed to `size_t`.
2. **int32 overflow in rotating-buffer-size CLI arg** — `DataInitialization.cpp` parsed the CLI arg as `int`. Changed to `size_t`.
3. **hipMalloc error handling** — `Rotating.cpp` now checks `hipMalloc` return code.
4. **Grouped GEMM TFLOPS reporting** — `BenchmarkTimer.cpp` summed only the first sub-gemm's FLOPs. Now sums all sub-gemms.
5. **Empty assembly guard** — `Run.py` detects kernels that produced empty source without error flags.
6. **hsaco filename alias** — `Source.py` creates an unsuffixed `.hsaco` alias for xnack-qualified targets.
7. **Assembly toolchain guard** — `Assembly.py` skips invalid/empty assembly files.
8. **Thread-local RNG** — `DataInitialization.cpp` replaces glibc `rand()` with thread-local RNG (already upstreamed in `tmp_rebuild` as `getThreadLocalRandInt()`; the `.cpp` side-effects still needed).
9. **Progress listener flush** — `ProgressListener.cpp` flushes stdout after progress updates.

Note: The `DataInitialization.hpp` `rand()→tl_rand()` change was **not needed** in `tmp_rebuild` because it already uses `getThreadLocalRandInt()`. The `origami/setup.py` fix was skipped (file doesn't exist in monorepo).

## How to apply

```bash
cd /workspace/cosmo/hipblaslt-tunning-helper

# 1. Apply the patch to tmp_rebuild/rocm-libraries
cd tmp_rebuild/rocm-libraries
git apply /workspace/cosmo/hipblaslt-tunning-helper/patches/0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch

# 2. Rebuild the Tensile client (C++ changes need recompilation)
cd projects/hipblaslt/tensilelite
rm -rf build_tmp/tensilelite/client
export LLVM_DIR=/opt/rocm/llvm
pip3 install invoke
invoke build-client
```

## How this patch was created

```bash
# Export from hipblaslt submodule (commit 7e2615bb5)
cd hipblaslt
git format-patch -1 7e2615bb5 --stdout > patches/0001-cosmo-tensile-tuning-fixes.patch

# Apply with --reject to tmp_rebuild, then manually resolve conflicts
cd tmp_rebuild/rocm-libraries
git apply --reject patches/0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch
# Manually fixed: Run.py (refactored function), Source.py (context shift),
#   BenchmarkTimer.cpp (m_solution-> vs m_solution.)
# Skipped: DataInitialization.hpp rand() (already fixed upstream), origami/setup.py (missing)
# Saved final applied diff
git diff projects/hipblaslt/tensilelite/ > patches/0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch
```
