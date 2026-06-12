## Ssymm

`aclblasSsymm` implements BLAS `symm` for multiplying a symmetric matrix with a dense matrix and writing the result to `C`.

## Goal

- Implement standard BLAS `symm` semantics on row-major dense buffers.
- Support left-side and right-side symmetric multiplication through `side`.
- Use only the valid symmetric half selected by `uplo`.

## Math Semantics

- `side = LEFT`: `C := alpha * A * B + beta * C`
- `side = RIGHT`: `C := alpha * B * A + beta * C`

Shape rules:

- When `side = LEFT`, logical shape of `A` is `m x m`.
- When `side = RIGHT`, logical shape of `A` is `n x n`.
- Logical shape of `B` and `C` is `m x n`.

## Storage Layout

This implementation uses row-major dense buffers:

- `A[row][col]` address: `row * lda + col`
- `B[row][col]` address: `row * ldb + col`
- `C[row][col]` address: `row * ldc + col`

`uplo` selects the valid half of `A`:

- `LOWER`: `A(i,j)` is valid when `i >= j`
- `UPPER`: `A(i,j)` is valid when `i <= j`

## Interface

```c
aclblasStatus_t aclblasSsymm(aclblasHandle handle,
                             aclblasSideMode_t side,
                             aclblasFillMode_t uplo,
                             int64_t m,
                             int64_t n,
                             const float *alpha,
                             const float *A,
                             int64_t lda,
                             const float *B,
                             int64_t ldb,
                             const float *beta,
                             float *C,
                             int64_t ldc);
```

## Parameter Constraints

- `handle` must be valid.
- `alpha`, `beta`, `A`, `B`, and `C` must be non-null.
- `m >= 0`, `n >= 0`
- When `side = LEFT`, `lda >= m`
- When `side = RIGHT`, `lda >= n`
- `ldb >= n`
- `ldc >= n`

## Unsupported Features

Current implementation only supports single-precision real domain (`float`). The following are explicitly unsupported:

- `transA / transB`
- batch / strided batch
- packed / banded / sparse
- column-major
- device scalar `alpha / beta`
- in-place aliasing among `A / B / C`
- other data types such as double / complex / half

## Current Support Scope

Current mainline-ready scope is:

- datatype: `float`
- layout: row-major dense buffers
- semantics: `side={LEFT,RIGHT}` and `uplo={LOWER,UPPER}`
- default execution chain: `GenericFallback + Phase1Optimized`
- formal optimized paths:
  - `OptimizedLeft / LeftCube`
  - `OptimizedRight / RightCube`

Debug switches still exist, but they are not part of the default dispatch contract:

- `SSYMM_DEBUG_PLAN=1`

## Fallback Triggers

The default dispatcher falls back to `GenericFallback / GenericKernel` when any of the following holds:

- `m <= 0` or `n <= 0` for quick-return/empty work
- small shapes below the formal plan threshold
- irregular leading dimensions (`lda/ldb/ldc` not matching the regular dense layout contract)
- unsupported or invalid enums/arguments

The current LEFT optimized backend is intentionally narrower than the generic path:
`LeftCube` is validated only for the frozen regular dense baseline shapes, so small-shape and irregular/padded LEFT cases remain on `GenericFallback / GenericKernel`.

For optimized paths, tails and mixed-diagonal chunks may still be handled by internal downgrade logic inside the selected formal backend. That is different from a full dispatch fallback.

## Known Non-Goals

This implementation does not target:

- extending the public API surface
- adding new datatypes
- batch or strided-batch variants
- transpose semantics
- column-major support
- treating probe or profile helpers as part of the production dispatch path

## Correctness Verification

Recommended regression command:

```bash
bash build.sh --ops=ssymm --run
```

`ssymm_profile --verify` runs CPU golden first and then checks device output.
When running through `build.sh`, `--device=<id>` is forwarded into both
`ssymm_test` and `ssymm_profile` via `TEST_DEVICE_ID`, so ACL setup uses the
selected device instead of always defaulting to device `0`.

## Test Invocation Modes

`ssymm_test` supports both the legacy direct filter mode and the new `build.sh`
run mode:

```bash
./build/test/ssymm/ssymm_test dispatch_regular_right_uses_optimized
```

When invoked by `build.sh --run`, the first argument is the test binary
directory used for copied config assets. `ssymm_test` treats that directory as
metadata and defaults to running `all` unless a second argument is provided as
an explicit case filter.

## Profiling Entry

Running without arguments:

```bash
./build/test/ssymm/ssymm_profile
```

This only prints usage and does not run profiling. Use one of the commands below for actual profiling.

Supported cases:

- `right_lower`
- `right_upper`
- `left_lower`
- `left_upper`
- `left_cube_pilot_lower`
- `left_cube_pilot_upper`

Supported suites:

- `baseline`
- `left_baseline`

Examples:

```bash
./build/test/ssymm/ssymm_profile --case right_lower --m 128 --n 512 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case right_upper --suite baseline --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case left_lower --suite left_baseline --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case left_upper --m 256 --n 64 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case left_cube_pilot_lower --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case left_lower_512x512 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case right_upper_256x512 --warmup 2 --repeat 5 --verify
```

Fixed shapes in `suite baseline`:

- `64 x 256`
- `128 x 512`
- `32 x 1024`
- `512 x 512`

Fixed shapes in `suite left_baseline`:

- `256 x 64`
- `256 x 256`
- `128 x 512`
- `512 x 128`

## Formal Baseline Freeze

Named regression filters added in `ssymm_test`:

- `left_lower_256x256`
- `left_upper_512x128`
- `right_lower_256x256`
- `right_upper_128x512`

For `ssymm_profile`, these names are output labels carried by `baseline_case=...`.
They are also accepted by `--case` as fixed-shape aliases:

- `left_lower`
- `left_upper`
- `right_lower`
- `right_upper`
- `left_lower_256x256`
- `left_upper_512x128`
- `right_lower_256x256`
- `right_upper_128x512`
- `left_lower_512x512`
- `right_upper_256x512`

Expected profiling output fields for the frozen baseline commands:

- `case`
- `m`
- `n`
- `lda`
- `ldb`
- `ldc`
- `warmup`
- `repeat`
- `verify`
- `baseline_case`
- `actual_path`
- `backend_path`
- `avg_ms`
- `total_e2e_ms`
- `alloc_ms`
- `h2d_ms`
- `tiling_config_copy_ms`
- `kernel_and_sync_ms`
- `d2h_ms`
- `free_ms`

`total_e2e_ms` is the sum across `repeat` runs. `avg_ms` is the per-run average.

## Path Regression Matrix

Task 8 adds named path-level tests in `ssymm_test` to keep dispatch behavior explicit:

- `dispatches_to_unified_for_phase1_right_lower`: `right_lower_256x256` must stay on `OptimizedRight / RightCube`
- `dispatches_to_unified_for_general_shapes`: `right_upper_256x512` must stay on `OptimizedRight / RightCube`
- `left_lower_padded_ld`: padded `lda/ldb/ldc` must fall back to `GenericFallback / GenericFallback`
- `left_upper_padded_ld`: padded `lda/ldb/ldc` must fall back to `GenericFallback / GenericFallback`
- `right_lower_padded_ld`: padded `lda/ldb/ldc` must fall back to `GenericFallback / GenericFallback`
- `right_upper_padded_ld`: padded `lda/ldb/ldc` must fall back to `GenericFallback / GenericFallback`
- `alpha_beta_general_case`: nontrivial `alpha/beta` on `right_lower_256x256` must still use `OptimizedRight / RightCube`
- `zero_size_quick_return`: zero-size input must return `ACLBLAS_STATUS_SUCCESS` and leave trace at `GenericFallback / GenericFallback`
- `invalid_enum_case`: invalid `side/uplo` must return `ACLBLAS_STATUS_INVALID_ENUM` and leave trace at `GenericFallback / GenericFallback`

## Frozen Baseline

Baseline snapshot captured on `2026-05-01` in `cann900_beta2_container` with `warmup=2` and `repeat=5`.
These numbers are inherited historical frozen baseline evidence. They are kept as
regression reference only, and must not be read as fresh Phase 1 clean-mainline
proof for current `backend_path` naming:

| baseline_case | command | actual_path | backend_path | verify | avg_ms | total_e2e_ms | alloc_ms | h2d_ms | tiling_config_copy_ms | kernel_and_sync_ms | d2h_ms | free_ms |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `left_lower_256x256` | `./build/test/ssymm/ssymm_profile --case left_lower --m 256 --n 256 --warmup 2 --repeat 5 --verify` | `OptimizedLeft` | `LeftCube` | `pass` | `7.960` | `39.802` | `0.014` | `0.126` | `0.028` | `7.387` | `0.055` | `0.011` |
| `left_upper_512x128` | `./build/test/ssymm/ssymm_profile --case left_upper --m 512 --n 128 --warmup 2 --repeat 5 --verify` | `OptimizedLeft` | `LeftCube` | `pass` | `16.266` | `81.329` | `0.012` | `0.183` | `0.022` | `15.639` | `0.058` | `0.011` |
| `right_lower_256x256` | `./build/test/ssymm/ssymm_profile --case right_lower --m 256 --n 256 --warmup 2 --repeat 5 --verify` | `OptimizedRight` | `historical frozen baseline` | `pass` | `4.169` | `20.844` | `0.012` | `0.101` | `0.023` | `3.982` | `0.036` | `0.011` |
| `right_upper_128x512` | `./build/test/ssymm/ssymm_profile --case right_upper --m 128 --n 512 --warmup 2 --repeat 5 --verify` | `OptimizedRight` | `historical frozen baseline` | `pass` | `40.418` | `202.088` | `0.014` | `0.145` | `0.035` | `39.999` | `0.036` | `0.012` |

Fresh Phase 1 clean-mainline evidence is recorded through the current branch
verification commands and their live `backend_path` output, not by rewriting
the inherited 2026-05-01 snapshot.

## Keep Or Regress Rule

The following shapes are the formal optimized representatives and must keep their path and backend stable:

- `left_lower_256x256` -> `OptimizedLeft / LeftCube`
- `left_upper_512x128` -> `OptimizedLeft / LeftCube`
- `right_lower_256x256` -> `OptimizedRight / RightCube`
- `right_upper_128x512` -> `OptimizedRight / RightCube`

Regression judgment for these four baselines:

- `actual_path` or `backend_path` change counts as regress unless the README baseline is intentionally refreshed.
- `avg_ms` and `kernel_and_sync_ms` should stay within `+10%` of the frozen baseline on the same container.
- `alloc_ms`, `h2d_ms`, `tiling_config_copy_ms`, `d2h_ms`, and `free_ms` are diagnostic counters; investigate large movement, but do not use them as the only performance gate.
- Fallback-only cases are correctness gates first; they do not need to hit an optimized backend.

Reproducibility commands used for the current freeze:

```bash
./build/test/ssymm/ssymm_test
./build/test/ssymm/ssymm_profile --case left_lower --m 256 --n 256 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case left_upper --m 512 --n 128 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case right_lower --m 256 --n 256 --warmup 2 --repeat 5 --verify
./build/test/ssymm/ssymm_profile --case right_upper --m 128 --n 512 --warmup 2 --repeat 5 --verify
```

When `SSYMM_DEBUG_PLAN=1` is enabled, debug output distinguishes:

- `candidate_path`: host-side recommended plan from `BuildSsymmExecutionPlan(...)`
- `actual_path`: the formal path actually taken (`GenericFallback / OptimizedLeft / OptimizedRight`)
- `backend_path`: the concrete backend implementation selected under that formal path

`candidate_path` and `actual_path` may differ if the recommended plan is later rejected by the formal dispatcher. `backend_path` is the source of truth for the concrete implementation that actually ran.
