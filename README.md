# Optimizing Matrix Multiply

A benchmark suite comparing three implementations of double-precision square matrix multiplication (`C := C + A * B`), targeting the [Perlmutter](https://docs.nersc.gov/systems/perlmutter/) supercomputer at NERSC.

The three implementations are:

- **Naive** — a straightforward triple-loop baseline
- **Blocked** — a cache-blocked implementation with an AVX2/FMA microkernel
- **BLAS** — a reference ceiling using `cblas_dgemm`

---

## Repository Structure

```
.
├── CMakeLists.txt       # Build system; configures all three benchmarks
├── benchmark.cpp        # Timing harness and correctness checker (shared)
├── dgemm-naive.c        # Triple-loop baseline implementation
├── dgemm-blocked.c      # Optimized blocked + AVX2 implementation
├── dgemm-blas.c         # BLAS reference implementation
├── job.in               # SLURM job script template (filled by CMake)
└── build/               # Out-of-source build directory (generated)
```

---

## Prerequisites

### On Perlmutter

The build requires the GNU programming environment and a BLAS library. Load the appropriate modules before configuring:

```bash
module swap PrgEnv-<current> PrgEnv-gnu
module load cmake
```

> **Note:** The CMakeLists enforces the GNU compiler. Using PrgEnv-intel or PrgEnv-cray will produce a warning and may cause build issues.

### General Requirements

| Requirement | Notes |
|---|---|
| CMake ≥ 3.14 | |
| GCC | Required; Clang/ICC not supported for submission |
| BLAS library | e.g., Cray LibSci (loaded automatically on Perlmutter), OpenBLAS, or MKL |
| AVX2 + FMA support | The blocked kernel targets `znver3` (AMD Zen 3, Perlmutter's CPU) |

---

## Building

Configure and build out-of-source from the repository root:

```bash
mkdir build && cd build
cmake ..
make
```

This produces three executables in the `build/` directory:

```
benchmark-naive
benchmark-blocked
benchmark-blas
```

CMake also generates corresponding SLURM job scripts:

```
job-naive
job-blocked
job-blas
```

### Build Options

| CMake Variable | Default | Description |
|---|---|---|
| `ALL_SIZES` | `OFF` | Test an extended set of matrix sizes (31–1025). Significantly increases runtime. Enable with `-DALL_SIZES=ON` |
| `MAX_SPEED` | `56` | Peak CPU throughput in GFlop/s. The default reflects Perlmutter's CPU: 3.5 GHz × 4 (AVX2 doubles) × 2 (pipelines) × 2 (FMA) = 56 GFlop/s |
| `GROUP_NO` | `00` | Two-digit group number for packaging a submission tarball via CPack |

Example with options:

```bash
cmake .. -DALL_SIZES=ON -DMAX_SPEED=56
```

---

## Running

### Interactively (for quick testing)

```bash
cd build
./benchmark-naive
./benchmark-blocked
./benchmark-blas
```

### On Perlmutter via SLURM (recommended for accurate results)

The generated job scripts pin the CPU frequency to 3.5 GHz, use a single core, and request the `debug` partition (2-minute limit):

```bash
cd build
sbatch job-naive
sbatch job-blocked
sbatch job-blas
```

Output is written to `job-<name>.o<jobid>` and errors to `job-<name>.e<jobid>`.

### Custom Matrix Sizes

You can pass explicit sizes as command-line arguments to any benchmark executable:

```bash
./benchmark-blocked 64 128 256 512 1024
```

---

## Output Format

Each benchmark prints one line per matrix size, followed by an overall average:

```
Description:    Simple blocked dgemm.

Size: 32        Mflops/s: 48312.45      Percentage: 86.27
Size: 64        Mflops/s: 51204.10      Percentage: 91.44
...
Average percentage of Peak = 78.34
```

**Percentage** is relative to `MAX_SPEED` (56 GFlop/s by default). BLAS should approach or exceed 100% on some sizes due to vendor tuning; the naive baseline will be far below.

---

## Implementation Details

### `dgemm-naive.c`
A straightforward `i-j-k` loop over all matrix elements. Serves as a correctness reference and performance baseline. Column-major storage matches Fortran/BLAS conventions.

### `dgemm-blocked.c`
A two-level optimized implementation:

1. **Cache blocking** — the outer `square_dgemm` tiles the matrix into `BLOCK_SIZE × BLOCK_SIZE` subblocks (default: 40) to improve L1/L2 cache reuse.
2. **AVX2/FMA microkernel** — the inner `do_block` routine processes 4-row × 8-column tiles using 256-bit AVX2 registers and fused multiply-add (`_mm256_fmadd_pd`). The inner loop is unrolled by 2 along the K dimension.
3. **Alignment hints** — pointers are annotated with `__builtin_assume_aligned(..., 32)` to allow the compiler to emit aligned loads where possible.

The block size can be tuned at compile time:

```bash
cmake .. -DCMAKE_C_FLAGS="-DBLOCK_SIZE=64"
```

### `dgemm-blas.c`
Wraps `cblas_dgemm` from the system BLAS. This is the performance ceiling—vendor-tuned BLAS (e.g., Cray LibSci on Perlmutter) exploits all available hardware features.

---

## Correctness Verification

The benchmarking harness in `benchmark.cpp` automatically checks correctness after each size. It computes the error bound:

```
|C_blocked - C_blas| ≤ 3 * ε_machine * n * |A| * |B|
```

If any element exceeds this bound, the program prints `*** FAILURE ***` and exits with a non-zero status. All three implementations must pass this check.

---

## Packaging for Submission

Set your group number at configure time to enable CPack:

```bash
cmake .. -DGROUP_NO=07
make package
```

This produces `CS5220Group07_hw1.tar.gz` containing `dgemm-blocked.c` and your report PDF (place `CS5220Group07_hw1.pdf` in the build directory before running `make package`).

---

## Notes on Perlmutter CPU Architecture

Perlmutter's CPU nodes use AMD EPYC Milan (Zen 3) processors. The build flags `-march=znver3` enable all relevant ISA extensions including AVX2 and FMA. The theoretical peak of **56 GFlop/s per core** assumes:
- 3.5 GHz clock (pinned via `--cpu-freq=3500000` in the SLURM script)
- 4 doubles per AVX2 register
- 2 FMA execution units
- 2 flops per FMA operation
