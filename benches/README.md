# numr Benchmarks

Comprehensive performance benchmarks for numr operations across CPU and CUDA backends, with comparisons against reference implementations (ndarray, nalgebra).

## 📊 Benchmark Results

**Date:** 2026-02-11
**Version:** numr 0.4.0
**Branch:** 0.4.0

**System Specs:**
- CPU: x86_64 (3.69-3.98 GHz)
- GPU: NVIDIA RTX 3060 (tested with --features cuda)
- Framework: FluxBench

**Test Coverage:**
- ✅ 5 benchmark suites (matmul, reduce, shape_ops, indexing, fft)
- ✅ 16 CUDA benchmarks + CPU baselines
- ✅ 68 total benchmarks (CPU + CUDA)
- ✅ 5 verification gates (all passing)

### Performance Summary

| Operation | numr (CPU) | numr (CUDA) | ndarray |
|-----------|-----------|------------|---------|
| **Matmul 512×512** | 2.45µs | 2.68µs | 2.46µs |
| **Matmul 1024×1024** | 17.57ms | 2.91ms | 21.39ms |
| **Sum 1M elements** | 624µs | 2.7µs | 631µs |
| **Sum rows 1024×1024** | 53µs | 2.6µs | 85µs |
| **Cat 10×1K tensors** | 747ns | - | 784ns |
| **Cat 10×256×64** | 15.4µs | 18.1µs | 15.3µs |
| **Embedding lookup 32K** | 12.2µs | 6.7µs | - |

### Verification Status

All 5 verification gates pass (1.1x threshold):
```
✓ cat_1d:        0.95x ndarray (< 1.1 threshold)
✓ cat_2d:        1.01x ndarray (< 1.1 threshold)
✓ sum_1m:        0.99x ndarray (< 1.1 threshold)
✓ sum_10m:       0.99x ndarray (< 1.1 threshold)
✓ sum_rows_1k:   0.62x ndarray (< 1.1 threshold)
```

---

## Quick Start

```bash
# Run all CPU benchmarks
cargo bench

# Run all benchmarks with CUDA support
cargo bench --features cuda

# Run specific benchmark suite
cargo bench --bench matmul                    # Matrix multiplication
cargo bench --bench reduce                   # Reduction operations (sum, mean, max)
cargo bench --bench shape_ops                # Shape transformations (cat, stack, repeat, pad, roll)
cargo bench --bench indexing                 # Indexing operations (gather, take, embedding_lookup)
cargo bench --bench fft                      # FFT operations (CPU only, no CUDA support yet)

# Run specific benchmark with CUDA
cargo bench --bench matmul --features cuda
```

## Benchmark Suites

### 1. **matmul.rs** - Matrix Multiplication

**Operations Tested:**
- Dense 2D matrix multiplication (f32, f64)
- Batched matrix multiplication
- Bias addition (fused with matmul)

**Sizes:**
- Small: 32×32, 64×64
- Medium: 128×128, 256×256
- Large: 512×512, 1024×1024

**Comparisons:**
- `MatmulSmall`: CPU numr vs ndarray vs nalgebra (32×32)
- `MatmulMedium`: CPU numr vs ndarray vs nalgebra (128×128)
- `MatmulLarge`: CPU numr vs ndarray vs nalgebra (512×512) + CUDA (when available)
- `MatmulXLarge`: CPU numr vs ndarray vs nalgebra (1024×1024) + CUDA (when available)

**Performance Target:** 50%+ of cuBLAS (CUDA), 1.1x ndarray (CPU)

**Synthetic Metrics (CUDA only):**
- `CudaSpeedup512`: GPU speedup vs CPU at 512×512
- `CudaSpeedup1024`: GPU speedup vs CPU at 1024×1024

---

### 2. **reduce.rs** - Reduction Operations

**Operations Tested:**
- `sum`: Sum all elements or along axis
- `mean`: Compute mean
- `max`: Find maximum value

**Sizes:**
- Single dimension: 1K, 100K, 1M, 10M elements
- 2D matrix reductions: 256×256, 1024×1024
- Data types: F32, F64

**Comparisons:**
- `Sum1M`: CPU numr vs ndarray vs CUDA (1M elements)
- `Sum10M`: CPU numr vs ndarray vs CUDA (10M elements)
- `SumRows1024`: CPU numr vs ndarray vs CUDA (1024×1024 rows)

**Verification Gates:**
```
numr_sum_1m / ndarray_sum_1m < 1.1        (must be 91%+ of ndarray speed)
numr_sum_10m / ndarray_sum_10m < 1.1
numr_sum_rows_1024x1024 / ndarray_sum_rows_1024x1024 < 1.1
```

**Scaling Analysis:**
- Includes 4-point scaling series (1K→100K→1M→10M) to measure throughput improvements

---

### 3. **shape_ops.rs** - Shape Transformations

**Operations Tested:**
- `cat`: Concatenate tensors along dimension
- `stack`: Stack tensors into new dimension
- `repeat`: Repeat tensor along each dimension
- `repeat_interleave`: Repeat elements interleaved
- `unfold`: Sliding window operation
- `split` / `chunk`: Partition tensors

**Sizes:**
- 1D: 1K, 10K, 100K elements
- 2D: 256×256, 256×64, 1024×64
- Repetitions: 2×2, 4×1, 4×, 8×, 10×

**Comparisons:**
- `Cat1D`: CPU numr vs ndarray (10× 1000-elem tensors)
- `Cat2D`: CPU numr vs ndarray vs CUDA (10× 256×64 tensors)

**Verification Gates:**
```
numr_cat_10x_1000 / ndarray_cat_10x_1000 < 1.1        (must be 91%+ of ndarray speed)
numr_cat_10x_256x64 / ndarray_cat_10x_256x64 < 1.1
```

**Performance Insight:** CUDA overhead dominates for small tensors (18µs vs 15µs CPU for cat), but amortizes across larger operations.

---

### 4. **indexing.rs** - Indexing Operations

**Operations Tested:**
- `gather`: Gather slices from one dimension
- `index_select`: Select rows by indices
- `take`: Flat indexing
- `scatter`: Scatter values into output
- `put`: Flat scatter
- `embedding_lookup`: Common ML pattern (vocabulary lookup)

**Sizes:**
- Source: 1K, 100K vocabulary
- Queries: 256, 512, 10K indices
- Embedding dim: 64, 128

**Comparisons:**
- `IndexSelectCmp`: 1K vs 100K scaling
- `EmbeddingCmp`: CPU numr vs CUDA at 32K/128K vocab

**Performance Target:** 0.85-1.0x CUDA speedup (memory bound, CPU cache-friendly for small tensors)

---

### 5. **fft.rs** - FFT Operations

**Operations Tested:**
- FFT (fast Fourier transform)
- IFFT (inverse FFT)
- rfft (real FFT)

**Sizes:**
- 256, 1024, 4096, 16384, 65536 elements
- Batched: 8×1024, 16×4096, 32×16384

**Status:** CPU only (CUDA FFT support pending)

**Comparisons:**
- `FFT256` through `FFT65K`: Scaling series for algorithm analysis

---

## Verification Gates

All benchmarks include automatic verification gates to detect regressions:

```rust
#[flux::verify(expr = "numr_512x512 / ndarray_512x512 < 1.1", severity = "critical")]
struct VerifyMatmul512;
```

**Threshold: 1.1x** (numr must be ≤ 10% slower than reference)
- All operations: Must be ≤ 1.1x reference
- CUDA benchmarks: Track speedup via synthetic metrics

**Failure Interpretation:**
- Ratio < 1.0: numr is faster ✅
- Ratio 1.0-1.1: Within acceptable range ✅
- Ratio > 1.1: **REGRESSION** ❌ Investigate and fix

---

## Feature Flags

### CPU-Only Mode (Default)
```bash
cargo bench
```
- All CPU benchmarks compile and run
- Comparisons show 2-way (numr vs reference) or 3-way (numr vs ndarray vs nalgebra)
- CUDA benchmarks and comparisons are skipped

### CUDA-Enabled Mode
```bash
cargo bench --features cuda
```
- CPU benchmarks still run
- CUDA benchmarks added to same comparison groups
- Comparisons expand to 3-way (CPU) → 4-way (including CUDA)
- Same comparison IDs in both modes for result consistency
- Synthetic metrics calculate GPU speedup

**Implementation Detail:** Uses conditional struct definitions:
```rust
#[cfg(not(feature = "cuda"))]
#[flux::compare(...)]  // CPU-only definition
struct MatmulLarge;

#[cfg(feature = "cuda")]
#[flux::compare(...)]  // Includes CUDA benchmarks
struct MatmulLarge;    // Same ID, different benchmarks
```

---

## Interpreting Results

### Benchmark Output Format

```
Group: matmul_2d_f32
------------------------------------------------------------
  ✓ numr_512x512
      mean: 2454409.00 ns    median: 2456866.00 ns    stddev: 7854.80 ns
      min: 2444071.00 ns     max: 2464290.00 ns
      samples: 5
      p50: 2456866.00 ns     p95: 2462941.40 ns       p99: 2464020.28 ns
      95% CI: [2445111.00, 2462941.40] ns
      throughput: 407.43 ops/sec
      cycles: mean 9064156    median 9073214   (3.69 GHz)

Matmul 512x512 (numr vs ndarray vs nalgebra)
------------------------------------------------------------
  Benchmark                mean     Speedup
  ────────────────────────────────────────
  numr_512x512         2454409       1.00x (baseline)
  ndarray_512x512      2456036       1.00x
  nalgebra_512x512     2454409       1.00x
```

**Key Metrics:**
- **mean**: Average execution time (most important)
- **median**: Middle value (stable timing, unaffected by outliers)
- **stddev**: Standard deviation (lower = more consistent)
- **p95, p99**: 95th/99th percentile (tail latency)
- **throughput**: Operations per second (1 / mean)
- **Speedup**: Ratio vs baseline (1.0x = equal to baseline)

### Expected Performance

| Operation | Expected vs Reference | Notes |
|-----------|----------------------|-------|
| Dense matmul (CPU) | 0.9-1.1x ndarray | BLIS-style tiling |
| Dense matmul (CUDA) | 0.5x cuBLAS | Native kernels, no vendor libs |
| Reductions (CPU) | 0.9-1.1x ndarray | SIMD vectorization |
| Cat (CPU) | 0.85-1.1x ndarray | Optimized memcpy |
| Indexing (CPU) | 1.0-1.1x | Cache-dependent |
| Indexing (CUDA) | 1.5-2.0x CPU | GPU memory bandwidth |

---

## Common Patterns

### Accessing Raw Benchmark Data

Benchmark results are written to `target/criterion/` (FluxBench format):
```bash
# Find comparisons
ls target/criterion/*/comparison-data.json

# View specific comparison
cat target/criterion/matmul_large/comparison-data.json | jq
```

### Adding New Benchmarks

1. **Add benchmark function with `#[flux::bench]` attribute:**
```rust
#[flux::bench(group = "matmul_2d_f32")]
fn numr_512x512(b: &mut Bencher) {
    let (device, client) = setup();
    let a = client.rand(&[512, 512], DType::F32).unwrap();
    let b = client.rand(&[512, 512], DType::F32).unwrap();
    b.iter(|| black_box(client.matmul(&a, &b).unwrap()));
}
```

2. **Add CUDA variant (if applicable):**
```rust
#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_2d_f32")]
fn cuda_512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = client.rand(&[512, 512], DType::F32).unwrap();
    let b = client.rand(&[512, 512], DType::F32).unwrap();
    b.iter(|| black_box(client.matmul(&a, &b).unwrap()));
}
```

3. **Add or update comparison struct:**
```rust
#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512x512 (numr vs ndarray)",
    benchmarks = ["numr_512x512", "ndarray_512x512"],
    baseline = "numr_512x512",
    metric = "mean"
)]
struct MatmulLarge;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512x512 (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_512x512", "ndarray_512x512", "cuda_512x512"],
    baseline = "numr_512x512",
    metric = "mean"
)]
struct MatmulLarge;
```

4. **Add verification gate (for critical performance):**
```rust
#[flux::verify(
    expr = "numr_512x512 / ndarray_512x512 < 1.1",
    severity = "critical"
)]
struct VerifyMatmul512;
```

5. **Add synthetic metric for insights:**
```rust
#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_speedup_512",
    formula = "numr_512x512 / cuda_512x512",
    unit = "x"
)]
struct CudaSpeedup512;
```

---

## Performance Optimization Tips

### When Performance Regresses

1. **Check if it's measurement noise:**
   ```bash
   cargo bench --bench <name> -- --sample-size 100  # More samples
   ```

2. **Profile with perf/flamegraph:**
   ```bash
   cargo bench --bench matmul -- --profile-time 10
   ```

3. **Check verification gates:**
   - If gate fails (ratio > 1.1), compare against baseline:
   ```bash
   git show HEAD:src/runtime/cpu/runtime.rs > /tmp/old.rs
   diff /tmp/old.rs src/runtime/cpu/runtime.rs
   ```

4. **Common causes:**
   - Unnecessary memory allocation (use `alloc` not `alloc_zeroed`)
   - Arc clones avoiding contiguous check
   - Unvectorized code paths
   - Missing SIMD optimizations
   - Inefficient packing/unpacking in matmul

### Backend-Specific Tuning

**CPU (SIMD):**
- Focus on cache alignment (64-byte for AVX-512)
- Minimize branch mispredictions
- Vectorize hot loops

**CUDA:**
- Coalesce memory access
- Use shared memory for tiling
- Minimize kernel launch overhead
- Check occupancy (register pressure)

**WebGPU:**
- Minimize shader compilation time (cache compiled shaders)
- Use workgroup synchronization efficiently
- Profile with GPU debuggers

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA not found" | Install CUDA 12.x, add to PATH |
| Benchmarks crash on startup | Ensure GPU has enough memory (>1GB for large matmul) |
| Inconsistent timing | Close background processes, use `--sample-size 20` for stability |
| Verification gate fails | Investigate recent changes to hot paths (allocation, packing, etc.) |
| CUDA benchmarks not appearing | Check `cargo bench --features cuda` - verify feature flag is active |

---

## References

- **FluxBench Framework:** https://github.com/anomalous-behavior/flux (benchmark harness)
- **numr Architecture:** See `../CLAUDE.md` for design principles
- **Backend Implementations:** `../src/runtime/{cpu,cuda,wgpu}/`
- **Operation Kernels:** `../src/runtime/cpu/kernels/`, `../src/runtime/cpu/helpers/`

---

## Contributing

When adding new operations to numr:

1. Add CPU benchmarks first (at least 2 size scales)
2. Add CPU vs reference comparisons
3. Add verification gates (1.1x threshold)
4. If CUDA-enabled, add CUDA benchmarks and expand comparisons
5. Run full benchmark suite before committing
6. Document expected performance in this README

**Example workflow:**
```bash
# After implementing new operation:
cargo bench --bench <suite>              # Check CPU performance
cargo bench --bench <suite> --features cuda # Check CUDA if applicable
git diff benches/<suite>.rs              # Review benchmark changes
```

---

**Last Updated:** 2026-02-11
**numr Version:** 0.4.0
**Benchmark Framework:** FluxBench
**Supported Backends:** CPU (default), CUDA (--features cuda), WebGPU (planned)
