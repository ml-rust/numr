# numr benchmarks

FluxBench suites for numr ops on the CPU and CUDA backends, with ndarray and
nalgebra as references. Results are not recorded here: run the suites on the
target machine and compare against the run before the change.

## Run

```bash
cargo bench                                  # every CPU suite
cargo bench --features cuda                  # CPU + CUDA
cargo bench --bench matmul                   # one suite
cargo bench --bench matmul --features cuda
cargo test --bench parallelism               # thread-count parity checks
```

Every CUDA bench calls `client.synchronize()` inside the timed closure.
Kernel launches are asynchronous, so a bench without it measures launch
overhead, not the kernel.

## Suites

| File               | Covers                                             |
| ------------------ | -------------------------------------------------- |
| `matmul.rs`        | Dense f32 matmul, batched, transposed layouts      |
| `matmul_int.rs`    | Integer matmul                                     |
| `gemm_epilogue.rs` | Fused GEMM epilogues                               |
| `reduce.rs`        | sum, mean, max over axes                           |
| `norm.rs`          | LayerNorm, RMSNorm                                 |
| `softmax.rs`       | Softmax over the last axis                         |
| `conv.rs`          | Convolutions                                       |
| `fft.rs`           | FFT (CPU)                                          |
| `shape_ops.rs`     | cat, stack, repeat, pad, roll                      |
| `indexing.rs`      | gather, take, embedding lookup                     |
| `parallelism.rs`   | Thread scaling and chunk tuning on the CPU backend |
| `ci_regression.rs` | The gate set CI runs                               |

## Gates

A suite declares its regression gates with `#[flux::verify]`:

```rust
#[flux::verify(expr = "numr_512x512 / ndarray_512x512 < 1.1", severity = "critical")]
struct VerifyMatmul512;
```

A `critical` gate over 1.1x the reference fails the run. Fix the regression, or
state in the change why the new ratio is right. Never raise a threshold to
pass.

## Adding a bench

- One file per op family under `benches/`, registered in `Cargo.toml` with
  `harness = false`.
- Build the reference (ndarray/nalgebra) case beside the numr case with the
  same shapes and dtype.
- Add a gate for every shape the change claims to speed up.
- Judge a change by the gate ratio and instruction counts, never by one
  wall-clock number from a loaded machine.
