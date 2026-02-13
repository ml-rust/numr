#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_numr(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

fn rand_complex(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    let real = client.rand(&[n], DType::F64).unwrap();
    client.cast(&real, DType::Complex128).unwrap()
}

// ---------------------------------------------------------------------------
// Group 1: Matmul Thread Scaling (512x512 matrix)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_threads_512", args = [1, 2, 4, 8])]
fn matmul_512x512(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let a = rand_numr(&[512, 512], &device);
    let bm = rand_numr(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 2: Batched Matmul Thread Scaling (32 x 128x128)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_batch_threads", args = [1, 2, 4, 8])]
fn matmul_batched_32x128x128(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let a = rand_numr(&[32, 128, 128], &device);
    let bm = rand_numr(&[32, 128, 128], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 3: Reduce Sum Thread Scaling (1M elements)
// ---------------------------------------------------------------------------

#[flux::bench(group = "reduce_sum_1m_threads", args = [1, 2, 4, 8])]
fn reduce_sum_1m(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 4: Reduce Sum Thread Scaling (10M elements)
// ---------------------------------------------------------------------------

#[flux::bench(group = "reduce_sum_10m_threads", args = [1, 2, 4, 8])]
fn reduce_sum_10m(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let t = rand_numr(&[10_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 5: Reduce Mean Thread Scaling (1M elements)
// ---------------------------------------------------------------------------

#[flux::bench(group = "reduce_mean_1m_threads", args = [1, 4])]
fn reduce_mean_1m(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.mean(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 6: FFT Thread Scaling (16384 elements)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_threads_16k", args = [1, 2, 4, 8])]
fn fft_16384(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let t = rand_complex(16384, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// Group 7: Batched FFT Thread Scaling (64 x 1024)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_batch_threads", args = [1, 2, 4, 8])]
fn fft_batched_64x1024(b: &mut Bencher, threads: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(threads), None));
    let real = client.rand(&[64, 1024], DType::F64).unwrap();
    let t = client.cast(&real, DType::Complex128).unwrap();
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// Group 8: Chunk Size Sensitivity (4 threads, reduce sum 10M)
// ---------------------------------------------------------------------------

#[flux::bench(group = "reduce_sum_chunk_sensitivity", args = [256, 1024, 4096, 16384])]
fn reduce_sum_10m_chunk(b: &mut Bencher, chunk_size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(4), Some(chunk_size)));
    let t = rand_numr(&[10_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// Group 9: Overhead Benchmarks (default vs custom config)
// ---------------------------------------------------------------------------

#[flux::bench(group = "overhead_matmul")]
fn matmul_512x512_default(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[512, 512], &device);
    let bm = rand_numr(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "overhead_matmul")]
fn matmul_512x512_custom_same(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client =
        CpuRuntime::default_client(&device).with_parallelism(ParallelismConfig::new(None, None));
    let a = rand_numr(&[512, 512], &device);
    let bm = rand_numr(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "overhead_reduce")]
fn reduce_sum_1m_default(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "overhead_reduce")]
fn reduce_sum_1m_custom_same(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client =
        CpuRuntime::default_client(&device).with_parallelism(ParallelismConfig::new(None, None));
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "overhead_fft")]
fn fft_1024_default(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(1024, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "overhead_fft")]
fn fft_1024_custom_same(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client =
        CpuRuntime::default_client(&device).with_parallelism(ParallelismConfig::new(None, None));
    let t = rand_complex(1024, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// Comparisons: Thread Scaling
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "matmul_512_threads",
    title = "Matmul 512×512 Thread Scaling",
    benchmarks = [
        "matmul_512x512@1",
        "matmul_512x512@2",
        "matmul_512x512@4",
        "matmul_512x512@8"
    ],
    baseline = "matmul_512x512@1",
    metric = "mean"
)]
struct MatmulScaling512;

#[flux::compare(
    id = "matmul_batch_threads",
    title = "Matmul Batched 32×128×128 Thread Scaling",
    benchmarks = [
        "matmul_batched_32x128x128@1",
        "matmul_batched_32x128x128@2",
        "matmul_batched_32x128x128@4",
        "matmul_batched_32x128x128@8"
    ],
    baseline = "matmul_batched_32x128x128@1",
    metric = "mean"
)]
struct MatmulBatchScaling;

#[flux::compare(
    id = "reduce_sum_1m_threads",
    title = "Reduce Sum 1M Thread Scaling",
    benchmarks = [
        "reduce_sum_1m@1",
        "reduce_sum_1m@2",
        "reduce_sum_1m@4",
        "reduce_sum_1m@8"
    ],
    baseline = "reduce_sum_1m@1",
    metric = "mean"
)]
struct ReduceSum1MScaling;

#[flux::compare(
    id = "reduce_sum_10m_threads",
    title = "Reduce Sum 10M Thread Scaling",
    benchmarks = [
        "reduce_sum_10m@1",
        "reduce_sum_10m@2",
        "reduce_sum_10m@4",
        "reduce_sum_10m@8"
    ],
    baseline = "reduce_sum_10m@1",
    metric = "mean"
)]
struct ReduceSum10MScaling;

#[flux::compare(
    id = "fft_16k_threads",
    title = "FFT 16384 Thread Scaling",
    benchmarks = [
        "fft_16384@1",
        "fft_16384@2",
        "fft_16384@4",
        "fft_16384@8"
    ],
    baseline = "fft_16384@1",
    metric = "mean"
)]
struct FFT16KScaling;

#[flux::compare(
    id = "fft_batch_threads",
    title = "FFT Batched 64×1024 Thread Scaling",
    benchmarks = [
        "fft_batched_64x1024@1",
        "fft_batched_64x1024@2",
        "fft_batched_64x1024@4",
        "fft_batched_64x1024@8"
    ],
    baseline = "fft_batched_64x1024@1",
    metric = "mean"
)]
struct FFTBatchScaling;

// ---------------------------------------------------------------------------
// Comparisons: Chunk Size Sensitivity
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "chunk_size_reduce",
    title = "Reduce Sum 10M Chunk Size Sensitivity",
    benchmarks = [
        "reduce_sum_10m_chunk@256",
        "reduce_sum_10m_chunk@1024",
        "reduce_sum_10m_chunk@4096",
        "reduce_sum_10m_chunk@16384"
    ],
    baseline = "reduce_sum_10m_chunk@1024",
    metric = "mean"
)]
struct ChunkSizeReduce;

// ---------------------------------------------------------------------------
// Comparisons: Overhead
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "overhead_matmul",
    title = "Matmul 512×512 Configuration Overhead",
    benchmarks = ["matmul_512x512_default", "matmul_512x512_custom_same"],
    baseline = "matmul_512x512_default",
    metric = "mean"
)]
struct OverheadMatmul;

#[flux::compare(
    id = "overhead_reduce",
    title = "Reduce Sum 1M Configuration Overhead",
    benchmarks = ["reduce_sum_1m_default", "reduce_sum_1m_custom_same"],
    baseline = "reduce_sum_1m_default",
    metric = "mean"
)]
struct OverheadReduce;

#[flux::compare(
    id = "overhead_fft",
    title = "FFT 1024 Configuration Overhead",
    benchmarks = ["fft_1024_default", "fft_1024_custom_same"],
    baseline = "fft_1024_default",
    metric = "mean"
)]
struct OverheadFFT;

// ---------------------------------------------------------------------------
// Synthetic Metrics: Scaling Efficiency
// ---------------------------------------------------------------------------

#[flux::synthetic(
    id = "matmul_512_4t_speedup",
    formula = "matmul_512x512@1 / matmul_512x512@4",
    unit = "x"
)]
struct Matmul512SpeedupRatio;

#[flux::synthetic(
    id = "reduce_sum_1m_4t_speedup",
    formula = "reduce_sum_1m@1 / reduce_sum_1m@4",
    unit = "x"
)]
struct ReduceSum1M4tSpeedup;

#[flux::synthetic(
    id = "reduce_sum_10m_4t_speedup",
    formula = "reduce_sum_10m@1 / reduce_sum_10m@4",
    unit = "x"
)]
struct ReduceSum10M4tSpeedup;

#[flux::synthetic(
    id = "fft_16k_4t_speedup",
    formula = "fft_16384@1 / fft_16384@4",
    unit = "x"
)]
struct FFT16K4tSpeedup;

// ---------------------------------------------------------------------------
// Synthetic Metrics: Configuration Overhead
// ---------------------------------------------------------------------------

#[flux::synthetic(
    id = "matmul_overhead_ratio",
    formula = "matmul_512x512_custom_same / matmul_512x512_default",
    unit = "x"
)]
struct MatmulOverheadRatio;

#[flux::synthetic(
    id = "reduce_overhead_ratio",
    formula = "reduce_sum_1m_custom_same / reduce_sum_1m_default",
    unit = "x"
)]
struct ReduceOverheadRatio;

#[flux::synthetic(
    id = "fft_overhead_ratio",
    formula = "fft_1024_custom_same / fft_1024_default",
    unit = "x"
)]
struct FFTOverheadRatio;

// ---------------------------------------------------------------------------
// Verification Gates: No Regression from Threading
// ---------------------------------------------------------------------------
// Single-operation kernels (batch_size=1) are inherently sequential.
// Threading only helps batched workloads. Verify that enabling threads
// doesn't cause regression (overhead must stay within 15%).

#[flux::verify(
    expr = "matmul_512x512@4 / matmul_512x512@1 < 1.15",
    severity = "warning"
)]
struct VerifyMatmul512NoRegression;

#[flux::verify(
    expr = "reduce_sum_10m@4 / reduce_sum_10m@1 < 1.15",
    severity = "warning"
)]
struct VerifyReduceSum10MNoRegression;

#[flux::verify(expr = "fft_16384@4 / fft_16384@1 < 1.15", severity = "warning")]
struct VerifyFFT16KNoRegression;

// ---------------------------------------------------------------------------
// Verification Gates: Configuration Overhead (must be strict)
// ---------------------------------------------------------------------------

#[flux::verify(
    expr = "matmul_512x512_custom_same / matmul_512x512_default < 1.10",
    severity = "warning"
)]
struct VerifyMatmulOverhead;

#[flux::verify(
    expr = "reduce_sum_1m_custom_same / reduce_sum_1m_default < 1.10",
    severity = "warning"
)]
struct VerifyReduceOverhead;

#[flux::verify(
    expr = "fft_1024_custom_same / fft_1024_default < 1.10",
    severity = "warning"
)]
struct VerifyFFTOverhead;

fn main() {
    fluxbench::run().unwrap();
}

// ---------------------------------------------------------------------------
// Unit Tests: Numerical Parity
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use numr::prelude::*;

    /// Matmul must produce bit-identical results regardless of thread count.
    /// Verifies that work partitioning doesn't affect floating-point accumulation order.
    #[test]
    fn test_matmul_parallelism_numerical_parity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = client.rand(&[512, 512], DType::F32).unwrap();
        let b = client.rand(&[512, 512], DType::F32).unwrap();

        let result_1t = client
            .with_parallelism(ParallelismConfig::new(Some(1), None))
            .matmul(&a, &b)
            .unwrap()
            .to_vec::<f32>();

        let result_4t = client
            .with_parallelism(ParallelismConfig::new(Some(4), None))
            .matmul(&a, &b)
            .unwrap()
            .to_vec::<f32>();

        let result_8t = client
            .with_parallelism(ParallelismConfig::new(Some(8), None))
            .matmul(&a, &b)
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(
            result_1t, result_4t,
            "Matmul results differ between 1-thread and 4-thread"
        );
        assert_eq!(
            result_1t, result_8t,
            "Matmul results differ between 1-thread and 8-thread"
        );
    }

    /// Reduction sum must produce bit-identical results regardless of thread count.
    /// Verifies that parallel chunk boundaries don't affect accumulation.
    #[test]
    fn test_reduce_sum_parallelism_numerical_parity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let t = client.rand(&[1_000_000], DType::F32).unwrap();

        let result_1t = client
            .with_parallelism(ParallelismConfig::new(Some(1), None))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        let result_4t = client
            .with_parallelism(ParallelismConfig::new(Some(4), None))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        let result_8t = client
            .with_parallelism(ParallelismConfig::new(Some(8), None))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(
            result_1t, result_4t,
            "Sum results differ between 1-thread and 4-thread"
        );
        assert_eq!(
            result_1t, result_8t,
            "Sum results differ between 1-thread and 8-thread"
        );
    }

    /// FFT must produce bit-identical results regardless of thread count.
    /// Single-batch FFTs are sequential, but batched FFTs split across threads.
    #[test]
    fn test_fft_parallelism_numerical_parity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let real = client.rand(&[16384], DType::F64).unwrap();
        let t = client.cast(&real, DType::Complex128).unwrap();

        let result_1t = client
            .with_parallelism(ParallelismConfig::new(Some(1), None))
            .fft(&t, FftDirection::Forward, FftNormalization::Backward)
            .unwrap()
            .to_vec::<f64>();

        let result_4t = client
            .with_parallelism(ParallelismConfig::new(Some(4), None))
            .fft(&t, FftDirection::Forward, FftNormalization::Backward)
            .unwrap()
            .to_vec::<f64>();

        let result_8t = client
            .with_parallelism(ParallelismConfig::new(Some(8), None))
            .fft(&t, FftDirection::Forward, FftNormalization::Backward)
            .unwrap()
            .to_vec::<f64>();

        assert_eq!(
            result_1t, result_4t,
            "FFT results differ between 1-thread and 4-thread"
        );
        assert_eq!(
            result_1t, result_8t,
            "FFT results differ between 1-thread and 8-thread"
        );
    }

    #[test]
    fn test_chunk_size_numerical_parity() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let t = client.rand(&[10_000_000], DType::F32).unwrap();

        let result_chunk_256 = client
            .with_parallelism(ParallelismConfig::new(Some(4), Some(256)))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        let result_chunk_1024 = client
            .with_parallelism(ParallelismConfig::new(Some(4), Some(1024)))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        let result_chunk_4096 = client
            .with_parallelism(ParallelismConfig::new(Some(4), Some(4096)))
            .sum(&t, &[0], false)
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(
            result_chunk_256, result_chunk_1024,
            "Sum results differ between chunk_size=256 and chunk_size=1024"
        );
        assert_eq!(
            result_chunk_1024, result_chunk_4096,
            "Sum results differ between chunk_size=1024 and chunk_size=4096"
        );
    }
}
