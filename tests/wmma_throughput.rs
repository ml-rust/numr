//! WMMA throughput measurement test.
//! Run with:
//!   cargo build --release --features cuda,f16 --test wmma_throughput
//!   target/release/deps/wmma_throughput-* wmma --nocapture

#[cfg(all(feature = "cuda", feature = "f16"))]
mod wmma_bench {
    use numr::prelude::*;
    use std::time::Instant;

    fn tflops(m: usize, n: usize, k: usize, batch: usize, elapsed_sec: f64) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64 * batch as f64;
        (flops / elapsed_sec) / 1e12
    }

    /// Time N kernel launches with proper device sync.
    fn time_kernel<F: Fn() -> Tensor<CudaRuntime>>(f: F, warmup: usize, iters: usize) -> f64 {
        for _ in 0..warmup {
            let t = f();
            let _ = t.to_vec::<half::f16>();
        }
        let start = Instant::now();
        let mut last = f();
        for _ in 1..iters {
            last = f();
        }
        let _ = last.to_vec::<half::f16>();
        start.elapsed().as_secs_f64() / iters as f64
    }

    fn time_kernel_f32<F: Fn() -> Tensor<CudaRuntime>>(f: F, warmup: usize, iters: usize) -> f64 {
        for _ in 0..warmup {
            let t = f();
            let _ = t.to_vec::<f32>();
        }
        let start = Instant::now();
        let mut last = f();
        for _ in 1..iters {
            last = f();
        }
        let _ = last.to_vec::<f32>();
        start.elapsed().as_secs_f64() / iters as f64
    }

    // RTX 3060 F16 tensor-core theoretical peak: ~101 TFLOPS
    const PEAK_TFLOPS: f64 = 101.0;

    #[ignore = "perf benchmark; prints throughput, asserts nothing - run explicitly with --ignored"]
    #[test]
    fn measure_wmma_scores_f16() {
        // Scores: A[64,512,64] @ B[64,64,512] → C[64,512,512]  (QK^T)
        let (batch, m, n, k) = (64, 512, 512, 64);
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        let a = client.rand(&[batch, m, k], DType::F16).unwrap();
        let b = client.rand(&[batch, k, n], DType::F16).unwrap();

        let elapsed = time_kernel(|| client.matmul(&a, &b).unwrap(), 5, 50);
        let tf = tflops(m, n, k, batch, elapsed);
        println!(
            "WMMA Scores  [64,512,64]@[64,64,512]→[64,512,512]: \
             {:.3}ms  {:.2} TFLOPS  ({:.1}% of 3060 F16 TC peak {:.0}T)",
            elapsed * 1e3,
            tf,
            tf / PEAK_TFLOPS * 100.0,
            PEAK_TFLOPS
        );
    }

    #[ignore = "perf benchmark; prints throughput, asserts nothing - run explicitly with --ignored"]
    #[test]
    fn measure_wmma_context_f16() {
        // Context: A[64,512,512] @ B[64,512,64] → C[64,512,64]  (attn@V)
        let (batch, m, n, k) = (64, 512, 64, 512);
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        let a = client.rand(&[batch, m, k], DType::F16).unwrap();
        let b = client.rand(&[batch, k, n], DType::F16).unwrap();

        let elapsed = time_kernel(|| client.matmul(&a, &b).unwrap(), 5, 50);
        let tf = tflops(m, n, k, batch, elapsed);
        println!(
            "WMMA Context [64,512,512]@[64,512,64]→[64,512,64]: \
             {:.3}ms  {:.2} TFLOPS  ({:.1}% of 3060 F16 TC peak {:.0}T)",
            elapsed * 1e3,
            tf,
            tf / PEAK_TFLOPS * 100.0,
            PEAK_TFLOPS
        );
    }

    #[ignore = "perf benchmark; prints throughput, asserts nothing - run explicitly with --ignored"]
    #[test]
    fn compare_fma_vs_wmma() {
        // Same logical shapes, F32 for FMA vs F16 for WMMA
        let (batch, m, n, k) = (64, 512, 512, 64);
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        let a16 = client.rand(&[batch, m, k], DType::F16).unwrap();
        let b16 = client.rand(&[batch, k, n], DType::F16).unwrap();
        let a32 = client.rand(&[batch, m, k], DType::F32).unwrap();
        let b32 = client.rand(&[batch, k, n], DType::F32).unwrap();

        let t_wmma = time_kernel(|| client.matmul(&a16, &b16).unwrap(), 5, 50);
        let t_fma = time_kernel_f32(|| client.matmul(&a32, &b32).unwrap(), 5, 50);
        let tf_wmma = tflops(m, n, k, batch, t_wmma);
        let tf_fma = tflops(m, n, k, batch, t_fma);
        println!(
            "Compare Scores shape: WMMA(F16)={:.2}T  FMA(F32)={:.2}T  speedup={:.2}x",
            tf_wmma,
            tf_fma,
            tf_wmma / tf_fma
        );
    }
}
