//! Profiling target for the small-M transposed-weight kernel against the
//! tiled `_bt` kernel it replaces, on the shapes behind the `MAX_SMALL_N`
//! table in `src/runtime/cuda/kernels/loader/matmul_f32_smallm.rs`.
//!
//! Every shape prints one line, then launches `matmul_f32_smallm_bt`
//! `ITERS` times and the tiled `_bt` kernel `ITERS` times, with a
//! synchronize between, so an nsys trace lines up with the printed order.
//! Nothing is timed here: nsys reports each launch. The small-M launcher
//! declines shapes outside `smallm_applies` (its wave bound depends on this
//! device's SM count) and the module cache that bypasses it is
//! crate-private, so on those shapes only the tiled kernel runs and the
//! line says so.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_matmul_smallm_profile
//! nsys profile --stats=true -o smallm_profile \
//!     ./target/release/examples/cuda_matmul_smallm_profile
//! nsys stats --report cuda_gpu_trace smallm_profile.nsys-rep
//! ncu --kernel-name regex:smallm --launch-count 8 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
//!     ./target/release/examples/cuda_matmul_smallm_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::dtype::DType;
    use numr::ops::RandomOps;
    use numr::runtime::cuda::kernels::{
        SmallmLimits, launch_matmul_kernel_bt, launch_matmul_smallm_bt_kernel, smallm_applies,
    };
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    use numr::runtime::{Device, RuntimeClient};
    use numr::tensor::Tensor;

    /// Enough launches for a profiler to sample, few enough that a
    /// replay-based collection stays quick.
    const ITERS: usize = 8;
    const ROWS: [usize; 6] = [1, 4, 8, 16, 32, 64];
    /// `(N, K)`: the table's rows plus widths between 256 and 5120.
    const SHAPES: [(usize, usize); 9] = [
        (48, 1000),
        (48, 5120),
        (48, 17408),
        (96, 5120),
        (256, 5120),
        (512, 5120),
        (1024, 5120),
        (5120, 5120),
        (5120, 17408),
    ];

    let device = CudaDevice::new(0);
    let client = match CudaClient::new(device.clone()) {
        Ok(client) => client,
        Err(e) => {
            eprintln!("CUDA client: {e:?}");
            std::process::exit(1);
        }
    };
    let limits = SmallmLimits::of(&client);
    println!("SMs: {}", limits.sm_count);
    println!("tuned max waves: {}", limits.max_waves);

    for &m in &ROWS {
        for &(n, k) in &SHAPES {
            let a = client.rand(&[m, k], DType::F32).expect("A");
            let w = client.rand(&[n, k], DType::F32).expect("W");
            let out = Tensor::<CudaRuntime>::empty(&[m, n], DType::F32, &device).expect("C");
            client.synchronize();

            let small = smallm_applies(DType::F32, m, n, 1, limits);
            println!(
                "M={m} N={n} K={k}: {} x{ITERS}, then tiled bt x{ITERS}",
                if small {
                    "small-M"
                } else {
                    "small-M declined (smallm_applies)"
                }
            );

            if small {
                for _ in 0..ITERS {
                    let ran = unsafe {
                        launch_matmul_smallm_bt_kernel(
                            &client,
                            DType::F32,
                            a.ptr(),
                            w.ptr(),
                            out.ptr(),
                            m,
                            n,
                            k,
                        )
                    }
                    .expect("small-M launch");
                    assert!(ran, "M={m} N={n} K={k}: small-M launcher declined");
                }
                client.synchronize();
            }

            for _ in 0..ITERS {
                let ran = unsafe {
                    launch_matmul_kernel_bt(
                        client.context(),
                        client.stream(),
                        device.id(),
                        DType::F32,
                        a.ptr(),
                        w.ptr(),
                        out.ptr(),
                        m,
                        n,
                        k,
                    )
                }
                .expect("tiled bt launch");
                assert!(ran, "M={m} N={n} K={k}: tiled bt launcher declined");
            }
            client.synchronize();
        }
    }
}
