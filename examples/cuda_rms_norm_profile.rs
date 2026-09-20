//! Timing target for the CUDA RMSNorm kernel on decode and prefill row counts.
//!
//! Every shape prints one line with the fastest launch in microseconds, once
//! for the raw kernel launch on preallocated buffers and once for the
//! `rms_norm` op (which also allocates its output). Each number is the minimum
//! over `ITERS` event-bracketed launches, the estimator least affected by a
//! loaded machine. An event bracket carries the event records themselves, so
//! on a microsecond kernel it overstates the duration and quantizes it; the
//! nsys trace below reports the kernel alone, and every shape's launches sit
//! in printed order, `ITERS + 1` raw then `ITERS + 1` through the op.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_rms_norm_profile
//! ./target/release/examples/cuda_rms_norm_profile
//! nsys profile --stats=false -o rms_norm_profile \
//!     ./target/release/examples/cuda_rms_norm_profile
//! nsys stats --report cuda_gpu_trace --format csv rms_norm_profile.nsys-rep
//! ncu --kernel-name regex:rms_norm --launch-count 4 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
//!     ./target/release/examples/cuda_rms_norm_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::dtype::DType;
    use numr::ops::{NormalizationOps, RandomOps};
    use numr::runtime::cuda::kernels::launch_rms_norm;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, time_launches};
    use numr::runtime::{Device, RuntimeClient};
    use numr::tensor::Tensor;

    /// Timed launches per shape.
    const ITERS: usize = 200;
    /// `(batch, hidden)`: decode rows at two widths, then short and long prefill.
    const SHAPES: [(usize, usize); 4] = [(1, 5120), (1, 4096), (5, 5120), (297, 5120)];
    const EPS: f32 = 1e-5;

    let device = CudaDevice::new(0);
    let client = match CudaClient::new(device.clone()) {
        Ok(client) => client,
        Err(e) => {
            eprintln!("CUDA client: {e:?}");
            std::process::exit(1);
        }
    };

    println!(
        "{:>8} {:>8} {:>14} {:>14}",
        "batch", "hidden", "kernel us", "op us"
    );
    for &(batch, hidden) in &SHAPES {
        let input = client.rand(&[batch, hidden], DType::F32).expect("input");
        let weight = client.rand(&[hidden], DType::F32).expect("weight");
        let out = Tensor::<CudaRuntime>::empty(&[batch, hidden], DType::F32, &device).expect("out");
        client.synchronize();

        let kernel_us = time_launches(&client, ITERS, || unsafe {
            launch_rms_norm(
                client.context(),
                client.stream(),
                device.id(),
                DType::F32,
                input.ptr(),
                weight.ptr(),
                out.ptr(),
                batch,
                hidden,
                EPS,
            )
        })
        .expect("kernel timing");

        let op_us = time_launches(&client, ITERS, || {
            std::hint::black_box(client.rms_norm(&input, &weight, EPS)?);
            Ok(())
        })
        .expect("op timing");

        println!("{batch:>8} {hidden:>8} {kernel_us:>14.2} {op_us:>14.2}");
    }
}
