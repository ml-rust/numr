//! WebGPU fused conv2d wall-time diagnostic: one 3x3 64-channel layer.
//!
//! Each sample runs one `conv2d_fused_into` with bias and per-channel PReLU, and
//! waits for the queue. It takes no readback. The first calls compile the
//! pipeline, so they stay out of the samples.
//!
//! Run with:
//! `cargo run --release --example wgpu_conv2d_profile --features wgpu`

#[cfg(feature = "wgpu")]
fn main() -> numr::error::Result<()> {
    use numr::dtype::DType;
    use numr::ops::{
        Conv2dConfig, ConvActivation, ConvLayout, FusedConvOps, PaddingMode, RandomOps,
    };
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
    use numr::runtime::{Runtime, RuntimeClient};
    use numr::tensor::Tensor;
    use std::time::Instant;

    const CHANNELS: usize = 64;
    const KERNEL: usize = 3;
    const WARM_RUNS: usize = 2;
    const SAMPLES: usize = 7;
    const SIZES: [(usize, usize); 3] = [(360, 640), (540, 960), (1080, 1920)];

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let dtype = DType::F32;
    let bias = client.rand(&[CHANNELS], dtype)?;
    let slope = client.rand(&[CHANNELS], dtype)?;

    println!("boundary=conv2d_fused_into + queue completion; upload and readback excluded");
    for (height, width) in SIZES {
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            let (image, kernel) = match layout {
                ConvLayout::Nchw => (
                    [1, CHANNELS, height, width],
                    [CHANNELS, CHANNELS, KERNEL, KERNEL],
                ),
                ConvLayout::Nhwc => (
                    [1, height, width, CHANNELS],
                    [CHANNELS, KERNEL, KERNEL, CHANNELS],
                ),
            };
            let input = client.rand(&image, dtype)?;
            let weight = client.rand(&kernel, dtype)?;
            let out = Tensor::<WgpuRuntime>::empty(&image, dtype, &device)?;
            let config = Conv2dConfig::new(layout).with_padding(PaddingMode::Same);
            let run = || -> numr::error::Result<()> {
                client.conv2d_fused_into(
                    &out,
                    &input,
                    &weight,
                    Some(&bias),
                    &config,
                    ConvActivation::Prelu(&slope),
                    None,
                )?;
                client.synchronize();
                Ok(())
            };

            for _ in 0..WARM_RUNS {
                run()?;
            }
            let mut samples_ms = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let start = Instant::now();
                run()?;
                samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            samples_ms.sort_by(f64::total_cmp);
            let median_ms = samples_ms[SAMPLES / 2];
            let macs = height * width * CHANNELS * CHANNELS * KERNEL * KERNEL;
            println!(
                "layout={layout:?} size={height}x{width} channels={CHANNELS} samples={SAMPLES} \
                 median_ms={median_ms:.3} min_ms={:.3} max_ms={:.3} gmac_per_s={:.1}",
                samples_ms[0],
                samples_ms[SAMPLES - 1],
                macs as f64 / median_ms / 1e6,
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("enable the `wgpu` feature");
}
