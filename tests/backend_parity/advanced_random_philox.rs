// Philox4x32-10 contract tests: known answers and cross-backend equality.

use numr::dtype::DType;
use numr::ops::AdvancedRandomOps;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Philox4x32-10 contract: known answers and cross-backend bit equality
// ============================================================================

/// Key `0x0123456789abcdef`, counter `0xffffffff`: block 0 is counter
/// `[0xffffffff, 0, 0, 0]`, block 1 carries into the hi word, `[0, 1, 0, 0]`.
/// Words derived from the Random123 Philox4x32-10 definition.
const PHILOX_KAT_KEY: u64 = 0x0123_4567_89ab_cdef;
const PHILOX_KAT_COUNTER: u64 = 0xffff_ffff;
const PHILOX_KAT_WORDS: [u32; 8] = [
    0x073402a4, 0x9775331e, 0x64c992ad, 0x325045ac, 0xd9412d6a, 0x59dd3f68, 0x9d46a012, 0x5d27ddd9,
];

/// Shapes: tails of 1-3 elements, an exact block, multi-row shapes.
const PHILOX_SHAPES: [&[usize]; 7] = [&[1], &[3], &[4], &[37], &[2, 129], &[1031], &[3, 5, 7]];

/// Counters: zero, a carry into the hi word mid-draw, a hi word as boostr
/// uses it (`c << 32`), and a wrap past `u64::MAX`.
const PHILOX_COUNTERS: [u64; 4] = [0, 0xffff_fffd, 7 << 32, u64::MAX - 2];

/// Keys: zero, small, both words set, all bits set.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
const PHILOX_KEYS: [u64; 4] = [0, 42, 0x0123_4567_89ab_cdef, u64::MAX];

fn philox_kat_f32_bits() -> Vec<u32> {
    PHILOX_KAT_WORDS
        .iter()
        .map(|&w| ((w >> 8) as f32 / 16_777_216.0).to_bits())
        .collect()
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Bit equality, naming the first differing index.
fn assert_bits_eq<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!(
            "{what}: first difference at {i}: got {:?}, want {:?}",
            got[i], want[i]
        );
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn assert_close_f64(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(1.0);
        assert!((g - w).abs() <= bound, "{what}: index {i}: {g} vs {w}");
    }
}

/// Every backend writes the known-answer words at the carry boundary.
#[test]
fn test_philox_uniform_known_answer_all_backends() {
    let want = philox_kat_f32_bits();
    let (cpu_client, _) = create_cpu_client();
    let cpu = cpu_client
        .philox_uniform(&[8], PHILOX_KAT_KEY, PHILOX_KAT_COUNTER, DType::F32)
        .unwrap()
        .to_vec::<f32>();
    assert_bits_eq(&f32_bits(&cpu), &want, "philox KAT cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|c, _| {
        let got = c
            .philox_uniform(&[8], PHILOX_KAT_KEY, PHILOX_KAT_COUNTER, DType::F32)
            .unwrap()
            .to_vec::<f32>();
        assert_bits_eq(&f32_bits(&got), &want, "philox KAT cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|w, _| {
        let got = w
            .philox_uniform(&[8], PHILOX_KAT_KEY, PHILOX_KAT_COUNTER, DType::F32)
            .unwrap()
            .to_vec::<f32>();
        assert_bits_eq(&f32_bits(&got), &want, "philox KAT wgpu");
    });
}

/// F32 `philox_uniform` is bit-identical on CPU, CUDA and WebGPU.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[test]
fn test_philox_uniform_f32_bit_parity() {
    let (cpu_client, _) = create_cpu_client();
    for shape in PHILOX_SHAPES {
        for key in PHILOX_KEYS {
            for counter in PHILOX_COUNTERS {
                let want = cpu_client
                    .philox_uniform(shape, key, counter, DType::F32)
                    .unwrap()
                    .to_vec::<f32>();
                let want = f32_bits(&want);
                let what = format!("shape={shape:?} key={key:#x} counter={counter:#x}");

                #[cfg(feature = "cuda")]
                with_cuda_backend(|c, _| {
                    let got = c
                        .philox_uniform(shape, key, counter, DType::F32)
                        .unwrap()
                        .to_vec::<f32>();
                    assert_bits_eq(&f32_bits(&got), &want, &format!("cuda {what}"));
                });

                #[cfg(feature = "wgpu")]
                with_wgpu_backend(|w, _| {
                    let got = w
                        .philox_uniform(shape, key, counter, DType::F32)
                        .unwrap()
                        .to_vec::<f32>();
                    assert_bits_eq(&f32_bits(&got), &want, &format!("wgpu {what}"));
                });
            }
        }
    }
}

/// Host memory the two-dispatch-row test needs: the reference bits, one backend
/// readback, and driver staging, each about 268 MB.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
const TWO_ROWS_MIN_AVAILABLE_BYTES: u64 = 3 << 30;

/// `MemAvailable` from `/proc/meminfo`, or `None` where it cannot be read.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn available_memory_bytes() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// Reinterpret in place, so a large draw holds one buffer, not two.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn into_f32_bits(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}

/// More blocks than one WebGPU dispatch row of 65535 workgroups x 256 threads.
/// Skips when host memory is below `TWO_ROWS_MIN_AVAILABLE_BYTES`.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[test]
fn test_philox_uniform_f32_bit_parity_two_dispatch_rows() {
    match available_memory_bytes() {
        Some(avail) if avail >= TWO_ROWS_MIN_AVAILABLE_BYTES => {}
        avail => {
            println!(
                "skipping two dispatch rows: available host memory {avail:?} bytes, \
                 need {TWO_ROWS_MIN_AVAILABLE_BYTES}"
            );
            return;
        }
    }
    let n = 65535 * 256 * 4 + 10;
    let (key, counter) = (PHILOX_KAT_KEY, 0xffff_0000u64);
    let (cpu_client, _) = create_cpu_client();
    let want = into_f32_bits(
        cpu_client
            .philox_uniform(&[n], key, counter, DType::F32)
            .unwrap()
            .to_vec::<f32>(),
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|c, _| {
        let got = into_f32_bits(
            c.philox_uniform(&[n], key, counter, DType::F32)
                .unwrap()
                .to_vec::<f32>(),
        );
        assert_bits_eq(&got, &want, "cuda two rows");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|w, device| {
        if device.max_storage_buffer_size() < (n * 4) as u64 {
            println!("skipping: adapter storage binding below {} bytes", n * 4);
            return;
        }
        let got = into_f32_bits(
            w.philox_uniform(&[n], key, counter, DType::F32)
                .unwrap()
                .to_vec::<f32>(),
        );
        assert_bits_eq(&got, &want, "wgpu two rows");
    });
}

/// Element `i` depends on `(key, counter, i)` only: a longer draw extends a shorter one.
#[test]
fn test_philox_uniform_prefix_independent_of_length() {
    fn check<F: Fn(usize) -> Vec<f32>>(draw: F, backend: &str) {
        let long = draw(1031);
        for n in [1usize, 2, 3, 5, 37, 258, 1030] {
            assert_bits_eq(&draw(n), &long[..n], &format!("{backend} prefix n={n}"));
        }
    }
    let (key, counter) = (u64::MAX, 0xffff_fffeu64);
    let (cpu_client, _) = create_cpu_client();
    check(
        |n| {
            cpu_client
                .philox_uniform(&[n], key, counter, DType::F32)
                .unwrap()
                .to_vec()
        },
        "cpu",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|c, _| {
        check(
            |n| {
                c.philox_uniform(&[n], key, counter, DType::F32)
                    .unwrap()
                    .to_vec()
            },
            "cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|w, _| {
        check(
            |n| {
                w.philox_uniform(&[n], key, counter, DType::F32)
                    .unwrap()
                    .to_vec()
            },
            "wgpu",
        );
    });
}

/// F64 output is the F32 output widened, bit-identical on CPU and CUDA.
#[test]
fn test_philox_uniform_f64_bit_parity() {
    let (cpu_client, _) = create_cpu_client();
    for shape in PHILOX_SHAPES {
        for counter in PHILOX_COUNTERS {
            let key = PHILOX_KAT_KEY;
            let want = cpu_client
                .philox_uniform(shape, key, counter, DType::F64)
                .unwrap()
                .to_vec::<f64>();
            let narrow = cpu_client
                .philox_uniform(shape, key, counter, DType::F32)
                .unwrap()
                .to_vec::<f32>();
            let widened: Vec<f64> = narrow.iter().map(|&v| v as f64).collect();
            assert_bits_eq(&want, &widened, &format!("cpu f64 widened {shape:?}"));

            #[cfg(feature = "cuda")]
            with_cuda_backend(|c, _| {
                let got = c
                    .philox_uniform(shape, key, counter, DType::F64)
                    .unwrap()
                    .to_vec::<f64>();
                assert_bits_eq(&got, &want, &format!("cuda f64 {shape:?} {counter:#x}"));
            });
        }
    }
}

/// F16 and BF16 round the 24-bit value to nearest even and stay below 1.0,
/// bit-identical on CPU and CUDA. 4096 elements reach the clamp near 1.0.
#[cfg(feature = "f16")]
#[test]
fn test_philox_uniform_half_bit_parity() {
    let (cpu_client, _) = create_cpu_client();
    for counter in PHILOX_COUNTERS {
        let key = PHILOX_KAT_KEY;
        let f16 = cpu_client
            .philox_uniform(&[4096], key, counter, DType::F16)
            .unwrap()
            .to_vec::<half::f16>();
        let bf16 = cpu_client
            .philox_uniform(&[4096], key, counter, DType::BF16)
            .unwrap()
            .to_vec::<half::bf16>();
        assert!(f16.iter().all(|v| (0.0..1.0).contains(&v.to_f64())));
        assert!(bf16.iter().all(|v| (0.0..1.0).contains(&v.to_f64())));
        let f16_bits: Vec<u16> = f16.iter().map(|v| v.to_bits()).collect();
        let bf16_bits: Vec<u16> = bf16.iter().map(|v| v.to_bits()).collect();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|c, _| {
            let got: Vec<u16> = c
                .philox_uniform(&[4096], key, counter, DType::F16)
                .unwrap()
                .to_vec::<half::f16>()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_bits_eq(&got, &f16_bits, &format!("cuda f16 {counter:#x}"));
            let got: Vec<u16> = c
                .philox_uniform(&[4096], key, counter, DType::BF16)
                .unwrap()
                .to_vec::<half::bf16>()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_bits_eq(&got, &bf16_bits, &format!("cuda bf16 {counter:#x}"));
        });
        #[cfg(not(feature = "cuda"))]
        let _ = (f16_bits, bf16_bits);
    }
}

/// `philox_randn` uses the uniform block layout on every backend. Values agree
/// within tolerance only: `ln`, `sqrt`, `cos` and `sin` round per backend.
/// CPU computes in f64. CUDA F32 computes in f32. WGSL `cos` and `sin` carry a
/// looser error bound, so WebGPU gets a wider tolerance.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[test]
fn test_philox_randn_f32_parity_within_tolerance() {
    let (cpu_client, _) = create_cpu_client();
    for shape in [&[3usize][..], &[37][..], &[1031][..]] {
        for counter in PHILOX_COUNTERS {
            let key = PHILOX_KAT_KEY;
            let want: Vec<f64> = cpu_client
                .philox_randn(shape, key, counter, DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .iter()
                .map(|&v| v as f64)
                .collect();
            let what = format!("randn {shape:?} {counter:#x}");

            #[cfg(feature = "cuda")]
            with_cuda_backend(|c, _| {
                let got: Vec<f64> = c
                    .philox_randn(shape, key, counter, DType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                assert_close_f64(&got, &want, 1e-5, &format!("cuda {what}"));
            });

            #[cfg(feature = "wgpu")]
            with_wgpu_backend(|w, _| {
                let got: Vec<f64> = w
                    .philox_randn(shape, key, counter, DType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                assert_close_f64(&got, &want, 1e-2, &format!("wgpu {what}"));
            });
        }
    }
}

/// CUDA F64, F16 and BF16 `philox_randn` match CPU within the dtype's precision.
#[cfg(feature = "cuda")]
#[test]
fn test_philox_randn_cuda_wide_and_half_parity() {
    let (cpu_client, _) = create_cpu_client();
    let (shape, key, counter) = (&[1031usize][..], PHILOX_KAT_KEY, 0xffff_fffdu64);
    let want = cpu_client
        .philox_randn(shape, key, counter, DType::F64)
        .unwrap()
        .to_vec::<f64>();
    with_cuda_backend(|c, _| {
        let got = c
            .philox_randn(shape, key, counter, DType::F64)
            .unwrap()
            .to_vec::<f64>();
        assert_close_f64(&got, &want, 1e-12, "cuda randn f64");

        #[cfg(feature = "f16")]
        {
            let got: Vec<f64> = c
                .philox_randn(shape, key, counter, DType::F16)
                .unwrap()
                .to_vec::<half::f16>()
                .iter()
                .map(|v| v.to_f64())
                .collect();
            assert_close_f64(&got, &want, 2e-3, "cuda randn f16");
            let got: Vec<f64> = c
                .philox_randn(shape, key, counter, DType::BF16)
                .unwrap()
                .to_vec::<half::bf16>()
                .iter()
                .map(|v| v.to_f64())
                .collect();
            assert_close_f64(&got, &want, 1.6e-2, "cuda randn bf16");
        }
    });
}
