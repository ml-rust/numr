// Backend parity for the WebGPU I64 cast kernels: I64 to and from F32, I32
// and U32.
//
// WGSL has no 64-bit integer, so these kernels emulate it in u32 halves. CPU
// is the reference: it converts through f64 and then applies a Rust `as`
// cast. Every comparison is exact, on raw bits for F32 output.
//
// Each case below targets one rule of that reference:
// - I64 -> I32 and I64 -> U32 saturate. They do not wrap.
// - I64 -> F32 rounds twice, to 53 bits and then to 24 bits. The double
//   rounding probes give a different answer under a single rounding.
// - F32 -> I64 truncates, saturates at +-2^63, and maps NaN to 0.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too.
#[cfg(feature = "wgpu")]
use numr::dtype::{DType, Element};
#[cfg(feature = "wgpu")]
use numr::ops::TypeConversionOps;
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

/// I64 values at every boundary the casts treat differently.
#[cfg(feature = "wgpu")]
fn i64_edges() -> Vec<i64> {
    let p = |e: u32| 1i64 << e;
    let mut v = vec![
        0,
        1,
        -1,
        p(24) - 1,
        p(24),
        p(24) + 1,
        p(24) + 3,
        i32::MAX as i64,
        i32::MAX as i64 + 1,
        i32::MIN as i64,
        i32::MIN as i64 - 1,
        u32::MAX as i64,
        u32::MAX as i64 + 1,
        p(53) - 1,
        p(53),
        p(53) + 1,
        p(53) + 3,
        p(62),
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
        // Double-rounding probes: one rounding to 24 bits rounds these up.
        // Rounding to 53 bits first lands on a tie that rounds down to even.
        p(60) + p(36) + 1,
        p(40) + p(16) + 1,
        // Exact 24-bit ties: one rounds down to even, one rounds up to even.
        p(40) + p(16),
        p(40) + 3 * p(16),
        // The largest i64 below 2^63 that f32 holds exactly.
        ((1i64 << 24) - 1) << 39,
    ];
    let negatives: Vec<i64> = v.iter().filter(|&&x| x > 0).map(|&x| -x).collect();
    v.extend(negatives);
    v
}

/// Deterministic I64 values spread over every bit length and both signs.
#[cfg(feature = "wgpu")]
fn i64_sweep() -> Vec<i64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut values = Vec::new();
    for bits in 1..=63u32 {
        for _ in 0..64 {
            let mask = (1u64 << bits) - 1;
            let top = 1u64 << (bits - 1);
            let magnitude = ((next() & mask) | top) as i64;
            values.push(magnitude);
            values.push(-magnitude);
        }
    }
    values
}

#[cfg(feature = "wgpu")]
fn f32_edges() -> Vec<f32> {
    let p = |e: i32| 2f32.powi(e);
    let mut v = vec![
        0.0,
        0.5,
        1.0 - p(-24),
        1.0,
        1.5,
        2.5,
        p(22) + 0.5,
        p(23),
        p(24),
        p(31),
        p(32),
        p(53),
        p(62),
        p(63),
        // The largest f32 below 2^63.
        ((1u64 << 24) - 1) as f32 * p(39),
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::INFINITY,
    ];
    let negatives: Vec<f32> = v.iter().map(|&x| -x).collect();
    v.extend(negatives);
    v.push(f32::NAN);
    v.push(-f32::NAN);
    v
}

/// Cast `data` to `dst` on CPU and WebGPU and compare elementwise through
/// `key`, which maps an output element to its exact bit pattern.
#[cfg(feature = "wgpu")]
fn assert_cast_parity<S, D>(data: &[S], dst: DType, key: fn(D) -> u64, label: &str)
where
    S: Element + std::fmt::Debug,
    D: Element + std::fmt::Debug,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    let shape = [data.len()];
    let cpu_src = Tensor::<CpuRuntime>::from_slice(data, &shape, &cpu_device).expect("cpu tensor");
    let expected: Vec<D> = cpu_client.cast(&cpu_src, dst).expect("cpu cast").to_vec();

    with_wgpu_backend_or_skip(|client, device| {
        let src = Tensor::<WgpuRuntime>::from_slice(data, &shape, &device).expect("wgpu tensor");
        let out = client.cast(&src, dst).expect("wgpu cast");
        assert_eq!(out.dtype(), dst, "{label}: output dtype");
        assert_eq!(out.shape(), &shape, "{label}: output shape");
        let got: Vec<D> = out.to_vec();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                key(*g),
                key(*e),
                "{label} at index {i}: WebGPU {g:?} vs CPU {e:?} for source {:?}",
                data[i]
            );
        }
    });
}

#[cfg(feature = "wgpu")]
fn f32_bits(v: f32) -> u64 {
    v.to_bits() as u64
}

#[cfg(feature = "wgpu")]
fn i32_bits(v: i32) -> u64 {
    v as u32 as u64
}

#[cfg(feature = "wgpu")]
fn u32_bits(v: u32) -> u64 {
    v as u64
}

#[cfg(feature = "wgpu")]
fn i64_bits(v: i64) -> u64 {
    v as u64
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_to_f32_edges_wgpu() {
    assert_cast_parity::<i64, f32>(&i64_edges(), DType::F32, f32_bits, "i64->f32 edges");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_to_f32_sweep_wgpu() {
    assert_cast_parity::<i64, f32>(&i64_sweep(), DType::F32, f32_bits, "i64->f32 sweep");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_to_f32_double_rounding_wgpu() {
    // The CPU reference rounds through f64. Pin that the probe needs it.
    let probe = (1i64 << 60) + (1i64 << 36) + 1;
    assert_eq!(((probe as f64) as f32).to_bits(), 2f32.powi(60).to_bits());
    assert_ne!((probe as f32).to_bits(), ((probe as f64) as f32).to_bits());
    assert_cast_parity::<i64, f32>(&[probe, -probe], DType::F32, f32_bits, "i64->f32 probe");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_to_i32_edges_wgpu() {
    assert_cast_parity::<i64, i32>(&i64_edges(), DType::I32, i32_bits, "i64->i32 edges");
    assert_cast_parity::<i64, i32>(&i64_sweep(), DType::I32, i32_bits, "i64->i32 sweep");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_to_u32_edges_wgpu() {
    assert_cast_parity::<i64, u32>(&i64_edges(), DType::U32, u32_bits, "i64->u32 edges");
    assert_cast_parity::<i64, u32>(&i64_sweep(), DType::U32, u32_bits, "i64->u32 sweep");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i32_to_i64_edges_wgpu() {
    let data = [
        0,
        1,
        -1,
        7,
        -7,
        i32::MAX,
        i32::MIN,
        i32::MAX - 1,
        i32::MIN + 1,
    ];
    assert_cast_parity::<i32, i64>(&data, DType::I64, i64_bits, "i32->i64");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_u32_to_i64_edges_wgpu() {
    let data = [0, 1, 7, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX];
    assert_cast_parity::<u32, i64>(&data, DType::I64, i64_bits, "u32->i64");
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_f32_to_i64_edges_wgpu() {
    assert_cast_parity::<f32, i64>(&f32_edges(), DType::I64, i64_bits, "f32->i64 edges");
}

/// Every f32 sign and exponent, with a spread of mantissas.
#[cfg(feature = "wgpu")]
#[test]
fn test_cast_f32_to_i64_all_exponents_wgpu() {
    let mut data = Vec::new();
    for sign in 0..2u32 {
        for exp in 0..256u32 {
            for frac in [0u32, 1, 0x40_0000, 0x55_5555, 0x7f_ffff] {
                data.push(f32::from_bits((sign << 31) | (exp << 23) | frac));
            }
        }
    }
    assert_cast_parity::<f32, i64>(&data, DType::I64, i64_bits, "f32->i64 exponents");
}

/// An I64 round trip through every 32-bit dtype keeps the in-range values.
#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_roundtrip_wgpu() {
    let data: Vec<i64> = vec![0, 1, -1, 12345, -12345, 1 << 20, -(1 << 20)];
    with_wgpu_backend_or_skip(|client, device| {
        let src = Tensor::<WgpuRuntime>::from_slice(&data, &[data.len()], &device).expect("tensor");
        for mid in [DType::F32, DType::I32] {
            let back = client
                .cast(&client.cast(&src, mid).expect("cast out"), DType::I64)
                .expect("cast back");
            assert_eq!(back.to_vec::<i64>(), data, "i64 round trip via {mid:?}");
        }
    });
}

/// A transposed view goes through `ensure_contiguous` before the kernel.
#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_strided_view_wgpu() {
    let data: Vec<i64> = vec![1, -2, 3, i64::MAX, i64::MIN, 1 << 40];
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_src = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device)
        .expect("cpu tensor")
        .transpose(0, 1)
        .expect("cpu transpose");
    let expected = cpu_client
        .cast(&cpu_src, DType::F32)
        .expect("cpu cast")
        .to_vec::<f32>();

    with_wgpu_backend_or_skip(|client, device| {
        let src = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &device)
            .expect("wgpu tensor")
            .transpose(0, 1)
            .expect("wgpu transpose");
        let got = client.cast(&src, DType::F32).expect("wgpu cast");
        assert_eq!(got.shape(), &[3, 2]);
        let got_bits: Vec<u32> = got.to_vec::<f32>().iter().map(|v| v.to_bits()).collect();
        let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got_bits, expected_bits, "i64->f32 on a transposed view");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_cast_i64_empty_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let empty_i64 = Tensor::<WgpuRuntime>::from_slice::<i64>(&[], &[0], &device).expect("i64");
        let empty_f32 =
            Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 3], &device).expect("f32");
        let to_f32 = client
            .cast(&empty_i64, DType::F32)
            .expect("i64->f32 on empty");
        assert_eq!(
            (to_f32.shape(), to_f32.dtype()),
            (&[0usize][..], DType::F32)
        );
        let to_i64 = client
            .cast(&empty_f32, DType::I64)
            .expect("f32->i64 on empty");
        assert_eq!(
            (to_i64.shape(), to_i64.dtype()),
            (&[0usize, 3][..], DType::I64)
        );
    });
}

/// More elements than one dispatch row of 65535 workgroups holds, so the
/// kernel runs on the second grid row too.
#[cfg(feature = "wgpu")]
#[test]
fn test_cast_u32_to_i64_two_grid_rows_wgpu() {
    let n = 65535 * 256 + 300;
    let data: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    with_wgpu_backend_or_skip(|client, device| {
        let src = Tensor::<WgpuRuntime>::from_slice(&data, &[n], &device).expect("tensor");
        let got = client.cast(&src, DType::I64).expect("cast").to_vec::<i64>();
        assert_eq!(got.len(), n);
        for (i, (&g, &s)) in got.iter().zip(data.iter()).enumerate() {
            assert_eq!(g, s as i64, "u32->i64 at index {i}");
        }
    });
}
