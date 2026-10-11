//! WebGPU strided copies and row gathers of sub-4-byte elements, against the
//! CPU runtime.
//!
//! A WGSL storage buffer addresses 4-byte words. U8, I8, Bool, F16 and BF16
//! elements, and quantized block bytes, need the byte kernels. Each case
//! builds the same view on both runtimes and requires identical bytes.
//!
//! Run: cargo test --features wgpu --test wgpu_byte_copies

#![cfg(feature = "wgpu")]

use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime, is_wgpu_available};
use numr::tensor::Tensor;

/// The WebGPU device, or `None` after one skip line.
fn wgpu_device() -> Option<WgpuDevice> {
    if !is_wgpu_available() {
        println!("skip: no WebGPU adapter");
        return None;
    }
    Some(WgpuDevice::new(0))
}

/// Distinct bytes with no period that divides small row lengths.
fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_right(13) as u8)
        .collect()
}

/// The same bytes as a tensor on both runtimes.
fn pair(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &WgpuDevice,
) -> (Tensor<CpuRuntime>, Tensor<WgpuRuntime>) {
    let cpu = Tensor::<CpuRuntime>::from_bytes(bytes, shape, dtype, &CpuDevice::new()).unwrap();
    let gpu = Tensor::<WgpuRuntime>::from_bytes(bytes, shape, dtype, device).unwrap();
    (cpu, gpu)
}

/// `view` applied on both runtimes yields identical bytes.
fn check<F, G>(label: &str, cpu: &Tensor<CpuRuntime>, gpu: &Tensor<WgpuRuntime>, f: F, g: G)
where
    F: Fn(&Tensor<CpuRuntime>) -> Tensor<CpuRuntime>,
    G: Fn(&Tensor<WgpuRuntime>) -> Tensor<WgpuRuntime>,
{
    let want = f(cpu).contiguous().unwrap().to_bytes().unwrap();
    let got = g(gpu).contiguous().unwrap().to_bytes().unwrap();
    assert_eq!(got.len(), want.len(), "{label}: byte length");
    if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
        panic!("{label}: first differing byte {i}");
    }
}

/// A Q8_0 weight's row regroup: `[groups, parts, per * row_bytes]` U8 bytes
/// permuted to `[parts, groups, ...]`. A 34-byte row leaves every cell
/// unaligned.
#[test]
fn u8_row_regroup_matches_cpu() {
    let Some(device) = wgpu_device() else { return };
    for (groups, parts, per, row_bytes) in [(3, 2, 1, 34), (4, 2, 3, 34), (2, 3, 2, 210)] {
        let len = groups * parts * per * row_bytes;
        let (cpu, gpu) = pair(
            &pattern(len),
            &[groups, parts, per * row_bytes],
            DType::U8,
            &device,
        );
        check(
            &format!("regroup {groups}x{parts}x{per}x{row_bytes}"),
            &cpu,
            &gpu,
            |t| t.permute(&[1, 0, 2]).unwrap(),
            |t| t.permute(&[1, 0, 2]).unwrap(),
        );
    }
}

/// U8 transposes, narrows at odd offsets, and flips (negative strides).
#[test]
fn u8_views_match_cpu() {
    let Some(device) = wgpu_device() else { return };
    let (cpu, gpu) = pair(&pattern(7 * 13), &[7, 13], DType::U8, &device);
    check(
        "transpose",
        &cpu,
        &gpu,
        |t| t.transpose(0, 1).unwrap(),
        |t| t.transpose(0, 1).unwrap(),
    );
    check(
        "narrow cols",
        &cpu,
        &gpu,
        |t| t.narrow(1, 3, 5).unwrap(),
        |t| t.narrow(1, 3, 5).unwrap(),
    );
    check(
        "narrow rows",
        &cpu,
        &gpu,
        |t| t.narrow(0, 1, 5).unwrap(),
        |t| t.narrow(0, 1, 5).unwrap(),
    );
    check(
        "flip",
        &cpu,
        &gpu,
        |t| t.flip(1).unwrap(),
        |t| t.flip(1).unwrap(),
    );
}

/// 2-byte elements: a transpose and a narrow that starts on a half word.
#[test]
fn two_byte_views_match_cpu() {
    let Some(device) = wgpu_device() else { return };
    for dtype in [DType::F16, DType::BF16, DType::I16] {
        let (cpu, gpu) = pair(&pattern(5 * 9 * 2), &[5, 9], dtype, &device);
        check(
            &format!("{dtype:?} transpose"),
            &cpu,
            &gpu,
            |t| t.transpose(0, 1).unwrap(),
            |t| t.transpose(0, 1).unwrap(),
        );
        check(
            &format!("{dtype:?} narrow"),
            &cpu,
            &gpu,
            |t| t.narrow(1, 1, 6).unwrap(),
            |t| t.narrow(1, 1, 6).unwrap(),
        );
    }
}

/// A rank-0 view at a nonzero offset copies its one element.
#[test]
fn scalar_view_copies() {
    let Some(device) = wgpu_device() else { return };
    let (cpu, gpu) = pair(&pattern(4 * 6), &[6], DType::F32, &device);
    check(
        "scalar",
        &cpu,
        &gpu,
        |t| t.narrow(0, 3, 1).unwrap().reshape(&[]).unwrap(),
        |t| t.narrow(0, 3, 1).unwrap().reshape(&[]).unwrap(),
    );
}

/// A word-kernel copy past 65535 workgroups of 256 threads runs whole.
#[test]
fn large_f32_transpose_matches_cpu() {
    let Some(device) = wgpu_device() else { return };
    let (rows, cols) = (4099, 4097);
    assert!(rows * cols > 65535 * 256);
    let values: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let gpu = Tensor::<WgpuRuntime>::from_slice(&values, &[rows, cols], &device).unwrap();
    let out: Vec<f32> = gpu.transpose(0, 1).unwrap().contiguous().unwrap().to_vec();
    for &(r, c) in &[
        (0, 0),
        (1, 0),
        (rows - 1, cols - 1),
        (rows - 1, 0),
        (17, cols - 3),
    ] {
        assert_eq!(
            out[c * rows + r],
            (r * cols + c) as f32,
            "element ({r}, {c})"
        );
    }
    let tail = out.len() - 1;
    assert_eq!(out[tail], ((rows - 1) * cols + cols - 1) as f32);
}

/// Row gathers of U8 (quantized block rows) and I16 match the CPU.
#[test]
fn byte_index_select_matches_cpu() {
    let Some(device) = wgpu_device() else { return };
    let cpu_client = CpuRuntime::default_client(&CpuDevice::new());
    let gpu_client = WgpuRuntime::default_client(&device);
    let ids = [4i32, 0, 4, 2, 5];
    let cpu_ids = Tensor::<CpuRuntime>::from_slice(&ids, &[ids.len()], &CpuDevice::new()).unwrap();
    let gpu_ids = Tensor::<WgpuRuntime>::from_slice(&ids, &[ids.len()], &device).unwrap();
    for (dtype, row_elems) in [(DType::U8, 34), (DType::U8, 210), (DType::I16, 9)] {
        let row_bytes = row_elems * dtype.size_in_bytes();
        let (cpu, gpu) = pair(&pattern(6 * row_bytes), &[6, row_elems], dtype, &device);
        let want = cpu_client
            .index_select(&cpu, 0, &cpu_ids)
            .unwrap()
            .to_bytes()
            .unwrap();
        let got = gpu_client
            .index_select(&gpu, 0, &gpu_ids)
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_eq!(got, want, "{dtype:?} rows of {row_elems}");
    }
    // A gather along the inner axis, so every output byte has its own source.
    let (cpu, gpu) = pair(&pattern(3 * 7), &[3, 7], DType::U8, &device);
    let want = cpu_client
        .index_select(&cpu, 1, &cpu_ids)
        .unwrap()
        .to_bytes()
        .unwrap();
    let got = gpu_client
        .index_select(&gpu, 1, &gpu_ids)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_eq!(got, want, "U8 inner-axis gather");
}
