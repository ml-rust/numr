//! Shape products that overflow `usize` must return an error, never wrap.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{DistanceMetric, DistanceOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn setup() -> (<CpuRuntime as Runtime>::Client, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

#[test]
fn empty_with_overflowing_shape_is_invalid_argument() {
    let (_, device) = setup();
    let result = Tensor::<CpuRuntime>::empty(&[usize::MAX, 2], DType::F32, &device);
    assert!(
        matches!(result, Err(Error::InvalidArgument { arg: "shape", .. })),
        "expected InvalidArgument for shape, got {:?}",
        result.map(|t| t.shape().to_vec())
    );
}

#[test]
fn empty_with_zero_extent_has_no_elements() {
    let (_, device) = setup();
    let t = Tensor::<CpuRuntime>::empty(&[usize::MAX, 0], DType::F32, &device)
        .expect("a shape with a zero extent is legal");
    assert_eq!(t.numel(), 0);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn empty_with_large_extents_that_overflow_is_error() {
    let (_, device) = setup();
    let result = Tensor::<CpuRuntime>::empty(&[1 << 32, 1 << 32], DType::F32, &device);
    assert!(matches!(result, Err(Error::InvalidArgument { .. })));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn pdist_with_overflowing_pair_count_is_error() {
    let (client, device) = setup();
    let n = (1usize << 32) + 1;
    let x = Tensor::<CpuRuntime>::empty(&[n, 0], DType::F32, &device)
        .expect("zero-element tensor is legal");
    assert!(client.pdist(&x, DistanceMetric::Euclidean).is_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn cdist_with_overflowing_output_is_error() {
    let (client, device) = setup();
    let x = Tensor::<CpuRuntime>::empty(&[(1usize << 32) + 1, 0], DType::F32, &device)
        .expect("zero-element tensor is legal");
    let y = Tensor::<CpuRuntime>::empty(&[1usize << 32, 0], DType::F32, &device)
        .expect("zero-element tensor is legal");
    assert!(client.cdist(&x, &y, DistanceMetric::Euclidean).is_err());
}
