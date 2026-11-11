#![cfg(feature = "cuda")]

use numr::dtype::{Complex64, Complex128};
use numr::error::Error;
use numr::ops::IndexingOps;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::{Layout, Storage, Tensor};

fn cuda() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).expect("actual CUDA device required");
    (client, device)
}

#[test]
fn actual_cuda_index_put_rejects_invalid_operands() {
    let (client, device) = cuda();
    let a = Tensor::<CudaRuntime>::from_slice(&[1f32, 2., 3., 4.], &[4], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], &device).unwrap();
    let src = Tensor::<CudaRuntime>::from_slice(&[8f32], &[1], &device).unwrap();
    let wrong_index = Tensor::<CudaRuntime>::from_slice(&[1f32], &[1], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &wrong_index, &src),
        Err(Error::DTypeMismatch { .. })
    ));
    let rank_two = index.reshape(&[1, 1]).unwrap();
    assert!(client.index_put(&a, 0, &rank_two, &src).is_err());
    assert!(matches!(
        client.index_put(&a, 1, &index, &src),
        Err(Error::InvalidDimension { .. })
    ));
    match client.index_put(&a, usize::MAX, &index, &src) {
        Err(Error::InvalidArgument { arg: "dim", reason }) => {
            assert!(reason.contains(&usize::MAX.to_string()));
        }
        other => panic!("Oversized dimension lost its original value: {other:?}"),
    }
    let wrong_shape = Tensor::<CudaRuntime>::from_slice(&[8f32, 9.], &[2], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &wrong_shape),
        Err(Error::ShapeMismatch { .. })
    ));
    let wrong_dtype = Tensor::<CudaRuntime>::from_slice(&[8f64], &[1], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &wrong_dtype),
        Err(Error::DTypeMismatch { .. })
    ));
    for value in [-1i64, 4] {
        let invalid = Tensor::<CudaRuntime>::from_slice(&[value], &[1], &device).unwrap();
        assert!(matches!(
            client.index_put(&a, 0, &invalid, &src),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }
    assert_eq!(a.to_vec::<f32>(), [1., 2., 3., 4.]);
}

#[test]
fn actual_cuda_index_put_rejects_every_foreign_empty_operand() {
    let (client, device) = cuda();
    let a = Tensor::<CudaRuntime>::from_slice(&[] as &[f32], &[0], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_slice(&[] as &[i32], &[0], &device).unwrap();
    let foreign_device = CudaDevice::new(1);
    // Empty non-owning storage exercises device admission without a second GPU.
    let foreign_float = Tensor::from_parts(
        unsafe { Storage::<CudaRuntime>::from_ptr(0, 0, numr::dtype::DType::F32, &foreign_device) },
        Layout::contiguous(&[0]),
    );
    let foreign_index = Tensor::from_parts(
        unsafe { Storage::<CudaRuntime>::from_ptr(0, 0, numr::dtype::DType::I32, &foreign_device) },
        Layout::contiguous(&[0]),
    );
    for result in [
        client.index_put(&foreign_float, 0, &index, &a),
        client.index_put(&a, 0, &foreign_index, &a),
        client.index_put(&a, 0, &index, &foreign_float),
    ] {
        assert!(matches!(result, Err(Error::DeviceMismatch)));
    }
}

#[test]
fn actual_cuda_index_put_rejects_complex_even_when_empty() {
    let (client, device) = cuda();
    let index = Tensor::<CudaRuntime>::from_slice(&[] as &[i32], &[0], &device).unwrap();
    let a = Tensor::<CudaRuntime>::from_slice(&[] as &[Complex64], &[0], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &a),
        Err(Error::UnsupportedDType { .. })
    ));
    let a = Tensor::<CudaRuntime>::from_slice(&[] as &[Complex128], &[0], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &a),
        Err(Error::UnsupportedDType { .. })
    ));
}

#[test]
fn actual_cuda_index_put_checks_empty_operand_metadata() {
    let (client, device) = cuda();
    let a = Tensor::<CudaRuntime>::from_slice(&[1f32, 2., 3., 4.], &[4], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_slice(&[] as &[i32], &[0], &device).unwrap();
    let src = Tensor::<CudaRuntime>::from_slice(&[] as &[f32], &[0], &device).unwrap();
    let wrong_index = Tensor::<CudaRuntime>::from_slice(&[] as &[f32], &[0], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &wrong_index, &src),
        Err(Error::DTypeMismatch { .. })
    ));
    let rank_two = index.reshape(&[0, 1]).unwrap();
    assert!(client.index_put(&a, 0, &rank_two, &src).is_err());
    let wrong_dtype = Tensor::<CudaRuntime>::from_slice(&[] as &[f64], &[0], &device).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &wrong_dtype),
        Err(Error::DTypeMismatch { .. })
    ));
    let wrong_shape = src.reshape(&[0, 1]).unwrap();
    assert!(matches!(
        client.index_put(&a, 0, &index, &wrong_shape),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.index_put(&a, 1, &index, &src),
        Err(Error::InvalidDimension { .. })
    ));
    assert_eq!(a.to_vec::<f32>(), [1., 2., 3., 4.]);
}

fn virtual_view(base: &Tensor<CudaRuntime>, shape: Vec<usize>) -> Tensor<CudaRuntime> {
    let strides = vec![0isize; shape.len()];
    Tensor::from_parts(base.storage().clone(), Layout::new(shape, strides, 0))
}

#[test]
fn actual_cuda_index_put_checks_virtual_counts_before_allocation() {
    let (client, device) = cuda();
    let scalar = Tensor::<CudaRuntime>::from_slice(&[1f64], &[1], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_slice(&[0i32], &[1], &device).unwrap();
    let empty_index = Tensor::<CudaRuntime>::from_slice(&[] as &[i32], &[0], &device).unwrap();
    let huge = u32::MAX as usize + 1;
    let a = virtual_view(&scalar, vec![huge]);
    assert!(client.index_put(&a, 0, &index, &scalar).is_err());
    let a = virtual_view(&scalar, vec![0, huge]);
    let src = virtual_view(&scalar, vec![0, 0]);
    assert!(client.index_put(&a, 1, &empty_index, &src).is_err());
    let a = virtual_view(&scalar, vec![huge, 0]);
    let src = virtual_view(&scalar, vec![huge, 0]);
    assert!(client.index_put(&a, 1, &empty_index, &src).is_err());
    let a = virtual_view(&scalar, vec![0, 1, huge]);
    let src = virtual_view(&scalar, vec![0, 0, huge]);
    assert!(client.index_put(&a, 1, &empty_index, &src).is_err());
    let huge_index = Tensor::from_parts(
        index.storage().clone(),
        Layout::new(vec![huge], vec![0isize], 0),
    );
    let a = virtual_view(&scalar, vec![0, 3]);
    let src = virtual_view(&scalar, vec![0, huge]);
    assert!(client.index_put(&a, 1, &huge_index, &src).is_err());
    let a = virtual_view(&scalar, vec![usize::MAX / 8 + 1]);
    assert!(client.index_put(&a, 0, &index, &scalar).is_err());
    let a = virtual_view(&scalar, vec![usize::MAX, 2]);
    let src = virtual_view(&scalar, vec![1, 2]);
    assert!(client.index_put(&a, 0, &index, &src).is_err());
}

#[test]
fn actual_cuda_index_put_rejects_unsupported_strided_rank() {
    let (client, device) = cuda();
    let base = Tensor::<CudaRuntime>::from_slice(&[1f32], &[1], &device).unwrap();
    let a = virtual_view(&base, vec![2; 9]);
    let src = virtual_view(&base, [vec![1], vec![2; 8]].concat());
    let index = Tensor::<CudaRuntime>::from_slice(&[0i32], &[1], &device).unwrap();
    assert!(client.index_put(&a, 0, &index, &src).is_err());
}

#[test]
fn actual_cuda_index_put_rejects_malformed_view_offsets_and_ranges() {
    let (client, device) = cuda();
    let a = Tensor::<CudaRuntime>::from_slice(&[1f32, 2., 3., 4.], &[4], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_slice(&[1i32, 3], &[2], &device).unwrap();
    let src = Tensor::<CudaRuntime>::from_slice(&[20f32, 40.], &[2], &device).unwrap();
    for (stride, offset) in [(-1isize, 0), (2, 0), (1, 4), (1, usize::MAX)] {
        let malformed = |tensor: &Tensor<CudaRuntime>| {
            Tensor::from_parts(
                tensor.storage().clone(),
                Layout::new(tensor.shape().to_vec(), vec![stride], offset),
            )
        };
        for result in [
            client.index_put(&malformed(&a), 0, &index, &src),
            client.index_put(&a, 0, &malformed(&index), &src),
            client.index_put(&a, 0, &index, &malformed(&src)),
        ] {
            assert!(matches!(
                result,
                Err(Error::InvalidArgument { .. } | Error::BackendLimitation { .. })
            ));
        }
    }
    assert_eq!(a.to_vec::<f32>(), [1., 2., 3., 4.]);
    assert_eq!(index.to_vec::<i32>(), [1, 3]);
    assert_eq!(src.to_vec::<f32>(), [20., 40.]);
}
