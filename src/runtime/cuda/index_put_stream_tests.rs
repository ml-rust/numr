use super::{CudaClient, CudaDevice, CudaRuntime};
use crate::ops::{BinaryOps, IndexingOps};
use crate::tensor::{Layout, Tensor};

#[test]
fn actual_cuda_index_put_selected_stream_capture_replays_updated_inputs() {
    let device = CudaDevice::new(0);
    let client = CudaClient::new_uncached(device.clone()).expect("actual CUDA device required");
    let a = Tensor::<CudaRuntime>::from_slice(&[1f32, 2., 3., 4.], &[4], &device).unwrap();
    let src = Tensor::<CudaRuntime>::from_slice(&[20f32, 40.], &[2], &device).unwrap();
    let index_backing = Tensor::<CudaRuntime>::from_slice(&[1i32, 9, 3, 9], &[4], &device).unwrap();
    let index = Tensor::<CudaRuntime>::from_parts(
        index_backing.storage().clone(),
        Layout::new(vec![2], vec![2isize], 0),
    );
    let ones = Tensor::<CudaRuntime>::from_slice(&[1f32; 4], &[4], &device).unwrap();
    let output = Tensor::<CudaRuntime>::from_slice(&[0f32; 4], &[4], &device).unwrap();
    device.sync().unwrap();

    // Warm every kernel before capture, including strided I32 normalization.
    let run = |cc: &CudaClient| -> crate::error::Result<()> {
        let result = cc.index_put(&a, 0, &index, &src)?;
        cc.add_into(&output, &result, &ones)
    };
    run(&client).unwrap();
    client.stream().synchronize().unwrap();
    assert_eq!(output.to_vec::<f32>(), [2., 21., 4., 41.]);
    assert_eq!(a.to_vec::<f32>(), [1., 2., 3., 4.]);

    let captured = CudaRuntime::capture_graph_into_with_arena(
        &client,
        &[&a, &index_backing, &src, &ones],
        &[&output],
        1 << 20,
        run,
    )
    .unwrap();
    captured.launch().unwrap();
    client.stream().synchronize().unwrap();
    assert_eq!(output.to_vec::<f32>(), [2., 21., 4., 41.]);
    assert_eq!(a.to_vec::<f32>(), [1., 2., 3., 4.]);

    let next_a = Tensor::<CudaRuntime>::from_slice(&[5f32, 6., 7., 8.], &[4], &device).unwrap();
    let next_src = Tensor::<CudaRuntime>::from_slice(&[50f32, 70.], &[2], &device).unwrap();
    let next_index = Tensor::<CudaRuntime>::from_slice(&[0i32, 9, 2, 9], &[4], &device).unwrap();
    device.sync().unwrap();
    client.copy_into(&a, &next_a).unwrap();
    client.copy_into(&src, &next_src).unwrap();
    client.copy_into(&index_backing, &next_index).unwrap();
    captured.launch().unwrap();
    client.stream().synchronize().unwrap();
    assert_eq!(output.to_vec::<f32>(), [51., 7., 71., 9.]);
    assert_eq!(a.to_vec::<f32>(), [5., 6., 7., 8.]);
    assert_eq!(src.to_vec::<f32>(), [50., 70.]);
    assert_eq!(index_backing.to_vec::<i32>(), [0, 9, 2, 9]);

    for invalid in [[-1i32, 9, -1, 9], [4, 9, 4, 9], [-1, 9, 4, 9]] {
        let next_index = Tensor::<CudaRuntime>::from_slice(&invalid, &[4], &device).unwrap();
        device.sync().unwrap();
        client.copy_into(&index_backing, &next_index).unwrap();
        captured.launch().unwrap();
        client.stream().synchronize().unwrap();
        assert_eq!(output.to_vec::<f32>(), [6., 7., 8., 9.]);
        assert_eq!(a.to_vec::<f32>(), [5., 6., 7., 8.]);
        assert_eq!(src.to_vec::<f32>(), [50., 70.]);
        assert_eq!(index_backing.to_vec::<i32>(), invalid);
        assert!(matches!(
            client.index_put(&a, 0, &index, &src),
            Err(crate::error::Error::IndexOutOfBounds { .. })
        ));
        assert_eq!(a.to_vec::<f32>(), [5., 6., 7., 8.]);
    }
}
