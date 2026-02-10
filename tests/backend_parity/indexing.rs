// Backend parity tests migrated from tests/index_ops/masked.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;
use numr::error::Error;
use numr::ops::IndexingOps;
#[cfg(feature = "cuda")]
use numr::runtime::Runtime;
#[cfg(feature = "cuda")]
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

#[test]
fn test_masked_ops_parity() {
    #[cfg(feature = "cuda")]
    let cpu_device = CpuDevice::new();
    #[cfg(feature = "cuda")]
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    #[cfg(feature = "cuda")]
    let a_cpu =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu_device);
    #[cfg(feature = "cuda")]
    let mask_row_cpu = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &cpu_device);
    #[cfg(feature = "cuda")]
    let cpu_select_row: Vec<f32> = cpu_client
        .masked_select(&a_cpu, &mask_row_cpu)
        .unwrap()
        .to_vec();
    #[cfg(feature = "cuda")]
    let cpu_fill_row: Vec<f32> = cpu_client
        .masked_fill(&a_cpu, &mask_row_cpu, -1.0)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &cuda_device,
        );
        let mask_row = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0, 1],
            &[1, 3],
            &cuda_device,
        );
        let select_row: Vec<f32> = cuda_client.masked_select(&a, &mask_row).unwrap().to_vec();
        assert_eq!(cpu_select_row, select_row);
        let fill_row: Vec<f32> = cuda_client
            .masked_fill(&a, &mask_row, -1.0)
            .unwrap()
            .to_vec();
        assert_eq!(cpu_fill_row, fill_row);

        let mask_col = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[2, 1],
            &cuda_device,
        );
        let select_col: Vec<f32> = cuda_client.masked_select(&a, &mask_col).unwrap().to_vec();
        assert_eq!(select_col, vec![1.0, 2.0, 3.0]);
        let fill_col: Vec<f32> = cuda_client
            .masked_fill(&a, &mask_col, 99.0)
            .unwrap()
            .to_vec();
        assert_eq!(fill_col, vec![99.0, 99.0, 99.0, 4.0, 5.0, 6.0]);

        let a3 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            &cuda_device,
        );
        let m3 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[1, 2, 1],
            &cuda_device,
        );
        let d3: Vec<f32> = cuda_client.masked_select(&a3, &m3).unwrap().to_vec();
        assert_eq!(d3, vec![1.0, 2.0, 5.0, 6.0]);

        let a64 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0],
            &[2, 2],
            &cuda_device,
        );
        let m64 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[2, 1],
            &cuda_device,
        );
        let d64: Vec<f64> = cuda_client
            .masked_fill(&a64, &m64, -999.0)
            .unwrap()
            .to_vec();
        assert_eq!(d64, vec![-999.0, -999.0, 3.0, 4.0]);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &wgpu_device,
        );
        let mask = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1u32, 0, 1, 0, 0, 1, 0, 1],
            &[2, 4],
            &wgpu_device,
        );

        let selected: Vec<f32> = wgpu_client.masked_select(&a, &mask).unwrap().to_vec();
        assert_eq!(selected, vec![1.0, 3.0, 6.0, 8.0]);

        let filled: Vec<f32> = wgpu_client.masked_fill(&a, &mask, -1.0).unwrap().to_vec();
        assert_eq!(filled, vec![-1.0, 2.0, -1.0, 4.0, 5.0, -1.0, 7.0, -1.0]);
    });
}

#[test]
fn test_take_put_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &cpu_device,
    );
    let idx_cpu = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &cpu_device);
    let put_values_cpu = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &cpu_device);
    let cpu_take: Vec<f32> = cpu_client.take(&a_cpu, &idx_cpu).unwrap().to_vec();
    let cpu_put: Vec<f32> = cpu_client
        .put(&a_cpu, &idx_cpu, &put_values_cpu)
        .unwrap()
        .to_vec();
    assert_eq!(cpu_take, vec![60.0, 10.0, 30.0, 50.0]);
    assert_eq!(cpu_put, vec![2.0, 20.0, 3.0, 40.0, 4.0, 1.0]);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &cuda_device,
        );
        let idx = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[5i32, 0, 2, 4],
            &[2, 2],
            &cuda_device,
        );
        let put_values = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &cuda_device,
        );

        let take: Vec<f32> = cuda_client.take(&a, &idx).unwrap().to_vec();
        assert_eq!(cpu_take, take);

        let put: Vec<f32> = cuda_client.put(&a, &idx, &put_values).unwrap().to_vec();
        assert_eq!(cpu_put, put);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &wgpu_device,
        );
        let idx = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[5i32, 0, 2, 4],
            &[2, 2],
            &wgpu_device,
        );
        let put_values = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &wgpu_device,
        );

        let take: Vec<f32> = wgpu_client.take(&a, &idx).unwrap().to_vec();
        assert_eq!(take, vec![60.0, 10.0, 30.0, 50.0]);

        let put: Vec<f32> = wgpu_client.put(&a, &idx, &put_values).unwrap().to_vec();
        assert_eq!(put, vec![2.0, 20.0, 3.0, 40.0, 4.0, 1.0]);
    });
}

#[test]
fn test_take_put_i64_indices_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &cpu_device,
    );
    let idx_cpu = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &cpu_device);
    let put_values_cpu = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &cpu_device);
    let cpu_take: Vec<f32> = cpu_client.take(&a_cpu, &idx_cpu).unwrap().to_vec();
    let cpu_put: Vec<f32> = cpu_client
        .put(&a_cpu, &idx_cpu, &put_values_cpu)
        .unwrap()
        .to_vec();
    assert_eq!(cpu_take, vec![60.0, 10.0, 30.0, 50.0]);
    assert_eq!(cpu_put, vec![2.0, 20.0, 3.0, 40.0, 4.0, 1.0]);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &cuda_device,
        );
        let idx = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[5i64, 0, 2, 4],
            &[2, 2],
            &cuda_device,
        );
        let put_values = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &cuda_device,
        );

        let take: Vec<f32> = cuda_client.take(&a, &idx).unwrap().to_vec();
        assert_eq!(cpu_take, take);

        let put: Vec<f32> = cuda_client.put(&a, &idx, &put_values).unwrap().to_vec();
        assert_eq!(cpu_put, put);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &wgpu_device,
        );
        let idx = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[5i64, 0, 2, 4],
            &[2, 2],
            &wgpu_device,
        );
        let put_values = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &wgpu_device,
        );

        let take: Vec<f32> = wgpu_client.take(&a, &idx).unwrap().to_vec();
        assert_eq!(take, vec![60.0, 10.0, 30.0, 50.0]);

        let put: Vec<f32> = wgpu_client.put(&a, &idx, &put_values).unwrap().to_vec();
        assert_eq!(put, vec![2.0, 20.0, 3.0, 40.0, 4.0, 1.0]);
    });
}

#[test]
fn test_take_put_reject_non_integer_indices() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &cpu_device,
    );
    let idx_cpu = Tensor::from_slice(&[0.0f32, 2.0], &[2], &cpu_device);
    let put_values_cpu = Tensor::from_slice(&[1.0f32, 2.0], &[2], &cpu_device);

    let take_err = cpu_client.take(&a_cpu, &idx_cpu).unwrap_err();
    match take_err {
        Error::InvalidArgument { arg, reason } => {
            assert_eq!(arg, "indices");
            assert!(reason.contains("I32 or I64"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    let put_err = cpu_client
        .put(&a_cpu, &idx_cpu, &put_values_cpu)
        .unwrap_err();
    match put_err {
        Error::InvalidArgument { arg, reason } => {
            assert_eq!(arg, "indices");
            assert!(reason.contains("I32 or I64"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
