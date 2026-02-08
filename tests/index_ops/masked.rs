//! Masked select and masked fill tests (including CUDA and WebGPU feature-gated tests)

use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Masked Select Tests
// ============================================================================

#[test]
fn test_masked_select_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0], &[4], &device);

    let result = client.masked_select(&a, &mask).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0]);
}

#[test]
fn test_masked_select_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0, 1, 0], &[2, 3], &device);

    let result = client.masked_select(&a, &mask).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 5.0]);
}

#[test]
fn test_masked_select_broadcast_row() {
    // Broadcast mask from [1, 3] to [2, 3]
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &device);

    let result = client.masked_select(&a, &mask).unwrap();

    // Mask [1, 0, 1] broadcasts to both rows, selects columns 0 and 2
    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_masked_select_broadcast_col() {
    // Broadcast mask from [2, 1] to [2, 3]
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0], &[2, 1], &device);

    let result = client.masked_select(&a, &mask).unwrap();

    // Mask [1; 0] broadcasts, selects entire first row
    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0]);
}

#[test]
fn test_masked_select_all_false() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 0, 0], &[4], &device);

    let result = client.masked_select(&a, &mask).unwrap();

    assert_eq!(result.shape(), &[0]);
    let data: Vec<f32> = result.to_vec();
    assert!(data.is_empty());
}

// ============================================================================
// Masked Fill Tests
// ============================================================================

#[test]
fn test_masked_fill_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0], &[4], &device);

    let result = client.masked_fill(&a, &mask, -1.0).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [-1.0, 2.0, -1.0, 4.0]);
}

#[test]
fn test_masked_fill_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0, 1, 0], &[2, 3], &device);

    let result = client.masked_fill(&a, &mask, 0.0).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
}

#[test]
fn test_masked_fill_broadcast_row() {
    // Broadcast mask from [1, 3] to [2, 3]
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &device);

    let result = client.masked_fill(&a, &mask, -1.0).unwrap();

    // Mask [1, 0, 1] broadcasts to both rows
    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [-1.0, 2.0, -1.0, -1.0, 5.0, -1.0]);
}

#[test]
fn test_masked_fill_broadcast_col() {
    // Broadcast mask from [2, 1] to [2, 3]
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 0], &[2, 1], &device);

    let result = client.masked_fill(&a, &mask, 99.0).unwrap();

    // Mask [1; 0] broadcasts, fills entire first row
    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [99.0, 99.0, 99.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_masked_fill_all_true() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 1, 1, 1], &[4], &device);

    let result = client.masked_fill(&a, &mask, 0.0).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [0.0, 0.0, 0.0, 0.0]);
}

// ============================================================================
// CUDA Masked Operations Tests (Feature-gated)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_masked_tests {
    use numr::ops::IndexingOps;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn test_cuda_masked_select_broadcast_row() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &device);

        let result = client.masked_select(&a, &mask).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cuda_masked_select_broadcast_col() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0], &[2, 1], &device);

        let result = client.masked_select(&a, &mask).unwrap();

        assert_eq!(result.shape(), &[3]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cuda_masked_fill_broadcast_row() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &device);

        let result = client.masked_fill(&a, &mask, -1.0).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [-1.0, 2.0, -1.0, -1.0, 5.0, -1.0]);
    }

    #[test]
    fn test_cuda_masked_fill_broadcast_col() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0], &[2, 1], &device);

        let result = client.masked_fill(&a, &mask, 99.0).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [99.0, 99.0, 99.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cuda_masked_select_3d_broadcast() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 3D tensor [2, 2, 2]
        let a = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            &device,
        );
        // Broadcast mask from [1, 2, 1] to [2, 2, 2]
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0], &[1, 2, 1], &device);

        let result = client.masked_select(&a, &mask).unwrap();

        // Mask broadcasts: selects first row of each 2x2 slice
        assert_eq!(result.shape(), &[4]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cuda_masked_fill_f64() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: [[1.0, 2.0], [3.0, 4.0]]
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device);
        // Mask: [[1], [0]] broadcasts to [[1, 1], [0, 0]]
        let mask = Tensor::<CudaRuntime>::from_slice(&[1u8, 0], &[2, 1], &device);

        let result = client.masked_fill(&a, &mask, -999.0).unwrap();

        // First row (mask=1) gets filled, second row (mask=0) unchanged
        let data: Vec<f64> = result.to_vec();
        assert_eq!(data, [-999.0, -999.0, 3.0, 4.0]);
    }
}

// ============================================================================
// WebGPU Masked Operations Tests (Feature-gated)
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_masked_tests {
    use numr::ops::IndexingOps;
    use numr::runtime::Runtime;
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn test_wgpu_masked_select_basic() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        // WebGPU uses U32 for mask (not U8 like CPU/CUDA)
        // WebGPU doesn't support broadcast for masked ops - shapes must match exactly
        let a = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        );
        let mask =
            Tensor::<WgpuRuntime>::from_slice(&[1u32, 0, 1, 0, 0, 1, 0, 1], &[2, 4], &device);

        let result = client.masked_select(&a, &mask).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [1.0, 3.0, 6.0, 8.0]);
    }

    #[test]
    fn test_wgpu_masked_fill_basic() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        // WebGPU uses U32 for mask (not U8 like CPU/CUDA)
        // WebGPU doesn't support broadcast for masked ops - shapes must match exactly
        let a = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        );
        let mask =
            Tensor::<WgpuRuntime>::from_slice(&[1u32, 0, 1, 0, 0, 1, 0, 1], &[2, 4], &device);

        let result = client.masked_fill(&a, &mask, -1.0).unwrap();

        assert_eq!(result.shape(), &[2, 4]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, [-1.0, 2.0, -1.0, 4.0, 5.0, -1.0, 7.0, -1.0]);
    }
}
