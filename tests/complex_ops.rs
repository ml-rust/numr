//! Integration tests for Complex Number Operations
//!
//! Tests high-level complex operations: conj, real, imag, angle

use numr::dtype::{Complex64, Complex128, DType};
use numr::ops::ComplexOps;
#[cfg(feature = "cuda")]
use numr::ops::{BinaryOps, UnaryOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Complex64 Operations
// ============================================================================

#[test]
fn test_complex64_conj() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test data: [1+2i, 3-4i, -5+6i, 7+0i]
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(-5.0, 6.0),
        Complex64::new(7.0, 0.0),
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device);
    let result = client.conj(&a).unwrap();

    let output: Vec<Complex64> = result.to_vec();

    // Verify conjugates: [1-2i, 3+4i, -5-6i, 7-0i]
    assert_eq!(output[0], Complex64::new(1.0, -2.0));
    assert_eq!(output[1], Complex64::new(3.0, 4.0));
    assert_eq!(output[2], Complex64::new(-5.0, -6.0));
    assert_eq!(output[3], Complex64::new(7.0, 0.0));
}

#[test]
fn test_complex64_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(-5.0, 6.0),
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[3], &device);
    let result = client.real(&a).unwrap();

    // Output should be F32
    assert_eq!(result.dtype(), DType::F32);

    let output: Vec<f32> = result.to_vec();
    assert_eq!(output, vec![1.0, 3.0, -5.0]);
}

#[test]
fn test_complex64_imag() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(-5.0, 6.0),
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[3], &device);
    let result = client.imag(&a).unwrap();

    // Output should be F32
    assert_eq!(result.dtype(), DType::F32);

    let output: Vec<f32> = result.to_vec();
    assert_eq!(output, vec![2.0, -4.0, 6.0]);
}

#[test]
fn test_complex64_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test data with known angles
    let data = vec![
        Complex64::new(1.0, 0.0),  // angle = 0
        Complex64::new(0.0, 1.0),  // angle = π/2
        Complex64::new(-1.0, 0.0), // angle = π
        Complex64::new(1.0, 1.0),  // angle = π/4
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device);
    let result = client.angle(&a).unwrap();

    // Output should be F32
    assert_eq!(result.dtype(), DType::F32);

    let output: Vec<f32> = result.to_vec();

    // Verify angles (with tolerance for floating point)
    assert!((output[0] - 0.0).abs() < 1e-6);
    assert!((output[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((output[2] - std::f32::consts::PI).abs() < 1e-6);
    assert!((output[3] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

// ============================================================================
// Complex128 Operations
// ============================================================================

#[test]
fn test_complex128_conj() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![Complex128::new(1.0, 2.0), Complex128::new(3.0, -4.0)];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2], &device);
    let result = client.conj(&a).unwrap();

    let output: Vec<Complex128> = result.to_vec();

    assert_eq!(output[0], Complex128::new(1.0, -2.0));
    assert_eq!(output[1], Complex128::new(3.0, 4.0));
}

#[test]
fn test_complex128_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![Complex128::new(1.5, 2.5), Complex128::new(-3.5, 4.5)];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2], &device);
    let result = client.real(&a).unwrap();

    // Output should be F64
    assert_eq!(result.dtype(), DType::F64);

    let output: Vec<f64> = result.to_vec();
    assert_eq!(output, vec![1.5, -3.5]);
}

#[test]
fn test_complex128_imag() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![Complex128::new(1.5, 2.5), Complex128::new(-3.5, 4.5)];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2], &device);
    let result = client.imag(&a).unwrap();

    // Output should be F64
    assert_eq!(result.dtype(), DType::F64);

    let output: Vec<f64> = result.to_vec();
    assert_eq!(output, vec![2.5, 4.5]);
}

#[test]
fn test_complex128_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![
        Complex128::new(1.0, 0.0), // angle = 0
        Complex128::new(0.0, 1.0), // angle = π/2
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2], &device);
    let result = client.angle(&a).unwrap();

    // Output should be F64
    assert_eq!(result.dtype(), DType::F64);

    let output: Vec<f64> = result.to_vec();

    assert!((output[0] - 0.0).abs() < 1e-10);
    assert!((output[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

// ============================================================================
// Real Type Behavior
// ============================================================================

#[test]
fn test_real_type_conj() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // For real types, conj should return the same tensor
    let data = vec![1.0f32, 2.0, -3.0, 4.0];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device);

    let result = client.conj(&a).unwrap();
    let output: Vec<f32> = result.to_vec();

    assert_eq!(output, data);
}

#[test]
fn test_real_type_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // For real types, real should return a copy
    let data = vec![1.0f32, 2.0, 3.0];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[3], &device);

    let result = client.real(&a).unwrap();
    let output: Vec<f32> = result.to_vec();

    assert_eq!(output, data);
}

#[test]
fn test_real_type_imag() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // For real types, imag should return zeros
    let data = vec![1.0f32, 2.0, 3.0];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[3], &device);

    let result = client.imag(&a).unwrap();
    let output: Vec<f32> = result.to_vec();

    assert_eq!(output, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_real_type_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // For real types: angle(x) = 0 if x >= 0, π if x < 0
    let data = vec![1.0f32, -2.0, 3.0, -5.0, 0.0];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[5], &device);

    let result = client.angle(&a).unwrap();
    let output: Vec<f32> = result.to_vec();

    // Expected: [0, π, 0, π, 0]
    assert_eq!(output[0], 0.0); // angle(1.0) = 0
    assert!((output[1] - std::f32::consts::PI).abs() < 1e-6); // angle(-2.0) = π
    assert_eq!(output[2], 0.0); // angle(3.0) = 0
    assert!((output[3] - std::f32::consts::PI).abs() < 1e-6); // angle(-5.0) = π
    assert_eq!(output[4], 0.0); // angle(0.0) = 0
}

// ============================================================================
// Multi-dimensional Tensors
// ============================================================================

#[test]
fn test_complex_2d_operations() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x3 matrix of complex numbers
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
        Complex64::new(9.0, 10.0),
        Complex64::new(11.0, 12.0),
    ];

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

    // Test conj
    let conj_result = client.conj(&a).unwrap();
    assert_eq!(conj_result.shape(), &[2, 3]);
    assert_eq!(conj_result.dtype(), DType::Complex64);

    // Test real extraction
    let real_result = client.real(&a).unwrap();
    assert_eq!(real_result.shape(), &[2, 3]);
    assert_eq!(real_result.dtype(), DType::F32);

    let real_data: Vec<f32> = real_result.to_vec();
    assert_eq!(real_data, vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);

    // Test imag extraction
    let imag_result = client.imag(&a).unwrap();
    assert_eq!(imag_result.shape(), &[2, 3]);
    assert_eq!(imag_result.dtype(), DType::F32);

    let imag_data: Vec<f32> = imag_result.to_vec();
    assert_eq!(imag_data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

// ============================================================================
// Real F64 Angle Tests
// ============================================================================

#[test]
fn test_real_f64_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // For F64 real types: angle(x) = 0 if x >= 0, π if x < 0
    let data = vec![1.0f64, -2.0, 3.0, -5.0, 0.0];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[5], &device);

    let result = client.angle(&a).unwrap();
    assert_eq!(result.dtype(), DType::F64);

    let output: Vec<f64> = result.to_vec();

    // Expected: [0, π, 0, π, 0]
    assert_eq!(output[0], 0.0); // angle(1.0) = 0
    assert!((output[1] - std::f64::consts::PI).abs() < 1e-10); // angle(-2.0) = π
    assert_eq!(output[2], 0.0); // angle(3.0) = 0
    assert!((output[3] - std::f64::consts::PI).abs() < 1e-10); // angle(-5.0) = π
    assert_eq!(output[4], 0.0); // angle(0.0) = 0
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_tensor_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Empty tensor
    let data: Vec<Complex64> = vec![];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[0], &device);

    let result = client.angle(&a).unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
    assert_eq!(result.dtype(), DType::F32);

    // Note: Calling to_vec() on empty tensors triggers alignment issues in bytemuck,
    // so we only verify the shape and dtype
}

#[test]
fn test_single_element_complex() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Single element
    let data = vec![Complex64::new(1.0, 1.0)];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[1], &device);

    let result = client.angle(&a).unwrap();
    let output: Vec<f32> = result.to_vec();

    // angle(1+i) = π/4
    assert!((output[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

#[test]
fn test_large_tensor_angle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Large tensor (10000 elements)
    let numel = 10000;
    let data: Vec<Complex64> = (0..numel)
        .map(|i| Complex64::new((i as f32).cos(), (i as f32).sin()))
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&data, &[numel], &device);
    let result = client.angle(&a).unwrap();

    assert_eq!(result.shape(), &[numel]);
    assert_eq!(result.numel(), numel);
    let output: Vec<f32> = result.to_vec();
    assert_eq!(output.len(), numel);
}

// ============================================================================
// Backend Parity Tests
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_conj() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(5.0, -6.0),
    ];

    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device);
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[3], &cuda_device);

    let cpu_result = cpu_client.conj(&cpu_tensor).unwrap();
    let cuda_result = cuda_client.conj(&cuda_tensor).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!((cpu_val.re - cuda_val.re).abs() < 1e-6);
        assert!((cpu_val.im - cuda_val.im).abs() < 1e-6);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_angle() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(0.0, -1.0),
    ];

    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device);
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[3], &cuda_device);

    let cpu_result = cpu_client.angle(&cpu_tensor).unwrap();
    let cuda_result = cuda_client.angle(&cuda_tensor).unwrap();

    let cpu_output: Vec<f32> = cpu_result.to_vec();
    let cuda_output: Vec<f32> = cuda_result.to_vec();

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val - cuda_val).abs() < 1e-6,
            "CPU: {}, CUDA: {}",
            cpu_val,
            cuda_val
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_angle_real() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    // Test angle() for real inputs (critical test for the fix)
    let data = vec![1.0f32, -2.0, 3.0, -5.0, 0.0];

    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[5], &cuda_device);

    let cpu_result = cpu_client.angle(&cpu_tensor).unwrap();
    let cuda_result = cuda_client.angle(&cuda_tensor).unwrap();

    let cpu_output: Vec<f32> = cpu_result.to_vec();
    let cuda_output: Vec<f32> = cuda_result.to_vec();

    for (i, (cpu_val, cuda_val)) in cpu_output.iter().zip(cuda_output.iter()).enumerate() {
        assert!(
            (cpu_val - cuda_val).abs() < 1e-6,
            "Mismatch at index {}: CPU = {}, CUDA = {}",
            i,
            cpu_val,
            cuda_val
        );
    }

    // Verify correct values
    assert_eq!(cpu_output[0], 0.0); // angle(1.0) = 0
    assert!((cpu_output[1] - std::f32::consts::PI).abs() < 1e-6); // angle(-2.0) = π
    assert_eq!(cpu_output[2], 0.0); // angle(3.0) = 0
    assert!((cpu_output[3] - std::f32::consts::PI).abs() < 1e-6); // angle(-5.0) = π
    assert_eq!(cpu_output[4], 0.0); // angle(0.0) = 0
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_cpu_parity_angle() {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};

    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let wgpu_device = WgpuDevice::new(0);
    let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

    let data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(0.0, -1.0),
    ];

    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[3], &cpu_device);
    let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[3], &wgpu_device);

    let cpu_result = cpu_client.angle(&cpu_tensor).unwrap();
    let wgpu_result = wgpu_client.angle(&wgpu_tensor).unwrap();

    let cpu_output: Vec<f32> = cpu_result.to_vec();
    let wgpu_output: Vec<f32> = wgpu_result.to_vec();

    for (cpu_val, wgpu_val) in cpu_output.iter().zip(wgpu_output.iter()) {
        assert!(
            (cpu_val - wgpu_val).abs() < 1e-6,
            "CPU: {}, WebGPU: {}",
            cpu_val,
            wgpu_val
        );
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_cpu_parity_angle_real() {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};

    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let wgpu_device = WgpuDevice::new(0);
    let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

    // Test angle() for real F32 inputs (critical test for the fix)
    let data = vec![1.0f32, -2.0, 3.0, -5.0, 0.0];

    let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
    let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[5], &wgpu_device);

    let cpu_result = cpu_client.angle(&cpu_tensor).unwrap();
    let wgpu_result = wgpu_client.angle(&wgpu_tensor).unwrap();

    let cpu_output: Vec<f32> = cpu_result.to_vec();
    let wgpu_output: Vec<f32> = wgpu_result.to_vec();

    for (i, (cpu_val, wgpu_val)) in cpu_output.iter().zip(wgpu_output.iter()).enumerate() {
        assert!(
            (cpu_val - wgpu_val).abs() < 1e-6,
            "Mismatch at index {}: CPU = {}, WebGPU = {}",
            i,
            cpu_val,
            wgpu_val
        );
    }

    // Verify correct values
    assert_eq!(cpu_output[0], 0.0); // angle(1.0) = 0
    assert!((cpu_output[1] - std::f32::consts::PI).abs() < 1e-6); // angle(-2.0) = π
    assert_eq!(cpu_output[2], 0.0); // angle(3.0) = 0
    assert!((cpu_output[3] - std::f32::consts::PI).abs() < 1e-6); // angle(-5.0) = π
    assert_eq!(cpu_output[4], 0.0); // angle(0.0) = 0
}

// ============================================================================
// CUDA Complex Arithmetic Tests (add, sub, mul, div)
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_add() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let a_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let b_data = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

    let cpu_a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2], &cpu_device);
    let cpu_b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2], &cpu_device);
    let cuda_a = Tensor::<CudaRuntime>::from_slice(&a_data, &[2], &cuda_device);
    let cuda_b = Tensor::<CudaRuntime>::from_slice(&b_data, &[2], &cuda_device);

    let cpu_result = cpu_client.add(&cpu_a, &cpu_b).unwrap();
    let cuda_result = cuda_client.add(&cuda_a, &cuda_b).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // (1+2i) + (5+6i) = 6+8i
    // (3+4i) + (7+8i) = 10+12i
    assert!((cpu_output[0].re - 6.0).abs() < 1e-6);
    assert!((cpu_output[0].im - 8.0).abs() < 1e-6);

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_mul() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let a_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let b_data = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

    let cpu_a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2], &cpu_device);
    let cpu_b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2], &cpu_device);
    let cuda_a = Tensor::<CudaRuntime>::from_slice(&a_data, &[2], &cuda_device);
    let cuda_b = Tensor::<CudaRuntime>::from_slice(&b_data, &[2], &cuda_device);

    let cpu_result = cpu_client.mul(&cpu_a, &cpu_b).unwrap();
    let cuda_result = cuda_client.mul(&cuda_a, &cuda_b).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // (1+2i) * (5+6i) = 1*5 - 2*6 + (1*6 + 2*5)i = -7 + 16i
    assert!((cpu_output[0].re - (-7.0)).abs() < 1e-6);
    assert!((cpu_output[0].im - 16.0).abs() < 1e-6);

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_div() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let a_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let b_data = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];

    let cpu_a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2], &cpu_device);
    let cpu_b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2], &cpu_device);
    let cuda_a = Tensor::<CudaRuntime>::from_slice(&a_data, &[2], &cuda_device);
    let cuda_b = Tensor::<CudaRuntime>::from_slice(&b_data, &[2], &cuda_device);

    let cpu_result = cpu_client.div(&cpu_a, &cpu_b).unwrap();
    let cuda_result = cuda_client.div(&cuda_a, &cuda_b).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // (1+2i) / (1+i) = (1+2i)(1-i) / 2 = (3+i) / 2 = 1.5 + 0.5i
    assert!((cpu_output[0].re - 1.5).abs() < 1e-6);
    assert!((cpu_output[0].im - 0.5).abs() < 1e-6);

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_neg() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2], &cuda_device);

    let cuda_result = cuda_client.neg(&cuda_tensor).unwrap();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // -(1+2i) = -1 - 2i
    assert!(
        (cuda_output[0].re - (-1.0)).abs() < 1e-6,
        "Expected -1.0, got {}",
        cuda_output[0].re
    );
    assert!(
        (cuda_output[0].im - (-2.0)).abs() < 1e-6,
        "Expected -2.0, got {}",
        cuda_output[0].im
    );

    // -(-3+4i) = 3 - 4i
    assert!(
        (cuda_output[1].re - 3.0).abs() < 1e-6,
        "Expected 3.0, got {}",
        cuda_output[1].re
    );
    assert!(
        (cuda_output[1].im - (-4.0)).abs() < 1e-6,
        "Expected -4.0, got {}",
        cuda_output[1].im
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_sqrt() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    // sqrt(3+4i) = 2+i (since (2+i)^2 = 4 + 4i - 1 = 3 + 4i)
    let data = vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 2.0)];
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2], &cuda_device);

    let cuda_result = cuda_client.sqrt(&cuda_tensor).unwrap();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // sqrt(3+4i) = 2+i
    assert!(
        (cuda_output[0].re - 2.0).abs() < 1e-5,
        "sqrt(3+4i) real: expected 2.0, got {}",
        cuda_output[0].re
    );
    assert!(
        (cuda_output[0].im - 1.0).abs() < 1e-5,
        "sqrt(3+4i) imag: expected 1.0, got {}",
        cuda_output[0].im
    );

    // sqrt(2i) = 1+i (since (1+i)^2 = 1 + 2i - 1 = 2i)
    assert!(
        (cuda_output[1].re - 1.0).abs() < 1e-5,
        "sqrt(2i) real: expected 1.0, got {}",
        cuda_output[1].re
    );
    assert!(
        (cuda_output[1].im - 1.0).abs() < 1e-5,
        "sqrt(2i) imag: expected 1.0, got {}",
        cuda_output[1].im
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_complex64_exp() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    // e^(i*pi) = -1 + 0i (Euler's identity)
    // e^(1+0i) = e ≈ 2.718
    let data = vec![
        Complex64::new(0.0, std::f32::consts::PI),
        Complex64::new(1.0, 0.0),
    ];
    let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2], &cuda_device);

    let cuda_result = cuda_client.exp(&cuda_tensor).unwrap();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    // e^(i*pi) = -1 (Euler's identity)
    assert!(
        (cuda_output[0].re - (-1.0)).abs() < 1e-5,
        "e^(i*pi) real: expected -1.0, got {}",
        cuda_output[0].re
    );
    assert!(
        cuda_output[0].im.abs() < 1e-5,
        "e^(i*pi) imag: expected 0.0, got {}",
        cuda_output[0].im
    );

    // e^(1+0i) = e
    assert!(
        (cuda_output[1].re - std::f32::consts::E).abs() < 1e-5,
        "e^1 real: expected {}, got {}",
        std::f32::consts::E,
        cuda_output[1].re
    );
    assert!(
        cuda_output[1].im.abs() < 1e-5,
        "e^1 imag: expected 0.0, got {}",
        cuda_output[1].im
    );
}

// ============================================================================
// make_complex Tests
// ============================================================================

#[test]
fn test_make_complex_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let imag = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device);

    let result = client.make_complex(&real, &imag).unwrap();

    assert_eq!(result.dtype(), DType::Complex64);
    assert_eq!(result.shape(), &[3]);

    let output: Vec<Complex64> = result.to_vec();
    assert_eq!(output[0], Complex64::new(1.0, 4.0));
    assert_eq!(output[1], Complex64::new(2.0, 5.0));
    assert_eq!(output[2], Complex64::new(3.0, 6.0));
}

#[test]
fn test_make_complex_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let real = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device);
    let imag = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 4.0], &[2], &device);

    let result = client.make_complex(&real, &imag).unwrap();

    assert_eq!(result.dtype(), DType::Complex128);
    assert_eq!(result.shape(), &[2]);

    let output: Vec<Complex128> = result.to_vec();
    assert_eq!(output[0], Complex128::new(1.0, 3.0));
    assert_eq!(output[1], Complex128::new(2.0, 4.0));
}

#[test]
fn test_make_complex_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x3 matrix
    let real =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let imag =
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3], &device);

    let result = client.make_complex(&real, &imag).unwrap();

    assert_eq!(result.dtype(), DType::Complex64);
    assert_eq!(result.shape(), &[2, 3]);

    let output: Vec<Complex64> = result.to_vec();
    assert_eq!(output[0], Complex64::new(1.0, 0.1));
    assert_eq!(output[5], Complex64::new(6.0, 0.6));
}

#[test]
fn test_make_complex_shape_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let imag = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device);

    let result = client.make_complex(&real, &imag);
    assert!(result.is_err());
}

#[test]
fn test_make_complex_dtype_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let real = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let imag = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 4.0], &[2], &device);

    let result = client.make_complex(&real, &imag);
    assert!(result.is_err());
}

#[test]
fn test_make_complex_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let real = Tensor::<CpuRuntime>::from_slice::<f32>(&[], &[0], &device);
    let imag = Tensor::<CpuRuntime>::from_slice::<f32>(&[], &[0], &device);

    let result = client.make_complex(&real, &imag).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.numel(), 0);
}

// ============================================================================
// complex_mul_real Tests
// ============================================================================

#[test]
fn test_complex64_mul_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.5], &[2], &device);

    let result = client.complex_mul_real(&complex, &real).unwrap();

    assert_eq!(result.dtype(), DType::Complex64);
    assert_eq!(result.shape(), &[2]);

    let output: Vec<Complex64> = result.to_vec();
    // (1+2i) * 2 = 2 + 4i
    assert!((output[0].re - 2.0).abs() < 1e-6);
    assert!((output[0].im - 4.0).abs() < 1e-6);
    // (3+4i) * 0.5 = 1.5 + 2i
    assert!((output[1].re - 1.5).abs() < 1e-6);
    assert!((output[1].im - 2.0).abs() < 1e-6);
}

#[test]
fn test_complex128_mul_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex128::new(1.0, 2.0), Complex128::new(3.0, 4.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 0.25], &[2], &device);

    let result = client.complex_mul_real(&complex, &real).unwrap();

    assert_eq!(result.dtype(), DType::Complex128);

    let output: Vec<Complex128> = result.to_vec();
    // (1+2i) * 3 = 3 + 6i
    assert!((output[0].re - 3.0).abs() < 1e-10);
    assert!((output[0].im - 6.0).abs() < 1e-10);
    // (3+4i) * 0.25 = 0.75 + 1i
    assert!((output[1].re - 0.75).abs() < 1e-10);
    assert!((output[1].im - 1.0).abs() < 1e-10);
}

#[test]
fn test_complex_mul_real_dtype_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(&[Complex64::new(1.0, 2.0)], &[1], &device);
    // Wrong dtype: Complex64 needs F32, not F64
    let real = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device);

    let result = client.complex_mul_real(&complex, &real);
    assert!(result.is_err());
}

#[test]
fn test_complex_mul_real_shape_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.5, 1.0], &[3], &device);

    let result = client.complex_mul_real(&complex, &real);
    assert!(result.is_err());
}

#[test]
fn test_real_mul_complex() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test that real_mul_complex produces same result as complex_mul_real
    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.5], &[2], &device);

    let result1 = client.complex_mul_real(&complex, &real).unwrap();
    let result2 = client.real_mul_complex(&real, &complex).unwrap();

    let output1: Vec<Complex64> = result1.to_vec();
    let output2: Vec<Complex64> = result2.to_vec();

    for (a, b) in output1.iter().zip(output2.iter()) {
        assert_eq!(a.re, b.re);
        assert_eq!(a.im, b.im);
    }
}

// ============================================================================
// complex_div_real Tests
// ============================================================================

#[test]
fn test_complex64_div_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex64::new(4.0, 6.0), Complex64::new(2.0, 4.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0], &[2], &device);

    let result = client.complex_div_real(&complex, &real).unwrap();

    assert_eq!(result.dtype(), DType::Complex64);
    assert_eq!(result.shape(), &[2]);

    let output: Vec<Complex64> = result.to_vec();
    // (4+6i) / 2 = 2 + 3i
    assert!((output[0].re - 2.0).abs() < 1e-6);
    assert!((output[0].im - 3.0).abs() < 1e-6);
    // (2+4i) / 2 = 1 + 2i
    assert!((output[1].re - 1.0).abs() < 1e-6);
    assert!((output[1].im - 2.0).abs() < 1e-6);
}

#[test]
fn test_complex128_div_real() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(
        &[Complex128::new(6.0, 9.0), Complex128::new(4.0, 8.0)],
        &[2],
        &device,
    );
    let real = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 4.0], &[2], &device);

    let result = client.complex_div_real(&complex, &real).unwrap();

    assert_eq!(result.dtype(), DType::Complex128);

    let output: Vec<Complex128> = result.to_vec();
    // (6+9i) / 3 = 2 + 3i
    assert!((output[0].re - 2.0).abs() < 1e-10);
    assert!((output[0].im - 3.0).abs() < 1e-10);
    // (4+8i) / 4 = 1 + 2i
    assert!((output[1].re - 1.0).abs() < 1e-10);
    assert!((output[1].im - 2.0).abs() < 1e-10);
}

#[test]
fn test_complex_div_real_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let complex = Tensor::<CpuRuntime>::from_slice(&[Complex64::new(1.0, 2.0)], &[1], &device);
    let real = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);

    let result = client.complex_div_real(&complex, &real).unwrap();
    let output: Vec<Complex64> = result.to_vec();

    // Division by zero produces Inf
    assert!(output[0].re.is_infinite());
    assert!(output[0].im.is_infinite());
}

// ============================================================================
// Backend Parity Tests for New Operations
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_make_complex() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let real_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let imag_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];

    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[5], &cpu_device);
    let cpu_imag = Tensor::<CpuRuntime>::from_slice(&imag_data, &[5], &cpu_device);
    let cuda_real = Tensor::<CudaRuntime>::from_slice(&real_data, &[5], &cuda_device);
    let cuda_imag = Tensor::<CudaRuntime>::from_slice(&imag_data, &[5], &cuda_device);

    let cpu_result = cpu_client.make_complex(&cpu_real, &cpu_imag).unwrap();
    let cuda_result = cuda_client.make_complex(&cuda_real, &cuda_imag).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_complex_mul_real() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let complex_data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(-5.0, 6.0),
    ];
    let real_data = vec![2.0f32, 0.5, -1.0];

    let cpu_complex = Tensor::<CpuRuntime>::from_slice(&complex_data, &[3], &cpu_device);
    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[3], &cpu_device);
    let cuda_complex = Tensor::<CudaRuntime>::from_slice(&complex_data, &[3], &cuda_device);
    let cuda_real = Tensor::<CudaRuntime>::from_slice(&real_data, &[3], &cuda_device);

    let cpu_result = cpu_client
        .complex_mul_real(&cpu_complex, &cpu_real)
        .unwrap();
    let cuda_result = cuda_client
        .complex_mul_real(&cuda_complex, &cuda_real)
        .unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_parity_complex_div_real() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    if !numr::runtime::cuda::is_cuda_available() {
        println!("CUDA not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);

    let complex_data = vec![
        Complex64::new(4.0, 6.0),
        Complex64::new(3.0, 9.0),
        Complex64::new(-10.0, 5.0),
    ];
    let real_data = vec![2.0f32, 3.0, 5.0];

    let cpu_complex = Tensor::<CpuRuntime>::from_slice(&complex_data, &[3], &cpu_device);
    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[3], &cpu_device);
    let cuda_complex = Tensor::<CudaRuntime>::from_slice(&complex_data, &[3], &cuda_device);
    let cuda_real = Tensor::<CudaRuntime>::from_slice(&real_data, &[3], &cuda_device);

    let cpu_result = cpu_client
        .complex_div_real(&cpu_complex, &cpu_real)
        .unwrap();
    let cuda_result = cuda_client
        .complex_div_real(&cuda_complex, &cuda_real)
        .unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let cuda_output: Vec<Complex64> = cuda_result.to_vec();

    for (cpu_val, cuda_val) in cpu_output.iter().zip(cuda_output.iter()) {
        assert!(
            (cpu_val.re - cuda_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, CUDA={}",
            cpu_val.re,
            cuda_val.re
        );
        assert!(
            (cpu_val.im - cuda_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, CUDA={}",
            cpu_val.im,
            cuda_val.im
        );
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_cpu_parity_make_complex() {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};

    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let wgpu_device = WgpuDevice::new(0);
    let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

    let real_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let imag_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];

    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[5], &cpu_device);
    let cpu_imag = Tensor::<CpuRuntime>::from_slice(&imag_data, &[5], &cpu_device);
    let wgpu_real = Tensor::<WgpuRuntime>::from_slice(&real_data, &[5], &wgpu_device);
    let wgpu_imag = Tensor::<WgpuRuntime>::from_slice(&imag_data, &[5], &wgpu_device);

    let cpu_result = cpu_client.make_complex(&cpu_real, &cpu_imag).unwrap();
    let wgpu_result = wgpu_client.make_complex(&wgpu_real, &wgpu_imag).unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let wgpu_output: Vec<Complex64> = wgpu_result.to_vec();

    for (cpu_val, wgpu_val) in cpu_output.iter().zip(wgpu_output.iter()) {
        assert!(
            (cpu_val.re - wgpu_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, WebGPU={}",
            cpu_val.re,
            wgpu_val.re
        );
        assert!(
            (cpu_val.im - wgpu_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, WebGPU={}",
            cpu_val.im,
            wgpu_val.im
        );
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_cpu_parity_complex_mul_real() {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};

    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let wgpu_device = WgpuDevice::new(0);
    let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

    let complex_data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(-5.0, 6.0),
    ];
    let real_data = vec![2.0f32, 0.5, -1.0];

    let cpu_complex = Tensor::<CpuRuntime>::from_slice(&complex_data, &[3], &cpu_device);
    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[3], &cpu_device);
    let wgpu_complex = Tensor::<WgpuRuntime>::from_slice(&complex_data, &[3], &wgpu_device);
    let wgpu_real = Tensor::<WgpuRuntime>::from_slice(&real_data, &[3], &wgpu_device);

    let cpu_result = cpu_client
        .complex_mul_real(&cpu_complex, &cpu_real)
        .unwrap();
    let wgpu_result = wgpu_client
        .complex_mul_real(&wgpu_complex, &wgpu_real)
        .unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let wgpu_output: Vec<Complex64> = wgpu_result.to_vec();

    for (cpu_val, wgpu_val) in cpu_output.iter().zip(wgpu_output.iter()) {
        assert!(
            (cpu_val.re - wgpu_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, WebGPU={}",
            cpu_val.re,
            wgpu_val.re
        );
        assert!(
            (cpu_val.im - wgpu_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, WebGPU={}",
            cpu_val.im,
            wgpu_val.im
        );
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_cpu_parity_complex_div_real() {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};

    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let wgpu_device = WgpuDevice::new(0);
    let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

    let complex_data = vec![
        Complex64::new(4.0, 6.0),
        Complex64::new(3.0, 9.0),
        Complex64::new(-10.0, 5.0),
    ];
    let real_data = vec![2.0f32, 3.0, 5.0];

    let cpu_complex = Tensor::<CpuRuntime>::from_slice(&complex_data, &[3], &cpu_device);
    let cpu_real = Tensor::<CpuRuntime>::from_slice(&real_data, &[3], &cpu_device);
    let wgpu_complex = Tensor::<WgpuRuntime>::from_slice(&complex_data, &[3], &wgpu_device);
    let wgpu_real = Tensor::<WgpuRuntime>::from_slice(&real_data, &[3], &wgpu_device);

    let cpu_result = cpu_client
        .complex_div_real(&cpu_complex, &cpu_real)
        .unwrap();
    let wgpu_result = wgpu_client
        .complex_div_real(&wgpu_complex, &wgpu_real)
        .unwrap();

    let cpu_output: Vec<Complex64> = cpu_result.to_vec();
    let wgpu_output: Vec<Complex64> = wgpu_result.to_vec();

    for (cpu_val, wgpu_val) in cpu_output.iter().zip(wgpu_output.iter()) {
        assert!(
            (cpu_val.re - wgpu_val.re).abs() < 1e-6,
            "Real mismatch: CPU={}, WebGPU={}",
            cpu_val.re,
            wgpu_val.re
        );
        assert!(
            (cpu_val.im - wgpu_val.im).abs() < 1e-6,
            "Imag mismatch: CPU={}, WebGPU={}",
            cpu_val.im,
            wgpu_val.im
        );
    }
}

// ============================================================================
// Large Tensor Tests (for parallelization code paths)
// ============================================================================

#[test]
fn test_make_complex_large() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Large tensor (10000 elements) to test Rayon parallelization
    let numel = 10000;
    let real_data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let imag_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1).collect();

    let real = Tensor::<CpuRuntime>::from_slice(&real_data, &[numel], &device);
    let imag = Tensor::<CpuRuntime>::from_slice(&imag_data, &[numel], &device);

    let result = client.make_complex(&real, &imag).unwrap();

    assert_eq!(result.shape(), &[numel]);
    assert_eq!(result.numel(), numel);
    assert_eq!(result.dtype(), DType::Complex64);

    let output: Vec<Complex64> = result.to_vec();
    assert_eq!(output.len(), numel);

    // Spot check
    assert!((output[0].re - 0.0).abs() < 1e-6);
    assert!((output[0].im - 0.0).abs() < 1e-6);
    assert!((output[100].re - 100.0).abs() < 1e-6);
    assert!((output[100].im - 10.0).abs() < 1e-6);
}

#[test]
fn test_complex_mul_real_large() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let numel = 10000;
    let complex_data: Vec<Complex64> = (0..numel)
        .map(|i| Complex64::new(i as f32, (i as f32) * 2.0))
        .collect();
    let real_data: Vec<f32> = (0..numel)
        .map(|i| if i % 2 == 0 { 2.0 } else { 0.5 })
        .collect();

    let complex = Tensor::<CpuRuntime>::from_slice(&complex_data, &[numel], &device);
    let real = Tensor::<CpuRuntime>::from_slice(&real_data, &[numel], &device);

    let result = client.complex_mul_real(&complex, &real).unwrap();

    assert_eq!(result.shape(), &[numel]);
    let output: Vec<Complex64> = result.to_vec();
    assert_eq!(output.len(), numel);
}
