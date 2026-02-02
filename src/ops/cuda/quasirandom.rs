//! Quasi-random sequence generation for CUDA runtime

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::quasirandom::{
    validate_halton_params, validate_latin_hypercube_params, validate_sobol_params,
};
use crate::ops::traits::QuasiRandomOps;
use crate::runtime::cuda::kernels::{
    launch_halton_f32, launch_halton_f64, launch_latin_hypercube_f32, launch_latin_hypercube_f64,
    launch_sobol_f32, launch_sobol_f64,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Supported data types for CUDA quasi-random operations
const SUPPORTED_DTYPES: &[DType] = &[DType::F32, DType::F64];

impl QuasiRandomOps<CudaRuntime> for CudaClient {
    fn sobol(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_sobol_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "sobol")?;

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device);

        // Launch CUDA kernel
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_sobol_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                DType::F64 => {
                    launch_sobol_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }

    fn halton(
        &self,
        n_points: usize,
        dimension: usize,
        skip: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_halton_params(n_points, dimension, dtype, SUPPORTED_DTYPES, "halton")?;

        // Allocate output: shape [n_points, dimension]
        let shape = vec![n_points, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device);

        // Launch CUDA kernel
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_halton_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                DType::F64 => {
                    launch_halton_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_points,
                        dimension,
                        skip,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }

    fn latin_hypercube(
        &self,
        n_samples: usize,
        dimension: usize,
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_latin_hypercube_params(
            n_samples,
            dimension,
            dtype,
            SUPPORTED_DTYPES,
            "latin_hypercube",
        )?;

        // Allocate output: shape [n_samples, dimension]
        let shape = vec![n_samples, dimension];
        let out = Tensor::<CudaRuntime>::empty(&shape, dtype, &self.device);

        // Generate random seed for shuffling
        let seed = generate_random_seed();

        // Launch CUDA kernel
        unsafe {
            match dtype {
                DType::F32 => {
                    launch_latin_hypercube_f32(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_samples,
                        dimension,
                        seed,
                    )?;
                }
                DType::F64 => {
                    launch_latin_hypercube_f64(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        out.storage().ptr(),
                        n_samples,
                        dimension,
                        seed,
                    )?;
                }
                _ => unreachable!("dtype validation should prevent this"),
            }
        }

        Ok(out)
    }
}

// ============================================================================
// Random Seed Generation Helper
// ============================================================================

/// Global atomic counter for generating unique seeds
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random seed combining atomic counter and system time.
///
/// This provides good entropy for parallel random number generation:
/// - Atomic counter ensures uniqueness across calls
/// - System time adds unpredictability
#[inline]
fn generate_random_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_component = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Combine counter and time using splitmix64-style mixing
    let mut z = counter.wrapping_add(time_component);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> (CudaDevice, CudaClient) {
        let device = CudaDevice::new(0).expect("CUDA device not available");
        let client = CudaRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_sobol_basic() {
        let (_device, client) = setup();

        let points = client.sobol(10, 2, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 2]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_halton_basic() {
        let (_device, client) = setup();

        let points = client.halton(10, 3, 0, DType::F32).unwrap();
        assert_eq!(points.shape(), &[10, 3]);

        // Verify points are in [0, 1)
        let data: Vec<f32> = points.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Point out of range: {}", val);
        }
    }

    #[test]
    fn test_latin_hypercube_basic() {
        let (_device, client) = setup();

        let samples = client.latin_hypercube(20, 4, DType::F32).unwrap();
        assert_eq!(samples.shape(), &[20, 4]);

        // Verify samples are in [0, 1)
        let data: Vec<f32> = samples.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0, "Sample out of range: {}", val);
        }
    }

    #[test]
    fn test_sobol_deterministic() {
        let (_device, client) = setup();

        let points1 = client.sobol(5, 2, 0, DType::F32).unwrap();
        let points2 = client.sobol(5, 2, 0, DType::F32).unwrap();

        let data1: Vec<f32> = points1.to_vec();
        let data2: Vec<f32> = points2.to_vec();

        // Same parameters should produce identical sequences
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert_eq!(v1, v2, "Sobol sequence not deterministic");
        }
    }

    #[test]
    fn test_error_zero_points() {
        let (_device, client) = setup();
        let result = client.sobol(0, 2, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unsupported_dtype() {
        let (_device, client) = setup();
        let result = client.sobol(10, 2, 0, DType::I32);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobol_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 6 dimensions (current implementation limit)
        let result = client.sobol(10, 6, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 6 dimensions (current implementation limit)
        // NOTE: Once full direction numbers are implemented, this should be updated to 1000
        let result = client.sobol(10, 7, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_halton_dimension_limit() {
        let (_device, client) = setup();

        // Should work up to 100 dimensions
        let result = client.halton(10, 100, 0, DType::F32);
        assert!(result.is_ok());

        // Should fail beyond 100
        let result = client.halton(10, 101, 0, DType::F32);
        assert!(result.is_err());
    }
}
