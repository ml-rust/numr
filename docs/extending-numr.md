# Extending numr

This guide covers how to extend numr with custom operations and kernels.

## Architecture Overview

numr separates **operation traits** (the interface) from **kernel implementations** (the computation):

```
┌─────────────────────────────────────────────────────────────────┐
│                     Operation Trait                             │
│              (defines interface, validates inputs)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
     │   CPU   │      │  CUDA   │      │  WebGPU │
     │  Impl   │      │  Impl   │      │  Impl   │
     └────┬────┘      └────┬────┘      └────┬────┘
          │                │                │
     ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
     │  SIMD   │      │   PTX   │      │  WGSL   │
     │ Kernel  │      │ Kernel  │      │ Shader  │
     └─────────┘      └─────────┘      └─────────┘
```

numr provides default kernels for all operations. You can:

1. **Use default kernels** - Works out of the box for all supported operations
2. **Replace specific kernels** - Swap in optimized kernels for hot paths
3. **Add new operations** - Define new traits and implement kernels for all backends

## Using Default Kernels

All operations work immediately with numr's built-in kernels:

```rust
use numr::prelude::*;

let a = Tensor::<CpuRuntime>::randn(&[1024, 1024], &device)?;
let b = Tensor::<CpuRuntime>::randn(&[1024, 1024], &device)?;

// Uses numr's tiled GEMM kernel
let c = client.matmul(&a, &b)?;

// Uses numr's SIMD reduction kernel
let sum = client.sum(&c)?;
```

## Custom CPU Kernels

### Replacing a Kernel

To replace numr's default kernel with your own optimized version:

```rust
use numr::ops::KernelDispatch;
use numr::runtime::cpu::CpuClient;

// Define your custom kernel
fn my_optimized_add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    // Your optimized implementation
    // e.g., using specific SIMD intrinsics, cache-aware tiling, etc.
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

// Register the custom kernel
impl KernelDispatch for MyCustomOps {
    fn dispatch_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        my_optimized_add_kernel(a, b, out);
    }
}
```

### Writing SIMD Kernels

numr's CPU kernels use portable SIMD. Here's the pattern:

```rust
use std::simd::{f32x8, SimdFloat};

pub fn add_simd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let simd_width = 8;
    let simd_end = n - (n % simd_width);

    // SIMD loop
    for i in (0..simd_end).step_by(simd_width) {
        let va = f32x8::from_slice(&a[i..]);
        let vb = f32x8::from_slice(&b[i..]);
        let vout = va + vb;
        vout.copy_to_slice(&mut out[i..]);
    }

    // Scalar remainder
    for i in simd_end..n {
        out[i] = a[i] + b[i];
    }
}
```

For architecture-specific intrinsics:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_avx512_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    // AVX-512 implementation
    let n = a.len();
    let mut i = 0;

    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let vout = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 16;
    }

    // Handle remainder with scalar or smaller SIMD
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}
```

## Custom CUDA Kernels

### PTX Kernel Template

numr compiles CUDA kernels to PTX at build time. Here's the pattern:

```cuda
// src/runtime/cuda/kernels/my_op.cu

template<typename T>
__global__ void my_op_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = /* your operation */;
    }
}

// Explicit instantiations for each dtype
extern "C" __global__ void my_op_f32(const float* a, const float* b, float* out, unsigned int n) {
    my_op_kernel<float>(a, b, out, n);
}

extern "C" __global__ void my_op_f64(const double* a, const double* b, double* out, unsigned int n) {
    my_op_kernel<double>(a, b, out, n);
}

extern "C" __global__ void my_op_i32(const int* a, const int* b, int* out, unsigned int n) {
    my_op_kernel<int>(a, b, out, n);
}
```

### Launching Custom CUDA Kernels

```rust
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MyOps<CudaRuntime> for CudaClient {
    fn my_op(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let n = a.numel();
        let output = self.allocate_like(a)?;

        // Select kernel based on dtype
        let kernel_name = match a.dtype() {
            DType::F32 => "my_op_f32",
            DType::F64 => "my_op_f64",
            DType::I32 => "my_op_i32",
            _ => return Err(Error::UnsupportedDType { dtype: a.dtype(), op: "my_op" }),
        };

        // Launch kernel
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        self.launch_kernel(
            kernel_name,
            grid_size as u32,
            block_size as u32,
            &[
                a.as_kernel_arg(),
                b.as_kernel_arg(),
                output.as_kernel_arg(),
                &(n as u32),
            ],
        )?;

        Ok(output)
    }
}
```

### Shared Memory and Tiling

For operations that benefit from shared memory:

```cuda
template<typename T, int TILE_SIZE>
__global__ void tiled_matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K
) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < N && tile * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Custom WebGPU Shaders

### WGSL Shader Template

```wgsl
// src/runtime/wgpu/shaders/my_op.wgsl

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn my_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) {
        return;
    }
    out[idx] = /* your operation */;
}
```

### Multi-dtype WGSL

WebGPU requires separate shaders per dtype. Use shader generation:

```rust
fn generate_my_op_shader(dtype: DType) -> String {
    let wgsl_type = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => panic!("Unsupported dtype for WebGPU: {:?}", dtype),
    };

    format!(r#"
@group(0) @binding(0) var<storage, read> a: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read> b: array<{wgsl_type}>;
@group(0) @binding(2) var<storage, read_write> out: array<{wgsl_type}>;

@compute @workgroup_size(256)
fn my_op(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{
        return;
    }}
    out[idx] = a[idx] + b[idx];
}}
"#)
}
```

### Launching Custom WebGPU Shaders

```rust
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl MyOps<WgpuRuntime> for WgpuClient {
    fn my_op(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let output = self.allocate_like(a)?;

        // Get or compile shader for this dtype
        let shader = self.get_or_compile_shader("my_op", a.dtype())?;

        // Calculate workgroup dispatch
        let workgroup_size = 256;
        let workgroup_count = (a.numel() + workgroup_size - 1) / workgroup_size;

        // Dispatch compute shader
        self.dispatch_compute(
            &shader,
            &[a.buffer(), b.buffer(), output.buffer()],
            workgroup_count as u32,
        )?;

        Ok(output)
    }
}
```

## Adding a New Operation

To add a completely new operation to numr:

### 1. Define the Trait

Create `src/ops/traits/my_category.rs`:

```rust
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Operations for my category
pub trait MyOps<R: Runtime> {
    /// Computes my_op(a, b) element-wise
    fn my_op(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Computes my_other_op with a scalar parameter
    fn my_other_op(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;
}
```

### 2. Export from mod.rs

Add to `src/ops/traits/mod.rs`:

```rust
pub mod my_category;
pub use my_category::MyOps;
```

### 3. CPU Implementation

Create `src/ops/cpu/my_category.rs`:

```rust
use crate::ops::traits::MyOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;
use crate::error::Result;
use crate::dtype::dispatch_dtype;

impl MyOps<CpuRuntime> for CpuClient {
    fn my_op(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        // Validate shapes
        a.shape().broadcast_with(b.shape())?;

        let output = self.allocate_output_shape(&broadcast_shape)?;

        // Dispatch based on dtype
        dispatch_dtype!(a.dtype(), T => {
            let a_data = a.as_slice::<T>()?;
            let b_data = b.as_slice::<T>()?;
            let out_data = output.as_mut_slice::<T>()?;

            // Call kernel
            kernels::my_op_kernel::<T>(a_data, b_data, out_data);
        });

        Ok(output)
    }

    fn my_other_op(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_op() {
        // Test implementation
    }
}
```

### 4. CUDA Implementation

Create `src/ops/cuda/my_category.rs` with similar structure, launching PTX kernels.

### 5. WebGPU Implementation

Create `src/ops/wgpu/my_category.rs` with similar structure, dispatching WGSL shaders.

### 6. Register in Backend mod.rs

Add to each backend's `mod.rs`:

```rust
pub mod my_category;
```

## Dtype Dispatch

numr uses the `dispatch_dtype!` macro for runtime dtype dispatch:

```rust
use crate::dtype::dispatch_dtype;

fn process_tensor<R: Runtime>(tensor: &Tensor<R>) -> Result<()> {
    dispatch_dtype!(tensor.dtype(), T => {
        // T is now the concrete type (f32, f64, i32, etc.)
        let data = tensor.as_slice::<T>()?;
        process_data::<T>(data);
    });
    Ok(())
}
```

For operations that only support specific dtypes:

```rust
dispatch_dtype!(tensor.dtype(), T => {
    match tensor.dtype() {
        DType::F32 | DType::F64 => {
            // Float-only operation
        }
        _ => return Err(Error::UnsupportedDType {
            dtype: tensor.dtype(),
            op: "my_float_op",
        }),
    }
});
```

## Testing Custom Kernels

### Numerical Accuracy

```rust
#[test]
fn test_kernel_accuracy() {
    let a = Tensor::<CpuRuntime>::randn(&[1024], &device)?;
    let b = Tensor::<CpuRuntime>::randn(&[1024], &device)?;

    let result = client.my_op(&a, &b)?;
    let expected = compute_expected(&a, &b);

    assert_allclose(&result, &expected, 1e-6, 1e-7);
}
```

### Backend Parity

```rust
#[test]
fn test_backend_parity() {
    let a_cpu = Tensor::<CpuRuntime>::randn(&[1024], &cpu_device)?;
    let b_cpu = Tensor::<CpuRuntime>::randn(&[1024], &cpu_device)?;

    let result_cpu = cpu_client.my_op(&a_cpu, &b_cpu)?;

    #[cfg(feature = "cuda")]
    {
        let a_cuda = a_cpu.to_device(&cuda_device)?;
        let b_cuda = b_cpu.to_device(&cuda_device)?;
        let result_cuda = cuda_client.my_op(&a_cuda, &b_cuda)?;

        assert_allclose(
            &result_cpu.to_vec::<f32>()?,
            &result_cuda.to_cpu()?.to_vec::<f32>()?,
            1e-5,
            1e-6,
        );
    }
}
```

### Edge Cases

Always test:

- Empty tensors (`[]`)
- Scalar tensors (`[1]`)
- Non-contiguous tensors (from transpose/slice)
- Broadcasting edge cases
- Special values (NaN, Inf, subnormals)
- All supported dtypes

## Performance Considerations

### Memory Access Patterns

- **Coalesced access**: Adjacent threads access adjacent memory
- **Avoid strided access**: Poor cache utilization
- **Contiguous tensors**: Call `.contiguous()` before kernels if needed

### CUDA-specific

- **Occupancy**: Balance registers vs threads per block
- **Shared memory**: Use for tile-based algorithms
- **Warp divergence**: Avoid conditionals within warps

### WebGPU-specific

- **Workgroup size**: 256 is a safe default, tune per GPU
- **Buffer alignment**: 256-byte alignment for storage buffers
- **Uniform buffers**: Use for small, read-only parameters

## Debugging

### CPU

```rust
// Print intermediate values
eprintln!("a[0..4] = {:?}", &a.as_slice::<f32>()?[0..4]);
```

### CUDA

Use `cuda-gdb` or print from device:

```cuda
if (idx == 0) {
    printf("First element: %f\n", a[0]);
}
```

### WebGPU

Use browser WebGPU debugging tools or dump buffers:

```rust
let data = client.read_buffer(&output)?;
eprintln!("Output: {:?}", &data[0..10]);
```
