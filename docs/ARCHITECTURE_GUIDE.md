# numr Architecture Guide

This document describes the internal architecture of numr for contributors and
adopters migrating from ndarray, nalgebra, or PyTorch-like workflows.

---

## Overview

numr is a multi-backend tensor library. The same user code runs on CPU, CUDA,
and WebGPU without modification — backends are selected at compile time via
feature flags, and tensor operations dispatch to backend-specific kernels
through Rust's trait system.

```
User code: client.add(&a, &b)
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
   CPU        CUDA       WebGPU
  (SIMD)    (PTX/nvcc)  (WGSL)
```

---

## Runtime Trait Hierarchy

Every backend implements three traits that together define a compute target.

### `Runtime` — backend identity

```
src/runtime/traits/runtime.rs
```

```rust
pub trait Runtime: Clone + Send + Sync + 'static {
    type Device: Device;
    type Client: RuntimeClient<Self>;
    type Allocator: Allocator;
    type RawHandle: Send + Sync;

    fn name() -> &'static str;
    fn allocate(size_bytes: usize, device: &Self::Device) -> Result<u64>;
    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device);
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> Result<()>;
    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device) -> Result<()>;
    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, device: &Self::Device) -> Result<()>;
    fn default_device() -> Self::Device;
    fn default_client(device: &Self::Device) -> Self::Client;
    // ...
}
```

`Runtime` owns the raw memory interface. It is purely a type-level marker
with static methods — no instances are created.

Concrete implementations: `CpuRuntime`, `CudaRuntime`, `WgpuRuntime`.

### `Device` — a specific GPU or CPU

```
src/runtime/traits/device.rs
```

```rust
pub trait Device: Clone + Send + Sync + 'static {
    fn id(&self) -> usize;
    fn name(&self) -> String;
}
```

A lightweight handle identifying a particular piece of hardware. For CPU this
is a singleton; for CUDA it maps to a device ordinal.

### `RuntimeClient` — operation dispatcher

```
src/runtime/traits/client.rs
```

```rust
pub trait RuntimeClient<R: Runtime>: Clone + Send + Sync {
    fn device(&self) -> &R::Device;
    fn synchronize(&self);
}
```

The client owns any per-device state (CUDA stream, WebGPU queue, parallelism
config) and is the receiver for all operation trait methods.

**All tensor operations are methods on the client**, not on tensors:

```rust
let result = client.add(&a, &b)?;      // BinaryOps::add
let reduced = client.sum(&a, &[0], false)?; // ReduceOps::sum
```

This design makes it impossible to accidentally mix backends — the client's
type determines which kernels run.

---

## Tensor Layout

```
src/tensor/core.rs    — Tensor<R> struct
src/tensor/storage.rs — Storage<R>, reference-counted device memory
src/tensor/layout.rs  — Layout (shape + strides + offset)
```

### `Tensor<R: Runtime>`

```rust
pub struct Tensor<R: Runtime> {
    id: TensorId,        // unique ID for autograd tracking
    storage: Storage<R>,  // Arc-wrapped device memory
    layout: Layout,       // shape, strides, offset
}
```

### `Storage<R>`

```rust
struct StorageInner<R: Runtime> {
    ptr: u64,             // raw device pointer (GPU address or CPU ptr)
    len: usize,           // number of elements
    dtype: DType,         // element type
    device: R::Device,    // device where memory lives
    owned: bool,          // if true, deallocate on drop
}
```

Storage is `Arc`-wrapped. Multiple tensors can share the same allocation —
this is how zero-copy views work. Memory is freed when the last reference
drops, via `Runtime::deallocate()` in the `Drop` impl.

### `Layout`

```rust
pub struct Layout {
    shape: Shape,       // size along each dimension
    strides: Strides,   // element offset between consecutive elements per dim
    offset: usize,      // starting element index in storage
}
```

Strides follow row-major convention: shape `[2, 3, 4]` produces strides
`[12, 4, 1]`.

---

## Zero-Copy Views

These operations create a new `Tensor` sharing the same `Storage`, only
changing the `Layout`:

| Operation                 | What changes                                                 |
| ------------------------- | ------------------------------------------------------------ |
| `reshape`                 | New shape + recomputed strides (contiguous input only)       |
| `transpose(d0, d1)`       | Swaps shape[d0]/shape[d1] and strides[d0]/strides[d1]        |
| `permute`                 | Arbitrary dimension reordering via stride permutation        |
| `unsqueeze(dim)`          | Inserts size-1 dimension (stride = next dim's stride × size) |
| `squeeze(dim)`            | Removes size-1 dimension                                     |
| `narrow(dim, start, len)` | Adjusts offset + shape along one dimension                   |
| `broadcast_to`            | Sets stride=0 for broadcast dimensions                       |
| `flip`                    | Negates stride, adjusts offset                               |

No data is copied. The resulting tensor is a view into the original storage.

If an operation needs contiguous memory (e.g., kernel launch), call
`.contiguous()` which returns a new tensor with freshly allocated, contiguous
storage — or returns `self` if already contiguous.

---

## Operation Architecture

### Three-Layer Dispatch (Primitive Ops)

Primitive operations like `add`, `exp`, `sum` follow this pattern:

```
1. Trait definition     — src/ops/traits/{op}.rs
2. Backend impl         — src/ops/{backend}/{op}.rs
3. Backend kernel       — src/runtime/cpu/kernels/{op}.rs (CPU)
                          src/runtime/cuda/kernels/{op}.cu      (CUDA)
                          src/runtime/wgpu/shaders/{op}.wgsl    (WebGPU)
```

**Concrete example: `client.add(&a, &b)`**

```
src/ops/traits/binary.rs          trait BinaryOps<R> { fn add(...) }
        │
        ├─ src/ops/cpu/binary.rs           impl BinaryOps<CpuRuntime> for CpuClient
        │      │
        │      └─ src/runtime/cpu/helpers/binary.rs     shape validation, broadcast
        │             │
        │             └─ src/runtime/cpu/kernels/binary.rs   SIMD kernel (AVX2/NEON)
        │
        ├─ src/ops/cuda/binary.rs          impl BinaryOps<CudaRuntime> for CudaClient
        │      │
        │      └─ launches PTX kernel: binary.ptx → add_f32
        │
        └─ src/ops/wgpu/binary.rs          impl BinaryOps<WgpuRuntime> for WgpuClient
               │
               └─ dispatches WGSL shader: binary.wgsl → add entry point
```

### Four-Layer Dispatch (Composite Ops)

Composite operations (softmax, layernorm, unfold) add `impl_generic/` to
guarantee the same algorithm across all backends:

```
1. Trait definition     — src/ops/traits/{op}.rs
2. Generic algorithm    — src/ops/impl_generic/{op}.rs
3. Backend impl         — src/ops/{backend}/{op}.rs  (delegates to impl_generic)
4. Optional fused kernel
```

The generic algorithm calls only primitive ops, so all backends execute the
same sequence:

```rust
// src/ops/impl_generic/shape.rs
pub fn unfold_impl<R: Runtime, C: ShapeOps<R>>(
    client: &C,
    tensor: &Tensor<R>,
    dim: isize,
    size: usize,
    step: usize,
) -> Result<Tensor<R>> {
    // Uses narrow (primitive) + stack (primitive) + permute (view)
    // Same algorithm regardless of backend
}
```

Backend impls delegate:

```rust
impl ShapeOps<CudaRuntime> for CudaClient {
    fn unfold(&self, tensor: &Tensor<CudaRuntime>, ...) -> Result<...> {
        unfold_impl(self, tensor, dim, size, step) // same code path
    }
}
```

### Why This Matters

- Adding a new primitive op = new files, not modifying existing files
- Composite ops produce identical numerical results across backends
- Optional fused kernels (CUDA softmax, etc.) must match `impl_generic` output

---

## Backend Kernel Mechanisms

### CPU: SIMD Kernels

```
src/runtime/cpu/kernels/         — kernel entry points
src/runtime/cpu/kernels/simd/    — AVX2/AVX-512/NEON implementations
```

CPU kernels dispatch on dtype and architecture:

```rust
pub unsafe fn binary_op_kernel<T: Element>(op: BinaryOp, a: *const T, b: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    match T::DTYPE {
        DType::F32 => { simd::binary::binary_f32(op, a, b, out, len); return; }
        DType::F64 => { simd::binary::binary_f64(op, a, b, out, len); return; }
        _ => {}
    }
    binary_op_scalar(op, a, b, out, len); // scalar fallback
}
```

Parallelism is controlled via `ParallelismConfig` on `CpuClient`, which
configures thread count and chunk size for rayon-based parallel iteration.

### CUDA: PTX Kernel Loading

```
build.rs                            — compiles .cu → .ptx via nvcc
src/runtime/cuda/kernels/*.cu       — CUDA C++ source (templated per dtype)
src/runtime/cuda/kernels/loader.rs  — loads PTX, caches modules per device
```

**Lifecycle:**

1. `build.rs` runs `nvcc -ptx -O3 -arch=sm_75` on each `.cu` file
2. PTX files written to `$OUT_DIR`, path stored in `CUDA_KERNEL_DIR` env var
3. At runtime, first use loads PTX via `Ptx::from_file()` and creates a `CudaModule`
4. Module cached in a global `HashMap<(device_index, module_name), Arc<CudaModule>>`
5. Kernel functions retrieved from module by name (e.g., `"add_f32"`)

CUDA kernels use C++ templates with `extern "C"` linkage for per-dtype
instantiation:

```cuda
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" {
    __global__ void add_f32(const float* a, const float* b, float* out, unsigned int n)
        { add_kernel<float>(a, b, out, n); }
    __global__ void add_f64(const double* a, const double* b, double* out, unsigned int n)
        { add_kernel<double>(a, b, out, n); }
}
```

### WebGPU: WGSL Shader Dispatch

```
src/runtime/wgpu/shaders/          — WGSL source (embedded as Rust strings)
src/runtime/wgpu/shaders/pipeline.rs — shader compilation + pipeline cache
```

**Lifecycle:**

1. WGSL source is embedded in Rust code as string constants
2. First use: `device.create_shader_module()` compiles WGSL → `ShaderModule`
3. A `ComputePipeline` is created with bind group layout (buffer bindings)
4. Both module and pipeline cached in `PipelineCache` (keyed by shader name + entry point)
5. Dispatch: create bind group → encode compute pass → `queue.submit()`

WebGPU supports F32, I32, U32 natively, plus F16 with the
`shader-f16` feature. Unsupported dtypes return `Error::UnsupportedDType`.

---

## Autograd

```
src/autograd/var.rs        — Var<R> struct
src/autograd/grad_fn.rs    — GradFn trait
src/autograd/backward.rs   — backward() traversal
src/autograd/var_ops/      — differentiable operations (var_add, var_matmul, etc.)
```

### `Var<R: Runtime>`

```rust
pub struct Var<R: Runtime> {
    tensor: Tensor<R>,                     // underlying data
    id: TensorId,                          // graph node identity
    requires_grad: bool,                   // leaf flag
    grad_fn: Option<Arc<dyn GradFn<R>>>,   // backward function (None for leaves)
}
```

`Var` wraps `Tensor` with gradient-tracking metadata. During the forward pass,
`var_*` functions create new `Var` nodes with `grad_fn` closures that capture
references to parent nodes.

### Backward Pass

`backward(&loss, &client)` performs reverse-mode AD:

1. Topological sort of the computation graph from `loss` to leaves
2. Walk in reverse order, calling each node's `grad_fn` to propagate gradients
3. Return `GradStore` mapping `TensorId → Tensor<R>` (gradient tensors)

Gradients are regular tensors — they use the same backend and operations as
the forward pass.

---

## DType Dispatch

```
src/dtype/mod.rs       — DType enum
src/dtype/element.rs   — Element trait (type-level ↔ value-level bridge)
```

Every operation must handle all supported dtypes at runtime. The
`dispatch_dtype!` macro bridges from the `DType` enum to generic `T: Element`
code:

```rust
dispatch_dtype!(tensor.dtype(), T => {
    kernels::binary_op::<T>(op, a, b, out)?;
}, "add");
```

This generates a match statement that monomorphizes the kernel for each dtype.

---

## Design Rationale

### Why traits, not enum dispatch?

Trait-based dispatch provides:

- **Compile-time safety**: missing backend implementations are compile errors
- **Zero-cost abstraction**: no runtime vtable lookup for operation dispatch
- **Independent compilation**: each backend compiles separately, no cross-deps
- **Extensibility**: new backends implement existing traits without modifying core

### Why operations on client, not on Tensor?

- Client carries backend state (CUDA stream, WebGPU queue, thread pool config)
- Prevents accidentally mixing backends in one expression
- Makes the compute target explicit in every call

### Why no vendor library dependencies?

numr uses native kernels exclusively — no cuBLAS, MKL, or vendor wrappers.
This ensures:

- Code works on any hardware the backend supports
- No 10GB+ SDK installation requirements
- Full portability to new backends (WebGPU, ROCm)
- Predictable, auditable kernel behavior

---

## Module Map

```
src/
├── lib.rs              — entry point, prelude, DefaultRuntime
├── error.rs            — Error enum (thiserror)
├── dtype/              — DType, Element, Complex64/128, dispatch macros
├── tensor/             — Tensor<R>, Storage<R>, Layout
├── runtime/
│   ├── traits/         — Runtime, Device, RuntimeClient
│   ├── cpu/            — CpuRuntime, CpuClient, SIMD kernels
│   ├── cuda/           — CudaRuntime, CudaClient, PTX loader
│   └── wgpu/           — WgpuRuntime, WgpuClient, WGSL pipelines
├── ops/
│   ├── traits/         — one file per operation category
│   ├── impl_generic/   — shared algorithms for composite ops
│   ├── cpu/            — CPU trait impls
│   ├── cuda/           — CUDA trait impls
│   └── wgpu/           — WebGPU trait impls
├── algorithm/          — FFT, linalg, special functions, polynomials
├── autograd/           — Var<R>, GradFn, backward, var_ops/
└── sparse/             — SparseTensor, COO/CSR/CSC (feature-gated)
```
