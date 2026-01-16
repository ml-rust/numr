// Scalar CUDA kernels (tensor-scalar operations)
// Supports: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar
// Types: f32, f64, i32, i64

extern "C" {

// ============================================================================
// F32 Scalar Operations
// ============================================================================

__global__ void add_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], scalar);
    }
}

// ============================================================================
// F64 Scalar Operations
// ============================================================================

__global__ void add_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = pow(a[idx], scalar);
    }
}

// ============================================================================
// I32 Scalar Operations
// ============================================================================

__global__ void add_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

// ============================================================================
// I64 Scalar Operations
// ============================================================================

__global__ void add_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

} // extern "C"
