// Unary element-wise CUDA kernels
// Supports: neg, abs, sqrt, exp, log, sin, cos, tan, tanh, recip, square, floor, ceil, round
// Types: f32, f64

extern "C" {

// ============================================================================
// F32 Unary Operations
// ============================================================================

__global__ void neg_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(a[idx]);
    }
}

__global__ void sqrt_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(a[idx]);
    }
}

__global__ void exp_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(a[idx]);
    }
}

__global__ void log_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(a[idx]);
    }
}

__global__ void sin_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinf(a[idx]);
    }
}

__global__ void cos_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosf(a[idx]);
    }
}

__global__ void tan_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanf(a[idx]);
    }
}

__global__ void tanh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(a[idx]);
    }
}

__global__ void recip_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / a[idx];
    }
}

__global__ void square_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floorf(a[idx]);
    }
}

__global__ void ceil_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceilf(a[idx]);
    }
}

__global__ void round_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = roundf(a[idx]);
    }
}

// ============================================================================
// F64 Unary Operations
// ============================================================================

__global__ void neg_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabs(a[idx]);
    }
}

__global__ void sqrt_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrt(a[idx]);
    }
}

__global__ void exp_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(a[idx]);
    }
}

__global__ void log_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log(a[idx]);
    }
}

__global__ void sin_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sin(a[idx]);
    }
}

__global__ void cos_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cos(a[idx]);
    }
}

__global__ void tan_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tan(a[idx]);
    }
}

__global__ void tanh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanh(a[idx]);
    }
}

__global__ void recip_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0 / a[idx];
    }
}

__global__ void square_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floor(a[idx]);
    }
}

__global__ void ceil_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceil(a[idx]);
    }
}

__global__ void round_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = round(a[idx]);
    }
}

// ============================================================================
// I32 Unary Operations
// ============================================================================

__global__ void neg_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = abs(a[idx]);
    }
}

__global__ void square_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        out[idx] = val * val;
    }
}

// ============================================================================
// I64 Unary Operations
// ============================================================================

__global__ void neg_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = llabs(a[idx]);
    }
}

__global__ void square_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long val = a[idx];
        out[idx] = val * val;
    }
}

} // extern "C"
