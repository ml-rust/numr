// Semiring Matrix Multiplication CUDA Kernels
// C[i,j] = reduce_k( combine(A[i,k], B[k,j]) )
//
// Semiring operations (passed as op parameter):
//   0 = MinPlus:  reduce=min, combine=+
//   1 = MaxPlus:  reduce=max, combine=+
//   2 = MaxMin:   reduce=max, combine=min
//   3 = MinMax:   reduce=min, combine=max
//   4 = OrAnd:    reduce=OR,  combine=AND
//   5 = PlusMax:  reduce=+,   combine=max
//
// Simple non-tiled kernel: one thread per output element.
// For semiring matmul, the inner loop is not amenable to the standard
// tiled GEMM shared-memory approach because the combine/reduce ops
// don't distribute the same way as (+, *). A simple kernel with good
// occupancy is the correct first implementation.

// ============================================================================
// F32 Kernels
// ============================================================================

extern "C" __global__ void semiring_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc;
    // Initialize with reduce identity
    switch (op) {
        case 0: // MinPlus: reduce=min, identity=+inf
        case 3: // MinMax: reduce=min, identity=+inf
            acc = __int_as_float(0x7f800000); // +inf
            break;
        case 1: // MaxPlus: reduce=max, identity=-inf
        case 2: // MaxMin: reduce=max, identity=-inf
            acc = __int_as_float(0xff800000); // -inf
            break;
        case 4: // OrAnd: reduce=OR, identity=0
        case 5: // PlusMax: reduce=+, identity=0
        default:
            acc = 0.0f;
            break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        float a_val = A[row * K + kk];
        float b_val = B[kk * N + col];

        float combined;
        switch (op) {
            case 0: // MinPlus: combine=+
            case 1: // MaxPlus: combine=+
                combined = a_val + b_val;
                break;
            case 2: // MaxMin: combine=min
                combined = fminf(a_val, b_val);
                break;
            case 3: // MinMax: combine=max
            case 5: // PlusMax: combine=max
                combined = fmaxf(a_val, b_val);
                break;
            case 4: // OrAnd: combine=AND
                combined = (a_val != 0.0f && b_val != 0.0f) ? 1.0f : 0.0f;
                break;
            default:
                combined = a_val + b_val;
                break;
        }

        // Reduce
        switch (op) {
            case 0: // MinPlus: reduce=min
            case 3: // MinMax: reduce=min
                acc = fminf(acc, combined);
                break;
            case 1: // MaxPlus: reduce=max
            case 2: // MaxMin: reduce=max
                acc = fmaxf(acc, combined);
                break;
            case 4: // OrAnd: reduce=OR
                if (combined != 0.0f) acc = 1.0f;
                break;
            case 5: // PlusMax: reduce=+
                acc = acc + combined;
                break;
            default:
                acc = fminf(acc, combined);
                break;
        }
    }

    C[row * N + col] = acc;
}

// Batched version
extern "C" __global__ void semiring_matmul_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op,
    unsigned int batch_size
) {
    unsigned int batch = blockIdx.z;
    if (batch >= batch_size) return;

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int a_offset = batch * M * K;
    unsigned int b_offset = batch * K * N;
    unsigned int c_offset = batch * M * N;

    float acc;
    switch (op) {
        case 0: case 3: acc = __int_as_float(0x7f800000); break;
        case 1: case 2: acc = __int_as_float(0xff800000); break;
        default: acc = 0.0f; break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        float a_val = A[a_offset + row * K + kk];
        float b_val = B[b_offset + kk * N + col];

        float combined;
        switch (op) {
            case 0: case 1: combined = a_val + b_val; break;
            case 2: combined = fminf(a_val, b_val); break;
            case 3: case 5: combined = fmaxf(a_val, b_val); break;
            case 4: combined = (a_val != 0.0f && b_val != 0.0f) ? 1.0f : 0.0f; break;
            default: combined = a_val + b_val; break;
        }

        switch (op) {
            case 0: case 3: acc = fminf(acc, combined); break;
            case 1: case 2: acc = fmaxf(acc, combined); break;
            case 4: if (combined != 0.0f) acc = 1.0f; break;
            case 5: acc = acc + combined; break;
            default: acc = fminf(acc, combined); break;
        }
    }

    C[c_offset + row * N + col] = acc;
}

// ============================================================================
// F64 Kernels
// ============================================================================

extern "C" __global__ void semiring_matmul_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double acc;
    switch (op) {
        case 0: case 3: acc = __longlong_as_double(0x7FF0000000000000LL); break; // +inf
        case 1: case 2: acc = __longlong_as_double(0xFFF0000000000000LL); break; // -inf
        default: acc = 0.0; break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        double a_val = A[row * K + kk];
        double b_val = B[kk * N + col];

        double combined;
        switch (op) {
            case 0: case 1: combined = a_val + b_val; break;
            case 2: combined = fmin(a_val, b_val); break;
            case 3: case 5: combined = fmax(a_val, b_val); break;
            case 4: combined = (a_val != 0.0 && b_val != 0.0) ? 1.0 : 0.0; break;
            default: combined = a_val + b_val; break;
        }

        switch (op) {
            case 0: case 3: acc = fmin(acc, combined); break;
            case 1: case 2: acc = fmax(acc, combined); break;
            case 4: if (combined != 0.0) acc = 1.0; break;
            case 5: acc = acc + combined; break;
            default: acc = fmin(acc, combined); break;
        }
    }

    C[row * N + col] = acc;
}

extern "C" __global__ void semiring_matmul_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op,
    unsigned int batch_size
) {
    unsigned int batch = blockIdx.z;
    if (batch >= batch_size) return;

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int a_offset = batch * M * K;
    unsigned int b_offset = batch * K * N;
    unsigned int c_offset = batch * M * N;

    double acc;
    switch (op) {
        case 0: case 3: acc = __longlong_as_double(0x7FF0000000000000LL); break;
        case 1: case 2: acc = __longlong_as_double(0xFFF0000000000000LL); break;
        default: acc = 0.0; break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        double a_val = A[a_offset + row * K + kk];
        double b_val = B[b_offset + kk * N + col];

        double combined;
        switch (op) {
            case 0: case 1: combined = a_val + b_val; break;
            case 2: combined = fmin(a_val, b_val); break;
            case 3: case 5: combined = fmax(a_val, b_val); break;
            case 4: combined = (a_val != 0.0 && b_val != 0.0) ? 1.0 : 0.0; break;
            default: combined = a_val + b_val; break;
        }

        switch (op) {
            case 0: case 3: acc = fmin(acc, combined); break;
            case 1: case 2: acc = fmax(acc, combined); break;
            case 4: if (combined != 0.0) acc = 1.0; break;
            case 5: acc = acc + combined; break;
            default: acc = fmin(acc, combined); break;
        }
    }

    C[c_offset + row * N + col] = acc;
}

// ============================================================================
// I32 Kernels
// ============================================================================

extern "C" __global__ void semiring_matmul_i32(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int acc;
    switch (op) {
        case 0: case 3: acc = 2147483647; break;  // INT_MAX
        case 1: case 2: acc = -2147483647 - 1; break;  // INT_MIN
        default: acc = 0; break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        int a_val = A[row * K + kk];
        int b_val = B[kk * N + col];

        int combined;
        switch (op) {
            case 0: case 1: combined = a_val + b_val; break;
            case 2: combined = min(a_val, b_val); break;
            case 3: case 5: combined = max(a_val, b_val); break;
            case 4: combined = (a_val != 0 && b_val != 0) ? 1 : 0; break;
            default: combined = a_val + b_val; break;
        }

        switch (op) {
            case 0: case 3: acc = min(acc, combined); break;
            case 1: case 2: acc = max(acc, combined); break;
            case 4: if (combined != 0) acc = 1; break;
            case 5: acc = acc + combined; break;
            default: acc = min(acc, combined); break;
        }
    }

    C[row * N + col] = acc;
}

extern "C" __global__ void semiring_matmul_batched_i32(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int op,
    unsigned int batch_size
) {
    unsigned int batch = blockIdx.z;
    if (batch >= batch_size) return;

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    unsigned int a_offset = batch * M * K;
    unsigned int b_offset = batch * K * N;
    unsigned int c_offset = batch * M * N;

    int acc;
    switch (op) {
        case 0: case 3: acc = 2147483647; break;
        case 1: case 2: acc = -2147483647 - 1; break;
        default: acc = 0; break;
    }

    for (unsigned int kk = 0; kk < K; kk++) {
        int a_val = A[a_offset + row * K + kk];
        int b_val = B[b_offset + kk * N + col];

        int combined;
        switch (op) {
            case 0: case 1: combined = a_val + b_val; break;
            case 2: combined = min(a_val, b_val); break;
            case 3: case 5: combined = max(a_val, b_val); break;
            case 4: combined = (a_val != 0 && b_val != 0) ? 1 : 0; break;
            default: combined = a_val + b_val; break;
        }

        switch (op) {
            case 0: case 3: acc = min(acc, combined); break;
            case 1: case 2: acc = max(acc, combined); break;
            case 4: if (combined != 0) acc = 1; break;
            case 5: acc = acc + combined; break;
            default: acc = min(acc, combined); break;
        }
    }

    C[c_offset + row * N + col] = acc;
}
