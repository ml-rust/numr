// Advanced linear algebra CUDA kernels
// Includes: rsf2csf (real to complex Schur form conversion)
//
// Uses C++ templates to eliminate f32/f64 duplication.
// QZ decomposition has been moved to linalg_qz.cu

#include "dtype_traits.cuh"

// ============================================================================
// RSF2CSF - Convert Real Schur Form to Complex Schur Form
// Processes 2x2 diagonal blocks representing complex conjugate eigenvalue pairs
// and transforms them to upper triangular form in complex space.
// ============================================================================

template<typename T>
__device__ void rsf2csf_impl(
    const T* __restrict__ z_in,     // [n, n] real Schur vectors
    const T* __restrict__ t_in,     // [n, n] real Schur form (quasi-triangular)
    T* __restrict__ z_real,         // [n, n] complex Schur vectors (real part)
    T* __restrict__ z_imag,         // [n, n] complex Schur vectors (imag part)
    T* __restrict__ t_real,         // [n, n] complex Schur form (real part)
    T* __restrict__ t_imag,         // [n, n] complex Schur form (imag part)
    unsigned int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const T eps = dtype_traits<T>::eps();

    // Initialize outputs: copy real parts, zero imaginary parts
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z_real[i * n + j] = z_in[i * n + j];
            z_imag[i * n + j] = dtype_traits<T>::zero();
            t_real[i * n + j] = t_in[i * n + j];
            t_imag[i * n + j] = dtype_traits<T>::zero();
        }
    }

    // Process 2x2 blocks on diagonal
    unsigned int i = 0;
    while (i < n) {
        if (i == n - 1) {
            // Last element is 1x1 block (real eigenvalue)
            i++;
            continue;
        }

        // Check if this is a 2x2 block (subdiagonal is non-zero)
        T subdiag = dtype_traits<T>::abs(t_real[(i + 1) * n + i]);
        T diag_scale = dtype_traits<T>::abs(t_real[i * n + i]) + dtype_traits<T>::abs(t_real[(i + 1) * n + (i + 1)]);
        T threshold = eps * dtype_traits<T>::max(diag_scale, dtype_traits<T>::one());

        if (subdiag <= threshold) {
            // 1x1 block (real eigenvalue)
            i++;
            continue;
        }

        // 2x2 block representing complex conjugate pair
        // Extract block elements
        T a = t_real[i * n + i];
        T b = t_real[i * n + (i + 1)];
        T c = t_real[(i + 1) * n + i];
        T d = t_real[(i + 1) * n + (i + 1)];

        // Compute eigenvalues: lambda = (a+d)/2 +/- sqrt((a-d)^2/4 + bc)
        T trace = a + d;
        T disc = (a - d) * (a - d) / (T)4 + b * c;

        if (disc >= dtype_traits<T>::zero()) {
            // Real eigenvalues (shouldn't happen for proper 2x2 block, but handle it)
            i += 2;
            continue;
        }

        T real_part = trace / dtype_traits<T>::two();
        T imag_part = dtype_traits<T>::sqrt(-disc);

        // Compute Givens rotation to transform 2x2 block to upper triangular
        // Using the eigenvector corresponding to lambda = real_part + i*imag_part
        // Eigenvector: [b, lambda - a] = [b, (d-a)/2 + i*imag_part]
        T v_real = b;
        T v_imag = dtype_traits<T>::zero();
        T w_real = (d - a) / dtype_traits<T>::two();
        T w_imag = imag_part;

        // Normalize the eigenvector
        T norm_sq = v_real * v_real + v_imag * v_imag + w_real * w_real + w_imag * w_imag;
        T norm = dtype_traits<T>::sqrt(norm_sq);
        if (norm > eps) {
            v_real /= norm;
            v_imag /= norm;
            w_real /= norm;
            w_imag /= norm;
        }

        // Build unitary transformation Q
        T q00_r = v_real, q00_i = v_imag;
        T q01_r = w_real, q01_i = -w_imag;
        T q10_r = w_real, q10_i = w_imag;
        T q11_r = -v_real, q11_i = v_imag;

        // Set the diagonal elements directly (eigenvalues)
        t_real[i * n + i] = real_part;
        t_imag[i * n + i] = imag_part;
        t_real[(i + 1) * n + (i + 1)] = real_part;
        t_imag[(i + 1) * n + (i + 1)] = -imag_part;

        // Zero out subdiagonal
        t_real[(i + 1) * n + i] = dtype_traits<T>::zero();
        t_imag[(i + 1) * n + i] = dtype_traits<T>::zero();

        // The (i, i+1) element
        t_real[i * n + (i + 1)] = b;
        t_imag[i * n + (i + 1)] = dtype_traits<T>::zero();

        // Update Z columns i and i+1: Z' = Z @ Q
        for (unsigned int row = 0; row < n; row++) {
            T z_ri_r = z_real[row * n + i];
            T z_ri_i = z_imag[row * n + i];
            T z_ri1_r = z_real[row * n + (i + 1)];
            T z_ri1_i = z_imag[row * n + (i + 1)];

            // Z'[row, i] = Z[row, i] * Q[i,i] + Z[row, i+1] * Q[i+1,i]
            T new_i_r = (z_ri_r * q00_r - z_ri_i * q00_i) + (z_ri1_r * q10_r - z_ri1_i * q10_i);
            T new_i_i = (z_ri_r * q00_i + z_ri_i * q00_r) + (z_ri1_r * q10_i + z_ri1_i * q10_r);

            // Z'[row, i+1] = Z[row, i] * Q[i,i+1] + Z[row, i+1] * Q[i+1,i+1]
            T new_i1_r = (z_ri_r * q01_r - z_ri_i * q01_i) + (z_ri1_r * q11_r - z_ri1_i * q11_i);
            T new_i1_i = (z_ri_r * q01_i + z_ri_i * q01_r) + (z_ri1_r * q11_i + z_ri1_i * q11_r);

            z_real[row * n + i] = new_i_r;
            z_imag[row * n + i] = new_i_i;
            z_real[row * n + (i + 1)] = new_i1_r;
            z_imag[row * n + (i + 1)] = new_i1_i;
        }

        // Update T rows 0 to i-1 for columns i and i+1 (right multiply by Q)
        for (unsigned int row = 0; row < i; row++) {
            T t_ri_r = t_real[row * n + i];
            T t_ri_i = t_imag[row * n + i];
            T t_ri1_r = t_real[row * n + (i + 1)];
            T t_ri1_i = t_imag[row * n + (i + 1)];

            T new_i_r = (t_ri_r * q00_r - t_ri_i * q00_i) + (t_ri1_r * q10_r - t_ri1_i * q10_i);
            T new_i_i = (t_ri_r * q00_i + t_ri_i * q00_r) + (t_ri1_r * q10_i + t_ri1_i * q10_r);

            T new_i1_r = (t_ri_r * q01_r - t_ri_i * q01_i) + (t_ri1_r * q11_r - t_ri1_i * q11_i);
            T new_i1_i = (t_ri_r * q01_i + t_ri_i * q01_r) + (t_ri1_r * q11_i + t_ri1_i * q11_r);

            t_real[row * n + i] = new_i_r;
            t_imag[row * n + i] = new_i_i;
            t_real[row * n + (i + 1)] = new_i1_r;
            t_imag[row * n + (i + 1)] = new_i1_i;
        }

        i += 2;
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void rsf2csf_f32(const float* z_in, const float* t_in, float* z_real, float* z_imag,
                            float* t_real, float* t_imag, unsigned int n) {
    rsf2csf_impl<float>(z_in, t_in, z_real, z_imag, t_real, t_imag, n);
}

__global__ void rsf2csf_f64(const double* z_in, const double* t_in, double* z_real, double* z_imag,
                            double* t_real, double* t_imag, unsigned int n) {
    rsf2csf_impl<double>(z_in, t_in, z_real, z_imag, t_real, t_imag, n);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void rsf2csf_f16(const __half* z_in, const __half* t_in, __half* z_real, __half* z_imag,
                            __half* t_real, __half* t_imag, unsigned int n) {
    rsf2csf_impl<__half>(z_in, t_in, z_real, z_imag, t_real, t_imag, n);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void rsf2csf_bf16(const __nv_bfloat16* z_in, const __nv_bfloat16* t_in,
                             __nv_bfloat16* z_real, __nv_bfloat16* z_imag,
                             __nv_bfloat16* t_real, __nv_bfloat16* t_imag, unsigned int n) {
    rsf2csf_impl<__nv_bfloat16>(z_in, t_in, z_real, z_imag, t_real, t_imag, n);
}

} // extern "C"
