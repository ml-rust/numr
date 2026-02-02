// Distribution sampling CUDA kernels
// Supports: bernoulli, beta, gamma, exponential, poisson, binomial, laplace, chi_squared, student_t, f_distribution
// Types: f32, f64, f16, bf16

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

// ============================================================================
// Random Number Generation - Device Functions
// ============================================================================

// xorshift128+ state per thread
struct XorShift128PlusState {
    unsigned long long s0;
    unsigned long long s1;
};

// Initialize state from seed and thread index
__device__ __forceinline__ void xorshift128plus_init(XorShift128PlusState* state, unsigned long long seed, unsigned int idx) {
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s0 = z ^ (z >> 31);

    z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s1 = z ^ (z >> 31);

    if (state->s0 == 0) state->s0 = 1;
    if (state->s1 == 0) state->s1 = 1;
}

__device__ __forceinline__ unsigned long long xorshift128plus_next(XorShift128PlusState* state) {
    unsigned long long s1 = state->s0;
    unsigned long long s0 = state->s1;
    unsigned long long result = s0 + s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

__device__ __forceinline__ double xorshift128plus_uniform(XorShift128PlusState* state) {
    return (double)(xorshift128plus_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

// Box-Muller for normal distribution
__device__ __forceinline__ void box_muller(XorShift128PlusState* state, double* z0, double* z1) {
    double u1 = xorshift128plus_uniform(state);
    double u2 = xorshift128plus_uniform(state);
    if (u1 < 1e-15) u1 = 1e-15;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

// ============================================================================
// Gamma Distribution Sampling - Marsaglia and Tsang's Method
// ============================================================================

__device__ __forceinline__ double sample_gamma(XorShift128PlusState* state, double shape, double scale) {
    // For shape < 1, use transformation: Gamma(a) = Gamma(a+1) * U^(1/a)
    double alpha = shape;
    double boost = 1.0;
    if (alpha < 1.0) {
        boost = pow(xorshift128plus_uniform(state), 1.0 / alpha);
        alpha += 1.0;
    }

    // Marsaglia and Tsang's method for alpha >= 1
    double d = alpha - 1.0/3.0;
    double c = 1.0 / sqrt(9.0 * d);

    while (true) {
        double x, v;
        do {
            double z0, z1;
            box_muller(state, &z0, &z1);
            x = z0;
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        double u = xorshift128plus_uniform(state);

        // Squeeze acceptance
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v * scale * boost;
        }
        // Full acceptance check
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v * scale * boost;
        }
    }
}

// ============================================================================
// Bernoulli Distribution
// ============================================================================

extern "C" {

__global__ void bernoulli_f32(float* out, double p, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (xorshift128plus_uniform(&state) < p) ? 1.0f : 0.0f;
    }
}

__global__ void bernoulli_f64(double* out, double p, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (xorshift128plus_uniform(&state) < p) ? 1.0 : 0.0;
    }
}

__global__ void bernoulli_f16(__half* out, double p, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2half((xorshift128plus_uniform(&state) < p) ? 1.0f : 0.0f);
    }
}

__global__ void bernoulli_bf16(__nv_bfloat16* out, double p, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2bfloat16((xorshift128plus_uniform(&state) < p) ? 1.0f : 0.0f);
    }
}

// ============================================================================
// Beta Distribution - Using Gamma relationship: Beta(a,b) = Ga/(Ga+Gb)
// ============================================================================

__global__ void beta_f32(float* out, double alpha, double beta, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double ga = sample_gamma(&state, alpha, 1.0);
        double gb = sample_gamma(&state, beta, 1.0);
        out[idx] = (float)(ga / (ga + gb));
    }
}

__global__ void beta_f64(double* out, double alpha, double beta, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double ga = sample_gamma(&state, alpha, 1.0);
        double gb = sample_gamma(&state, beta, 1.0);
        out[idx] = ga / (ga + gb);
    }
}

__global__ void beta_f16(__half* out, double alpha, double beta, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double ga = sample_gamma(&state, alpha, 1.0);
        double gb = sample_gamma(&state, beta, 1.0);
        out[idx] = __float2half((float)(ga / (ga + gb)));
    }
}

__global__ void beta_bf16(__nv_bfloat16* out, double alpha, double beta, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double ga = sample_gamma(&state, alpha, 1.0);
        double gb = sample_gamma(&state, beta, 1.0);
        out[idx] = __float2bfloat16((float)(ga / (ga + gb)));
    }
}

// ============================================================================
// Gamma Distribution
// ============================================================================

__global__ void gamma_f32(float* out, double shape, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)sample_gamma(&state, shape, scale);
    }
}

__global__ void gamma_f64(double* out, double shape, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = sample_gamma(&state, shape, scale);
    }
}

__global__ void gamma_f16(__half* out, double shape, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2half((float)sample_gamma(&state, shape, scale));
    }
}

__global__ void gamma_bf16(__nv_bfloat16* out, double shape, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2bfloat16((float)sample_gamma(&state, shape, scale));
    }
}

// ============================================================================
// Exponential Distribution - Inverse transform: -ln(U)/rate
// ============================================================================

__global__ void exponential_f32(float* out, double rate, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state);
        if (u < 1e-15) u = 1e-15;
        out[idx] = (float)(-log(u) / rate);
    }
}

__global__ void exponential_f64(double* out, double rate, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state);
        if (u < 1e-15) u = 1e-15;
        out[idx] = -log(u) / rate;
    }
}

__global__ void exponential_f16(__half* out, double rate, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state);
        if (u < 1e-15) u = 1e-15;
        out[idx] = __float2half((float)(-log(u) / rate));
    }
}

__global__ void exponential_bf16(__nv_bfloat16* out, double rate, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state);
        if (u < 1e-15) u = 1e-15;
        out[idx] = __float2bfloat16((float)(-log(u) / rate));
    }
}

// ============================================================================
// Poisson Distribution - Knuth algorithm for small lambda, normal approx for large
// ============================================================================

__device__ __forceinline__ double sample_poisson(XorShift128PlusState* state, double lambda) {
    if (lambda < 30.0) {
        // Knuth algorithm for small lambda
        double L = exp(-lambda);
        double k = 0.0;
        double p = 1.0;
        do {
            k += 1.0;
            p *= xorshift128plus_uniform(state);
        } while (p > L);
        return k - 1.0;
    } else {
        // Normal approximation for large lambda
        double z0, z1;
        box_muller(state, &z0, &z1);
        double result = round(lambda + sqrt(lambda) * z0);
        return (result < 0.0) ? 0.0 : result;
    }
}

__global__ void poisson_f32(float* out, double lambda, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)sample_poisson(&state, lambda);
    }
}

__global__ void poisson_f64(double* out, double lambda, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = sample_poisson(&state, lambda);
    }
}

__global__ void poisson_f16(__half* out, double lambda, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2half((float)sample_poisson(&state, lambda));
    }
}

__global__ void poisson_bf16(__nv_bfloat16* out, double lambda, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2bfloat16((float)sample_poisson(&state, lambda));
    }
}

// ============================================================================
// Binomial Distribution - Direct simulation for small n, normal approx for large
// ============================================================================

__device__ __forceinline__ double sample_binomial(XorShift128PlusState* state, unsigned long long n, double p) {
    if (n < 25) {
        // Direct simulation for small n
        double successes = 0.0;
        for (unsigned long long i = 0; i < n; i++) {
            if (xorshift128plus_uniform(state) < p) {
                successes += 1.0;
            }
        }
        return successes;
    } else {
        // Normal approximation for large n
        double mean = (double)n * p;
        double stddev = sqrt(mean * (1.0 - p));
        double z0, z1;
        box_muller(state, &z0, &z1);
        double result = round(mean + stddev * z0);
        if (result < 0.0) result = 0.0;
        if (result > (double)n) result = (double)n;
        return result;
    }
}

__global__ void binomial_f32(float* out, unsigned long long n, double p, unsigned long long seed, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)sample_binomial(&state, n, p);
    }
}

__global__ void binomial_f64(double* out, unsigned long long n, double p, unsigned long long seed, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = sample_binomial(&state, n, p);
    }
}

__global__ void binomial_f16(__half* out, unsigned long long n, double p, unsigned long long seed, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2half((float)sample_binomial(&state, n, p));
    }
}

__global__ void binomial_bf16(__nv_bfloat16* out, unsigned long long n, double p, unsigned long long seed, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2bfloat16((float)sample_binomial(&state, n, p));
    }
}

// ============================================================================
// Laplace Distribution - Inverse transform sampling
// ============================================================================

__global__ void laplace_f32(float* out, double loc, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state) - 0.5;
        double abs_u = fabs(u);
        if (abs_u < 1e-15) abs_u = 1e-15;
        double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        out[idx] = (float)(loc - scale * sign_u * log(1.0 - 2.0 * abs_u));
    }
}

__global__ void laplace_f64(double* out, double loc, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state) - 0.5;
        double abs_u = fabs(u);
        if (abs_u < 1e-15) abs_u = 1e-15;
        double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        out[idx] = loc - scale * sign_u * log(1.0 - 2.0 * abs_u);
    }
}

__global__ void laplace_f16(__half* out, double loc, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state) - 0.5;
        double abs_u = fabs(u);
        if (abs_u < 1e-15) abs_u = 1e-15;
        double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        out[idx] = __float2half((float)(loc - scale * sign_u * log(1.0 - 2.0 * abs_u)));
    }
}

__global__ void laplace_bf16(__nv_bfloat16* out, double loc, double scale, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double u = xorshift128plus_uniform(&state) - 0.5;
        double abs_u = fabs(u);
        if (abs_u < 1e-15) abs_u = 1e-15;
        double sign_u = (u >= 0.0) ? 1.0 : -1.0;
        out[idx] = __float2bfloat16((float)(loc - scale * sign_u * log(1.0 - 2.0 * abs_u)));
    }
}

// ============================================================================
// Chi-Squared Distribution - Chi2(df) = Gamma(df/2, 2)
// ============================================================================

__global__ void chi_squared_f32(float* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)sample_gamma(&state, df / 2.0, 2.0);
    }
}

__global__ void chi_squared_f64(double* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = sample_gamma(&state, df / 2.0, 2.0);
    }
}

__global__ void chi_squared_f16(__half* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2half((float)sample_gamma(&state, df / 2.0, 2.0));
    }
}

__global__ void chi_squared_bf16(__nv_bfloat16* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = __float2bfloat16((float)sample_gamma(&state, df / 2.0, 2.0));
    }
}

// ============================================================================
// Student's t Distribution - T = Z / sqrt(V/df) where Z ~ N(0,1), V ~ Chi2(df)
// ============================================================================

__global__ void student_t_f32(float* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double z0, z1;
        box_muller(&state, &z0, &z1);
        double v = sample_gamma(&state, df / 2.0, 2.0);
        out[idx] = (float)(z0 / sqrt(v / df));
    }
}

__global__ void student_t_f64(double* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double z0, z1;
        box_muller(&state, &z0, &z1);
        double v = sample_gamma(&state, df / 2.0, 2.0);
        out[idx] = z0 / sqrt(v / df);
    }
}

__global__ void student_t_f16(__half* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double z0, z1;
        box_muller(&state, &z0, &z1);
        double v = sample_gamma(&state, df / 2.0, 2.0);
        out[idx] = __float2half((float)(z0 / sqrt(v / df)));
    }
}

__global__ void student_t_bf16(__nv_bfloat16* out, double df, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double z0, z1;
        box_muller(&state, &z0, &z1);
        double v = sample_gamma(&state, df / 2.0, 2.0);
        out[idx] = __float2bfloat16((float)(z0 / sqrt(v / df)));
    }
}

// ============================================================================
// F Distribution - F = (X1/df1) / (X2/df2) where X1 ~ Chi2(df1), X2 ~ Chi2(df2)
// ============================================================================

__global__ void f_distribution_f32(float* out, double df1, double df2, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double x1 = sample_gamma(&state, df1 / 2.0, 2.0);
        double x2 = sample_gamma(&state, df2 / 2.0, 2.0);
        out[idx] = (float)((x1 / df1) / (x2 / df2));
    }
}

__global__ void f_distribution_f64(double* out, double df1, double df2, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double x1 = sample_gamma(&state, df1 / 2.0, 2.0);
        double x2 = sample_gamma(&state, df2 / 2.0, 2.0);
        out[idx] = (x1 / df1) / (x2 / df2);
    }
}

__global__ void f_distribution_f16(__half* out, double df1, double df2, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double x1 = sample_gamma(&state, df1 / 2.0, 2.0);
        double x2 = sample_gamma(&state, df2 / 2.0, 2.0);
        out[idx] = __float2half((float)((x1 / df1) / (x2 / df2)));
    }
}

__global__ void f_distribution_bf16(__nv_bfloat16* out, double df1, double df2, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        double x1 = sample_gamma(&state, df1 / 2.0, 2.0);
        double x2 = sample_gamma(&state, df2 / 2.0, 2.0);
        out[idx] = __float2bfloat16((float)((x1 / df1) / (x2 / df2)));
    }
}

} // extern "C"

// ============================================================================
// Multinomial Count Kernel - CDF lookup and counting for multinomial sampling
// ============================================================================

// Binary search to find category for a uniform sample
// Returns the index i where cdf[i-1] < u <= cdf[i]
template<typename T>
__device__ __forceinline__ unsigned int binary_search_cdf(const T* cdf, unsigned int k, T u) {
    unsigned int lo = 0;
    unsigned int hi = k;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (cdf[mid] <= u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return min(lo, k - 1);
}

// Each block processes one sample, threads cooperate on trials
// Grid: n_samples blocks
// Block: min(n_trials, 256) threads
extern "C" __global__ void multinomial_count_f32(
    const float* __restrict__ cdf,       // [k] - cumulative distribution function
    const float* __restrict__ uniforms,  // [n_samples, n_trials] - uniform random samples
    float* __restrict__ counts,          // [n_samples, k] - output counts
    unsigned int k,
    unsigned int n_trials
) {
    unsigned int sample_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // Each thread has its own local counts in shared memory
    extern __shared__ unsigned int shared_counts[];

    // Initialize shared memory counts to zero
    for (unsigned int c = tid; c < k; c += block_size) {
        shared_counts[c] = 0;
    }
    __syncthreads();

    // Each thread processes multiple trials
    for (unsigned int t = tid; t < n_trials; t += block_size) {
        float u = uniforms[sample_idx * n_trials + t];
        unsigned int category = binary_search_cdf(cdf, k, u);
        atomicAdd(&shared_counts[category], 1);
    }
    __syncthreads();

    // Write results to global memory
    for (unsigned int c = tid; c < k; c += block_size) {
        counts[sample_idx * k + c] = (float)shared_counts[c];
    }
}

extern "C" __global__ void multinomial_count_f64(
    const double* __restrict__ cdf,
    const double* __restrict__ uniforms,
    double* __restrict__ counts,
    unsigned int k,
    unsigned int n_trials
) {
    unsigned int sample_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ unsigned int shared_counts[];

    for (unsigned int c = tid; c < k; c += block_size) {
        shared_counts[c] = 0;
    }
    __syncthreads();

    for (unsigned int t = tid; t < n_trials; t += block_size) {
        double u = uniforms[sample_idx * n_trials + t];
        unsigned int category = binary_search_cdf(cdf, k, u);
        atomicAdd(&shared_counts[category], 1);
    }
    __syncthreads();

    for (unsigned int c = tid; c < k; c += block_size) {
        counts[sample_idx * k + c] = (double)shared_counts[c];
    }
}
