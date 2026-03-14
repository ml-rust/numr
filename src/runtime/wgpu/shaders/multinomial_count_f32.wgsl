// Multinomial count shader for f32
// Performs CDF lookup for uniform samples and counts occurrences per category

const WORKGROUP_SIZE: u32 = 256u;

struct MultinomialCountParams {
    k: u32,           // Number of categories
    n_trials: u32,    // Number of trials per sample
    n_samples: u32,   // Number of samples
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> cdf: array<f32>;
@group(0) @binding(1) var<storage, read_write> uniforms: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<f32>;
@group(0) @binding(3) var<uniform> params: MultinomialCountParams;

// Binary search to find category for uniform sample
fn find_category(u: f32, k: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = k;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (cdf[mid] <= u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return min(lo, k - 1u);
}

@compute @workgroup_size(256)
fn multinomial_count_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    let k = params.k;
    let n_trials = params.n_trials;
    let n_samples = params.n_samples;

    if (sample_idx >= n_samples) {
        return;
    }

    // Initialize counts for this sample to zero
    for (var c: u32 = 0u; c < k; c++) {
        counts[sample_idx * k + c] = f32(0.0);
    }

    // Process each trial
    for (var t_idx: u32 = 0u; t_idx < n_trials; t_idx++) {
        let u = uniforms[sample_idx * n_trials + t_idx];
        let category = find_category(u, k);
        counts[sample_idx * k + category] += f32(1.0);
    }
}
