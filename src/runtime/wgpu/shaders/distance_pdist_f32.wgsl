const WORKGROUP_SIZE: u32 = 256u;

// Distance metric constants (same as cdist)
const METRIC_EUCLIDEAN: u32 = 0u;
const METRIC_SQEUCLIDEAN: u32 = 1u;
const METRIC_MANHATTAN: u32 = 2u;
const METRIC_CHEBYSHEV: u32 = 3u;
const METRIC_MINKOWSKI: u32 = 4u;
const METRIC_COSINE: u32 = 5u;
const METRIC_CORRELATION: u32 = 6u;
const METRIC_HAMMING: u32 = 7u;
const METRIC_JACCARD: u32 = 8u;

struct Params {
    n: u32,
    d: u32,
    metric: u32,
    p: f32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sqeuclidean_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let diff = x[i_offset + k] - x[j_offset + k];
        sum += diff * diff;
    }
    return sum;
}

fn manhattan_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += abs(x[i_offset + k] - x[j_offset + k]);
    }
    return sum;
}

fn chebyshev_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var max_val: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let abs_diff = abs(x[i_offset + k] - x[j_offset + k]);
        if (abs_diff > max_val) {
            max_val = abs_diff;
        }
    }
    return max_val;
}

fn minkowski_dist(i_offset: u32, j_offset: u32, d: u32, p: f32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += pow(abs(x[i_offset + k] - x[j_offset + k]), p);
    }
    return pow(sum, 1.0 / p);
}

fn cosine_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let ak = x[i_offset + k];
        let bk = x[j_offset + k];
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }
    let denom = sqrt(norm_a * norm_b);
    if (denom == 0.0) {
        return 0.0;
    }
    return 1.0 - dot / denom;
}

fn correlation_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum_a += x[i_offset + k];
        sum_b += x[j_offset + k];
    }
    let mean_a = sum_a / f32(d);
    let mean_b = sum_b / f32(d);

    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let da = x[i_offset + k] - mean_a;
        let db = x[j_offset + k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = sqrt(var_a * var_b);
    if (denom == 0.0) {
        return 0.0;
    }
    return 1.0 - cov / denom;
}

fn hamming_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        if (x[i_offset + k] != x[j_offset + k]) {
            count += 1.0;
        }
    }
    return count / f32(d);
}

fn jaccard_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var intersection: f32 = 0.0;
    var union_count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let a_nonzero = x[i_offset + k] != 0.0;
        let b_nonzero = x[j_offset + k] != 0.0;
        if (a_nonzero && b_nonzero) {
            intersection += 1.0;
        }
        if (a_nonzero || b_nonzero) {
            union_count += 1.0;
        }
    }
    if (union_count == 0.0) {
        return 0.0;
    }
    return 1.0 - intersection / union_count;
}

fn compute_distance(i_offset: u32, j_offset: u32, d: u32, metric: u32, p: f32) -> f32 {
    switch (metric) {
        case METRIC_EUCLIDEAN: {
            return sqrt(sqeuclidean_dist(i_offset, j_offset, d));
        }
        case METRIC_SQEUCLIDEAN: {
            return sqeuclidean_dist(i_offset, j_offset, d);
        }
        case METRIC_MANHATTAN: {
            return manhattan_dist(i_offset, j_offset, d);
        }
        case METRIC_CHEBYSHEV: {
            return chebyshev_dist(i_offset, j_offset, d);
        }
        case METRIC_MINKOWSKI: {
            return minkowski_dist(i_offset, j_offset, d, p);
        }
        case METRIC_COSINE: {
            return cosine_dist(i_offset, j_offset, d);
        }
        case METRIC_CORRELATION: {
            return correlation_dist(i_offset, j_offset, d);
        }
        case METRIC_HAMMING: {
            return hamming_dist(i_offset, j_offset, d);
        }
        case METRIC_JACCARD: {
            return jaccard_dist(i_offset, j_offset, d);
        }
        default: {
            return 0.0;
        }
    }
}

// Convert condensed index k to (i, j) where i < j
fn condensed_to_ij(k: u32, n: u32) -> vec2<u32> {
    var i: u32 = 0u;
    var count: u32 = 0u;
    loop {
        let row_count = n - 1u - i;
        if (count + row_count > k) {
            let j = k - count + i + 1u;
            return vec2<u32>(i, j);
        }
        count += row_count;
        i++;
    }
    return vec2<u32>(0u, 0u); // Should never reach
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let total = params.n * (params.n - 1u) / 2u;
    if (k >= total) {
        return;
    }

    let ij = condensed_to_ij(k, params.n);
    let i = ij.x;
    let j = ij.y;

    let i_offset = i * params.d;
    let j_offset = j * params.d;

    let dist = compute_distance(i_offset, j_offset, params.d, params.metric, params.p);
    out[k] = dist;
}
