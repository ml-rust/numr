// Distance computation shaders - F32
//
// cdist_f32:             Pairwise distances between two point sets
// pdist_f32:             Pairwise distances within one point set (condensed)
// squareform_f32:        Condensed to square distance matrix
// squareform_inverse_f32: Square to condensed distance matrix

const WORKGROUP_SIZE: u32 = 256u;

// Distance metric constants
const METRIC_EUCLIDEAN: u32 = 0u;
const METRIC_SQEUCLIDEAN: u32 = 1u;
const METRIC_MANHATTAN: u32 = 2u;
const METRIC_CHEBYSHEV: u32 = 3u;
const METRIC_MINKOWSKI: u32 = 4u;
const METRIC_COSINE: u32 = 5u;
const METRIC_CORRELATION: u32 = 6u;
const METRIC_HAMMING: u32 = 7u;
const METRIC_JACCARD: u32 = 8u;

// ============================================================================
// cdist_f32
// ============================================================================

struct CdistParams {
    n: u32,
    m: u32,
    d: u32,
    metric: u32,
    p: f32,
}

@group(0) @binding(0) var<storage, read_write> cdist_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cdist_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> cdist_out: array<f32>;
@group(0) @binding(3) var<uniform> cdist_params: CdistParams;

fn cdist_sqeuclidean(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let diff = cdist_x[x_offset + k] - cdist_y[y_offset + k];
        sum += diff * diff;
    }
    return sum;
}

fn cdist_manhattan(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += abs(cdist_x[x_offset + k] - cdist_y[y_offset + k]);
    }
    return sum;
}

fn cdist_chebyshev(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var max_val: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let abs_diff = abs(cdist_x[x_offset + k] - cdist_y[y_offset + k]);
        if (abs_diff > max_val) {
            max_val = abs_diff;
        }
    }
    return max_val;
}

fn cdist_minkowski(x_offset: u32, y_offset: u32, d: u32, p: f32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += pow(abs(cdist_x[x_offset + k] - cdist_y[y_offset + k]), p);
    }
    return pow(sum, 1.0 / p);
}

fn cdist_cosine(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let ak = cdist_x[x_offset + k];
        let bk = cdist_y[y_offset + k];
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

fn cdist_correlation(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum_a += cdist_x[x_offset + k];
        sum_b += cdist_y[y_offset + k];
    }
    let mean_a = sum_a / f32(d);
    let mean_b = sum_b / f32(d);

    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let da = cdist_x[x_offset + k] - mean_a;
        let db = cdist_y[y_offset + k] - mean_b;
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

fn cdist_hamming(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        if (cdist_x[x_offset + k] != cdist_y[y_offset + k]) {
            count += 1.0;
        }
    }
    return count / f32(d);
}

fn cdist_jaccard(x_offset: u32, y_offset: u32, d: u32) -> f32 {
    var intersection: f32 = 0.0;
    var union_count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let a_nonzero = cdist_x[x_offset + k] != 0.0;
        let b_nonzero = cdist_y[y_offset + k] != 0.0;
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

fn cdist_compute_distance(x_offset: u32, y_offset: u32, d: u32, metric: u32, p: f32) -> f32 {
    switch (metric) {
        case METRIC_EUCLIDEAN: {
            return sqrt(cdist_sqeuclidean(x_offset, y_offset, d));
        }
        case METRIC_SQEUCLIDEAN: {
            return cdist_sqeuclidean(x_offset, y_offset, d);
        }
        case METRIC_MANHATTAN: {
            return cdist_manhattan(x_offset, y_offset, d);
        }
        case METRIC_CHEBYSHEV: {
            return cdist_chebyshev(x_offset, y_offset, d);
        }
        case METRIC_MINKOWSKI: {
            return cdist_minkowski(x_offset, y_offset, d, p);
        }
        case METRIC_COSINE: {
            return cdist_cosine(x_offset, y_offset, d);
        }
        case METRIC_CORRELATION: {
            return cdist_correlation(x_offset, y_offset, d);
        }
        case METRIC_HAMMING: {
            return cdist_hamming(x_offset, y_offset, d);
        }
        case METRIC_JACCARD: {
            return cdist_jaccard(x_offset, y_offset, d);
        }
        default: {
            return 0.0;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cdist_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = cdist_params.n * cdist_params.m;
    if (idx >= total) {
        return;
    }

    let i = idx / cdist_params.m;
    let j = idx % cdist_params.m;

    let x_offset = i * cdist_params.d;
    let y_offset = j * cdist_params.d;

    let dist = cdist_compute_distance(x_offset, y_offset, cdist_params.d, cdist_params.metric, cdist_params.p);
    cdist_out[idx] = dist;
}

// ============================================================================
// pdist_f32
// ============================================================================

struct PdistParams {
    n: u32,
    d: u32,
    metric: u32,
    p: f32,
}

@group(0) @binding(0) var<storage, read_write> pdist_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pdist_out: array<f32>;
@group(0) @binding(2) var<uniform> pdist_params: PdistParams;

fn pdist_sqeuclidean(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let diff = pdist_x[i_offset + k] - pdist_x[j_offset + k];
        sum += diff * diff;
    }
    return sum;
}

fn pdist_manhattan(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += abs(pdist_x[i_offset + k] - pdist_x[j_offset + k]);
    }
    return sum;
}

fn pdist_chebyshev(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var max_val: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let abs_diff = abs(pdist_x[i_offset + k] - pdist_x[j_offset + k]);
        if (abs_diff > max_val) {
            max_val = abs_diff;
        }
    }
    return max_val;
}

fn pdist_minkowski(i_offset: u32, j_offset: u32, d: u32, p: f32) -> f32 {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum += pow(abs(pdist_x[i_offset + k] - pdist_x[j_offset + k]), p);
    }
    return pow(sum, 1.0 / p);
}

fn pdist_cosine(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let ak = pdist_x[i_offset + k];
        let bk = pdist_x[j_offset + k];
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

fn pdist_correlation(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        sum_a += pdist_x[i_offset + k];
        sum_b += pdist_x[j_offset + k];
    }
    let mean_a = sum_a / f32(d);
    let mean_b = sum_b / f32(d);

    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let da = pdist_x[i_offset + k] - mean_a;
        let db = pdist_x[j_offset + k] - mean_b;
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

fn pdist_hamming(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        if (pdist_x[i_offset + k] != pdist_x[j_offset + k]) {
            count += 1.0;
        }
    }
    return count / f32(d);
}

fn pdist_jaccard(i_offset: u32, j_offset: u32, d: u32) -> f32 {
    var intersection: f32 = 0.0;
    var union_count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {
        let a_nonzero = pdist_x[i_offset + k] != 0.0;
        let b_nonzero = pdist_x[j_offset + k] != 0.0;
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

fn pdist_compute_distance(i_offset: u32, j_offset: u32, d: u32, metric: u32, p: f32) -> f32 {
    switch (metric) {
        case METRIC_EUCLIDEAN: {
            return sqrt(pdist_sqeuclidean(i_offset, j_offset, d));
        }
        case METRIC_SQEUCLIDEAN: {
            return pdist_sqeuclidean(i_offset, j_offset, d);
        }
        case METRIC_MANHATTAN: {
            return pdist_manhattan(i_offset, j_offset, d);
        }
        case METRIC_CHEBYSHEV: {
            return pdist_chebyshev(i_offset, j_offset, d);
        }
        case METRIC_MINKOWSKI: {
            return pdist_minkowski(i_offset, j_offset, d, p);
        }
        case METRIC_COSINE: {
            return pdist_cosine(i_offset, j_offset, d);
        }
        case METRIC_CORRELATION: {
            return pdist_correlation(i_offset, j_offset, d);
        }
        case METRIC_HAMMING: {
            return pdist_hamming(i_offset, j_offset, d);
        }
        case METRIC_JACCARD: {
            return pdist_jaccard(i_offset, j_offset, d);
        }
        default: {
            return 0.0;
        }
    }
}

// Convert condensed index k to (i, j) where i < j
fn pdist_condensed_to_ij(k: u32, n: u32) -> vec2<u32> {
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
fn pdist_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let total = pdist_params.n * (pdist_params.n - 1u) / 2u;
    if (k >= total) {
        return;
    }

    let ij = pdist_condensed_to_ij(k, pdist_params.n);
    let i = ij.x;
    let j = ij.y;

    let i_offset = i * pdist_params.d;
    let j_offset = j * pdist_params.d;

    let dist = pdist_compute_distance(i_offset, j_offset, pdist_params.d, pdist_params.metric, pdist_params.p);
    pdist_out[k] = dist;
}

// ============================================================================
// squareform_f32
// ============================================================================

struct SquareformParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> sqf_condensed: array<f32>;
@group(0) @binding(1) var<storage, read_write> sqf_square: array<f32>;
@group(0) @binding(2) var<uniform> sqf_params: SquareformParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn squareform_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = sqf_params.n * sqf_params.n;
    if (idx >= total) {
        return;
    }

    let i = idx / sqf_params.n;
    let j = idx % sqf_params.n;

    if (i == j) {
        // Diagonal is zero
        sqf_square[idx] = 0.0;
    } else if (i < j) {
        // Upper triangle: k = n*i - i*(i+1)/2 + j - i - 1
        let k = sqf_params.n * i - i * (i + 1u) / 2u + j - i - 1u;
        sqf_square[idx] = sqf_condensed[k];
    } else {
        // Lower triangle: mirror from upper
        let k = sqf_params.n * j - j * (j + 1u) / 2u + i - j - 1u;
        sqf_square[idx] = sqf_condensed[k];
    }
}

// ============================================================================
// squareform_inverse_f32
// ============================================================================

struct SquareformInverseParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> sqfi_square: array<f32>;
@group(0) @binding(1) var<storage, read_write> sqfi_condensed: array<f32>;
@group(0) @binding(2) var<uniform> sqfi_params: SquareformInverseParams;

fn sqfi_condensed_to_ij(k: u32, n: u32) -> vec2<u32> {
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
    return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn squareform_inverse_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let total = sqfi_params.n * (sqfi_params.n - 1u) / 2u;
    if (k >= total) {
        return;
    }

    let ij = sqfi_condensed_to_ij(k, sqfi_params.n);
    let i = ij.x;
    let j = ij.y;

    sqfi_condensed[k] = sqfi_square[i * sqfi_params.n + j];
}
