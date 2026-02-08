//! Distance computation WGSL kernel launchers.
//!
//! Provides launchers for pairwise distance computation using various metrics.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::DistanceMetric;

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Convert DistanceMetric to shader index
pub fn metric_to_index(metric: DistanceMetric) -> u32 {
    match metric {
        DistanceMetric::Euclidean => 0,
        DistanceMetric::SquaredEuclidean => 1,
        DistanceMetric::Manhattan => 2,
        DistanceMetric::Chebyshev => 3,
        DistanceMetric::Minkowski(_) => 4,
        DistanceMetric::Cosine => 5,
        DistanceMetric::Correlation => 6,
        DistanceMetric::Hamming => 7,
        DistanceMetric::Jaccard => 8,
    }
}

/// Get Minkowski p value from metric
pub fn metric_p_value(metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Minkowski(p) => p as f32,
        _ => 2.0,
    }
}

/// Generate WGSL shader for cdist operation
fn generate_cdist_shader() -> String {
    format!(
        r#"
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

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

struct Params {{
    n: u32,
    m: u32,
    d: u32,
    metric: u32,
    p: f32,
}}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn sqeuclidean_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let diff = x[x_offset + k] - y[y_offset + k];
        sum += diff * diff;
    }}
    return sum;
}}

fn manhattan_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum += abs(x[x_offset + k] - y[y_offset + k]);
    }}
    return sum;
}}

fn chebyshev_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var max_val: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let abs_diff = abs(x[x_offset + k] - y[y_offset + k]);
        if (abs_diff > max_val) {{
            max_val = abs_diff;
        }}
    }}
    return max_val;
}}

fn minkowski_dist(x_offset: u32, y_offset: u32, d: u32, p: f32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum += pow(abs(x[x_offset + k] - y[y_offset + k]), p);
    }}
    return pow(sum, 1.0 / p);
}}

fn cosine_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let ak = x[x_offset + k];
        let bk = y[y_offset + k];
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }}
    let denom = sqrt(norm_a * norm_b);
    if (denom == 0.0) {{
        return 0.0;
    }}
    return 1.0 - dot / denom;
}}

fn correlation_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum_a += x[x_offset + k];
        sum_b += y[y_offset + k];
    }}
    let mean_a = sum_a / f32(d);
    let mean_b = sum_b / f32(d);

    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let da = x[x_offset + k] - mean_a;
        let db = y[y_offset + k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }}
    let denom = sqrt(var_a * var_b);
    if (denom == 0.0) {{
        return 0.0;
    }}
    return 1.0 - cov / denom;
}}

fn hamming_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        if (x[x_offset + k] != y[y_offset + k]) {{
            count += 1.0;
        }}
    }}
    return count / f32(d);
}}

fn jaccard_dist(x_offset: u32, y_offset: u32, d: u32) -> f32 {{
    var intersection: f32 = 0.0;
    var union_count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let a_nonzero = x[x_offset + k] != 0.0;
        let b_nonzero = y[y_offset + k] != 0.0;
        if (a_nonzero && b_nonzero) {{
            intersection += 1.0;
        }}
        if (a_nonzero || b_nonzero) {{
            union_count += 1.0;
        }}
    }}
    if (union_count == 0.0) {{
        return 0.0;
    }}
    return 1.0 - intersection / union_count;
}}

fn compute_distance(x_offset: u32, y_offset: u32, d: u32, metric: u32, p: f32) -> f32 {{
    switch (metric) {{
        case METRIC_EUCLIDEAN: {{
            return sqrt(sqeuclidean_dist(x_offset, y_offset, d));
        }}
        case METRIC_SQEUCLIDEAN: {{
            return sqeuclidean_dist(x_offset, y_offset, d);
        }}
        case METRIC_MANHATTAN: {{
            return manhattan_dist(x_offset, y_offset, d);
        }}
        case METRIC_CHEBYSHEV: {{
            return chebyshev_dist(x_offset, y_offset, d);
        }}
        case METRIC_MINKOWSKI: {{
            return minkowski_dist(x_offset, y_offset, d, p);
        }}
        case METRIC_COSINE: {{
            return cosine_dist(x_offset, y_offset, d);
        }}
        case METRIC_CORRELATION: {{
            return correlation_dist(x_offset, y_offset, d);
        }}
        case METRIC_HAMMING: {{
            return hamming_dist(x_offset, y_offset, d);
        }}
        case METRIC_JACCARD: {{
            return jaccard_dist(x_offset, y_offset, d);
        }}
        default: {{
            return 0.0;
        }}
    }}
}}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.n * params.m;
    if (idx >= total) {{
        return;
    }}

    let i = idx / params.m;
    let j = idx % params.m;

    let x_offset = i * params.d;
    let y_offset = j * params.d;

    let dist = compute_distance(x_offset, y_offset, params.d, params.metric, params.p);
    out[idx] = dist;
}}
"#,
        workgroup_size = WORKGROUP_SIZE
    )
}

/// Generate WGSL shader for pdist operation
fn generate_pdist_shader() -> String {
    format!(
        r#"
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

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

struct Params {{
    n: u32,
    d: u32,
    metric: u32,
    p: f32,
}}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sqeuclidean_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let diff = x[i_offset + k] - x[j_offset + k];
        sum += diff * diff;
    }}
    return sum;
}}

fn manhattan_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum += abs(x[i_offset + k] - x[j_offset + k]);
    }}
    return sum;
}}

fn chebyshev_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var max_val: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let abs_diff = abs(x[i_offset + k] - x[j_offset + k]);
        if (abs_diff > max_val) {{
            max_val = abs_diff;
        }}
    }}
    return max_val;
}}

fn minkowski_dist(i_offset: u32, j_offset: u32, d: u32, p: f32) -> f32 {{
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum += pow(abs(x[i_offset + k] - x[j_offset + k]), p);
    }}
    return pow(sum, 1.0 / p);
}}

fn cosine_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let ak = x[i_offset + k];
        let bk = x[j_offset + k];
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }}
    let denom = sqrt(norm_a * norm_b);
    if (denom == 0.0) {{
        return 0.0;
    }}
    return 1.0 - dot / denom;
}}

fn correlation_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        sum_a += x[i_offset + k];
        sum_b += x[j_offset + k];
    }}
    let mean_a = sum_a / f32(d);
    let mean_b = sum_b / f32(d);

    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let da = x[i_offset + k] - mean_a;
        let db = x[j_offset + k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }}
    let denom = sqrt(var_a * var_b);
    if (denom == 0.0) {{
        return 0.0;
    }}
    return 1.0 - cov / denom;
}}

fn hamming_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        if (x[i_offset + k] != x[j_offset + k]) {{
            count += 1.0;
        }}
    }}
    return count / f32(d);
}}

fn jaccard_dist(i_offset: u32, j_offset: u32, d: u32) -> f32 {{
    var intersection: f32 = 0.0;
    var union_count: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k++) {{
        let a_nonzero = x[i_offset + k] != 0.0;
        let b_nonzero = x[j_offset + k] != 0.0;
        if (a_nonzero && b_nonzero) {{
            intersection += 1.0;
        }}
        if (a_nonzero || b_nonzero) {{
            union_count += 1.0;
        }}
    }}
    if (union_count == 0.0) {{
        return 0.0;
    }}
    return 1.0 - intersection / union_count;
}}

fn compute_distance(i_offset: u32, j_offset: u32, d: u32, metric: u32, p: f32) -> f32 {{
    switch (metric) {{
        case METRIC_EUCLIDEAN: {{
            return sqrt(sqeuclidean_dist(i_offset, j_offset, d));
        }}
        case METRIC_SQEUCLIDEAN: {{
            return sqeuclidean_dist(i_offset, j_offset, d);
        }}
        case METRIC_MANHATTAN: {{
            return manhattan_dist(i_offset, j_offset, d);
        }}
        case METRIC_CHEBYSHEV: {{
            return chebyshev_dist(i_offset, j_offset, d);
        }}
        case METRIC_MINKOWSKI: {{
            return minkowski_dist(i_offset, j_offset, d, p);
        }}
        case METRIC_COSINE: {{
            return cosine_dist(i_offset, j_offset, d);
        }}
        case METRIC_CORRELATION: {{
            return correlation_dist(i_offset, j_offset, d);
        }}
        case METRIC_HAMMING: {{
            return hamming_dist(i_offset, j_offset, d);
        }}
        case METRIC_JACCARD: {{
            return jaccard_dist(i_offset, j_offset, d);
        }}
        default: {{
            return 0.0;
        }}
    }}
}}

// Convert condensed index k to (i, j) where i < j
fn condensed_to_ij(k: u32, n: u32) -> vec2<u32> {{
    var i: u32 = 0u;
    var count: u32 = 0u;
    loop {{
        let row_count = n - 1u - i;
        if (count + row_count > k) {{
            let j = k - count + i + 1u;
            return vec2<u32>(i, j);
        }}
        count += row_count;
        i++;
    }}
    return vec2<u32>(0u, 0u); // Should never reach
}}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let k = gid.x;
    let total = params.n * (params.n - 1u) / 2u;
    if (k >= total) {{
        return;
    }}

    let ij = condensed_to_ij(k, params.n);
    let i = ij.x;
    let j = ij.y;

    let i_offset = i * params.d;
    let j_offset = j * params.d;

    let dist = compute_distance(i_offset, j_offset, params.d, params.metric, params.p);
    out[k] = dist;
}}
"#,
        workgroup_size = WORKGROUP_SIZE
    )
}

/// Generate WGSL shader for squareform operation
fn generate_squareform_shader() -> String {
    format!(
        r#"
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

struct Params {{
    n: u32,
}}

@group(0) @binding(0) var<storage, read_write> condensed: array<f32>;
@group(0) @binding(1) var<storage, read_write> square: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.n * params.n;
    if (idx >= total) {{
        return;
    }}

    let i = idx / params.n;
    let j = idx % params.n;

    if (i == j) {{
        // Diagonal is zero
        square[idx] = 0.0;
    }} else if (i < j) {{
        // Upper triangle: k = n*i - i*(i+1)/2 + j - i - 1
        let k = params.n * i - i * (i + 1u) / 2u + j - i - 1u;
        square[idx] = condensed[k];
    }} else {{
        // Lower triangle: mirror from upper
        let k = params.n * j - j * (j + 1u) / 2u + i - j - 1u;
        square[idx] = condensed[k];
    }}
}}
"#,
        workgroup_size = WORKGROUP_SIZE
    )
}

/// Generate WGSL shader for squareform_inverse operation
fn generate_squareform_inverse_shader() -> String {
    format!(
        r#"
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

struct Params {{
    n: u32,
}}

@group(0) @binding(0) var<storage, read_write> square: array<f32>;
@group(0) @binding(1) var<storage, read_write> condensed: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Convert condensed index k to (i, j) where i < j
fn condensed_to_ij(k: u32, n: u32) -> vec2<u32> {{
    var i: u32 = 0u;
    var count: u32 = 0u;
    loop {{
        let row_count = n - 1u - i;
        if (count + row_count > k) {{
            let j = k - count + i + 1u;
            return vec2<u32>(i, j);
        }}
        count += row_count;
        i++;
    }}
    return vec2<u32>(0u, 0u);
}}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let k = gid.x;
    let total = params.n * (params.n - 1u) / 2u;
    if (k >= total) {{
        return;
    }}

    let ij = condensed_to_ij(k, params.n);
    let i = ij.x;
    let j = ij.y;

    condensed[k] = square[i * params.n + j];
}}
"#,
        workgroup_size = WORKGROUP_SIZE
    )
}

/// Launch cdist kernel - pairwise distances between two point sets.
pub fn launch_cdist(
    cache: &PipelineCache,
    queue: &Queue,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "cdist")?;

    let name = "cdist_f32";
    let shader = generate_cdist_shader();
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[x, y, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cdist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cdist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch pdist kernel - pairwise distances within one point set (condensed).
pub fn launch_pdist(
    cache: &PipelineCache,
    queue: &Queue,
    x: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "pdist")?;

    let name = "pdist_f32";
    let shader = generate_pdist_shader();
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[x, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pdist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pdist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch squareform kernel - condensed to square.
pub fn launch_squareform(
    cache: &PipelineCache,
    queue: &Queue,
    condensed: &Buffer,
    square: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "squareform")?;

    let name = "squareform_f32";
    let shader = generate_squareform_shader();
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[condensed, square, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("squareform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("squareform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch squareform_inverse kernel - square to condensed.
pub fn launch_squareform_inverse(
    cache: &PipelineCache,
    queue: &Queue,
    square: &Buffer,
    condensed: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "squareform_inverse")?;

    let name = "squareform_inverse_f32";
    let shader = generate_squareform_inverse_shader();
    let module = cache.get_or_create_module(name, &shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[square, condensed, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("squareform_inverse"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("squareform_inverse"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// Export metric utilities for use by the ops implementation
pub use metric_p_value as distance_metric_p_value;
pub use metric_to_index as distance_metric_to_index;
