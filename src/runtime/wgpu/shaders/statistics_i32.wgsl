// Statistics shaders - I32
// mode_dim_i32: Find most frequent value in sorted data along reduce dimension

struct ModeParams {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> sorted: array<i32>;
@group(0) @binding(1) var<storage, read_write> mode_values: array<i32>;
@group(0) @binding(2) var<storage, read_write> mode_counts: array<i32>;
@group(0) @binding(3) var<uniform> params: ModeParams;

@compute @workgroup_size(1)
fn mode_dim_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    let total_outputs = params.outer_size * params.inner_size;

    if (out_idx >= total_outputs) {
        return;
    }

    let outer = out_idx / params.inner_size;
    let inner = out_idx % params.inner_size;
    let base = outer * params.reduce_size * params.inner_size + inner;

    if (params.reduce_size == 0u) {
        return;
    }

    // Initialize with first element
    var best_val = sorted[base];
    var best_count: i32 = 1;
    var curr_val = best_val;
    var curr_count: i32 = 1;

    // Scan through sorted slice
    for (var r: u32 = 1u; r < params.reduce_size; r = r + 1u) {
        let idx = base + r * params.inner_size;
        let val = sorted[idx];

        if (val == curr_val) {
            curr_count = curr_count + 1;
        } else {
            if (curr_count > best_count) {
                best_val = curr_val;
                best_count = curr_count;
            }
            curr_val = val;
            curr_count = 1;
        }
    }

    // Check final run
    if (curr_count > best_count) {
        best_val = curr_val;
        best_count = curr_count;
    }

    mode_values[out_idx] = best_val;
    mode_counts[out_idx] = best_count;
}
