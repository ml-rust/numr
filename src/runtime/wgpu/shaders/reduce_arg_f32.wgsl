// Argmax / argmin. F32 only.
// Entry points: argmax_f32, argmin_f32
// Uses WORKGROUP_SIZE from reduce_f32.wgsl.

// ============================================================================
// Argmax / Argmin (returns index of max/min value)
// ============================================================================

var<workgroup> argmax_shared_val: array<f32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> argreduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> argreduce_output: array<u32>;
@group(0) @binding(2) var<uniform> argreduce_params: ArgReduceParams;

// Argmax and argmin follow the CPU rule in `cpu/kernels/reduce/special.rs`:
// - Equal values resolve to the lowest index. Lane `tid` scans tid, tid + 256, ...
//   so every merge compares the index on a tie.
// - NaN never wins. A row of only NaN returns index 0.
// WGSL leaves NaN comparisons to the implementation, so the scan tests NaN bits.

// Index of a lane that holds no candidate yet. Every real index is lower.
const ARG_NO_INDEX: u32 = 0xffffffffu;

fn arg_is_nan(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// True when candidate (v, i) replaces the current best (cur, cur_i).
fn arg_takes(is_max: bool, v: f32, i: u32, cur: f32, cur_i: u32) -> bool {
    let wins = select((v < cur), (v > cur), is_max);
    return wins || (v == cur && i < cur_i);
}

fn arg_reduce_f32(is_max: bool, tid: u32, output_idx: u32) {
    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    // Seed with the identity: -inf for max, +inf for min.
    var best: f32 = select(bitcast<f32>(0x7f800000u), bitcast<f32>(0xff800000u), is_max);
    var best_idx: u32 = ARG_NO_INDEX;
    var i: u32 = tid;

    while (i < reduce_size) {
        let val = argreduce_input[base_offset + i * inner_size];
        if (!arg_is_nan(val) && arg_takes(is_max, val, i, best, best_idx)) {
            best = val;
            best_idx = i;
        }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = best;
    argmax_shared_idx[tid] = best_idx;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s && arg_takes(is_max, argmax_shared_val[tid + s], argmax_shared_idx[tid + s],
                                 argmax_shared_val[tid], argmax_shared_idx[tid])) {
            argmax_shared_val[tid] = argmax_shared_val[tid + s];
            argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let found = argmax_shared_idx[0];
        argreduce_output[output_idx] = select(found, 0u, found == ARG_NO_INDEX);
    }
}

@compute @workgroup_size(256)
fn argmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    if (group_id.x >= argreduce_params.numel_out) {
        return;
    }
    arg_reduce_f32(true, local_id.x, group_id.x);
}

@compute @workgroup_size(256)
fn argmin_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    if (group_id.x >= argreduce_params.numel_out) {
        return;
    }
    arg_reduce_f32(false, local_id.x, group_id.x);
}

