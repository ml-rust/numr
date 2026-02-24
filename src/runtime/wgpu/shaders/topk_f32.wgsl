// Auto-generated topk operations for f32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SORT_SIZE: u32 = 512u;

var<workgroup> shared_vals: array<f32, 512>;
var<workgroup> shared_idxs: array<i32, 512>;

struct TopkParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    k: u32,
    largest: u32,
    sorted: u32,
}

@group(0) @binding(0) var<storage, read_write> topk_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> topk_indices: array<i32>;
@group(0) @binding(3) var<uniform> topk_params: TopkParams;

fn compare_less_f32(a: f32, b: f32) -> bool {
    return a < b;
}

fn bitonic_cas_f32(i: u32, j: u32, dir: bool) {
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    let swap = select(compare_less_f32(vi, vj), compare_less_f32(vj, vi), dir);
    if (swap) {
        shared_vals[i] = vj;
        shared_vals[j] = vi;
        let ti = shared_idxs[i];
        shared_idxs[i] = shared_idxs[j];
        shared_idxs[j] = ti;
    }
}

@compute @workgroup_size(256)
fn topk_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = topk_params.outer_size;
    let sort_size = topk_params.sort_size;
    let inner_size = topk_params.inner_size;
    let k = topk_params.k;
    let largest = topk_params.largest != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {
        p = p << 1u;
    }
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        if (i < sort_size) {
            let idx = base_offset + i * inner_size;
            shared_vals[i] = topk_input[idx];
            shared_idxs[i] = i32(i);
        } else {
            shared_vals[i] = select(f32(3.402823e+38), f32(-3.402823e+38), largest);
            shared_idxs[i] = i32(i);
        }
    }
    workgroupBarrier();

    // Bitonic sort (descending if largest, ascending if smallest)
    for (var k_: u32 = 2u; k_ <= n; k_ = k_ << 1u) {
        for (var j: u32 = k_ >> 1u; j > 0u; j = j >> 1u) {
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                // For largest: descending (true), for smallest: ascending (false)
                let ascending_local = ((ij / k_) % 2u == 0u) != largest;

                if (ij_pair < n) {
                    bitonic_cas_f32(ij, ij_pair, ascending_local);
                }
            }
            workgroupBarrier();
        }
    }

    // Write top-k values and indices
    let out_base = outer_idx * k * inner_size + inner_idx;
    for (var i = tid; i < k; i = i + WORKGROUP_SIZE) {
        let out_idx = out_base + i * inner_size;
        topk_values[out_idx] = shared_vals[i];
        topk_indices[out_idx] = shared_idxs[i];
    }
}
