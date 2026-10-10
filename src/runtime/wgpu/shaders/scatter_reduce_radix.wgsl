// Stable LSD radix sort of scatter_reduce's keys, with source positions as
// values. After it, each destination's contributions sit in one run of the
// sorted keys, in increasing source position.
//
// One pass per 8-bit digit of the key, each three entry points:
// scatter_reduce_radix_hist counts each tile's digits,
// scatter_reduce_radix_scan turns the counts into each tile's exclusive offset
// within its digit and each digit's total, and scatter_reduce_radix_scatter
// moves every entry to its sorted slot.
//
// WGSL has no subgroup ballot, so the scatter ranks entries inside a tile with
// workgroup bit masks: each round of 256 entries sets bit `tid` of its digit's
// 256-bit mask, and an entry's rank is the population count of the bits below
// its own. Rounds walk the tile in source order, so every pass is stable and the
// final order within one key is increasing source position.
//
// Every count and mask update is an integer atomic, so the result does not
// depend on the order the atomics land in.
//
// Geometry: 256 invocations per workgroup, SR_ITEMS rounds per tile. The
// launcher in index/scatter_reduce_sort.rs mirrors SR_TILE.

const SR_THREADS: u32 = 256u;
const SR_RADIX: u32 = 256u;
const SR_ITEMS: u32 = 16u;
const SR_TILE: u32 = 4096u;
// 32-bit words in one digit's 256-bit mask.
const SR_MASK_WORDS: u32 = 8u;

struct SrRadixParams {
    n: u32,
    shift: u32,
    tiles: u32,
    _pad: u32,
}

var<workgroup> sr_counts: array<atomic<u32>, 256>;
var<workgroup> sr_scan_buf: array<u32, 256>;
var<workgroup> sr_chunk_total: u32;
// Digit-major: digit `d` owns words [d * 8, d * 8 + 8).
var<workgroup> sr_masks: array<atomic<u32>, 2048>;
var<workgroup> sr_offsets: array<u32, 256>;

fn sr_tile_id(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.x + wid.y * nwg.x;
}

// Inclusive Hillis-Steele scan of 256 values held one per invocation. Every
// invocation of the workgroup calls it.
fn sr_block_inclusive_scan(tid: u32, v: u32) -> u32 {
    sr_scan_buf[tid] = v;
    workgroupBarrier();
    for (var off = 1u; off < SR_THREADS; off = off << 1u) {
        var x = 0u;
        if (tid >= off) {
            x = sr_scan_buf[tid - off];
        }
        workgroupBarrier();
        sr_scan_buf[tid] = sr_scan_buf[tid] + x;
        workgroupBarrier();
    }
    let out = sr_scan_buf[tid];
    workgroupBarrier();
    return out;
}

@group(0) @binding(0) var<storage, read> sr_hist_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> sr_hist_out: array<u32>;
@group(0) @binding(2) var<uniform> sr_hist_params: SrRadixParams;

// Digit counts of one tile, written digit-major: hist[digit * tiles + tile].
@compute @workgroup_size(256)
fn scatter_reduce_radix_hist(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let tile = sr_tile_id(wid, nwg);
    let tiles = sr_hist_params.tiles;
    if (tile >= tiles) {
        return;
    }
    atomicStore(&sr_counts[tid], 0u);
    workgroupBarrier();

    let base = tile * SR_TILE;
    let n = sr_hist_params.n;
    let shift = sr_hist_params.shift;
    for (var k = tid; k < SR_TILE; k = k + SR_THREADS) {
        let i = base + k;
        if (i < n) {
            atomicAdd(&sr_counts[(sr_hist_keys[i] >> shift) & (SR_RADIX - 1u)], 1u);
        }
    }
    workgroupBarrier();

    sr_hist_out[tid * tiles + tile] = atomicLoad(&sr_counts[tid]);
}

@group(0) @binding(0) var<storage, read_write> sr_scan_hist: array<u32>;
@group(0) @binding(1) var<storage, read_write> sr_scan_totals: array<u32>;
@group(0) @binding(2) var<uniform> sr_scan_params: SrRadixParams;

// One workgroup per digit: replaces the digit's row of counts with each tile's
// exclusive offset within the digit, and writes the digit's total.
@compute @workgroup_size(256)
fn scatter_reduce_radix_scan(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let tiles = sr_scan_params.tiles;
    let row = wid.x * tiles;
    var carry = 0u;
    for (var c = 0u; c < tiles; c = c + SR_THREADS) {
        let t = c + tid;
        var v = 0u;
        if (t < tiles) {
            v = sr_scan_hist[row + t];
        }
        let incl = sr_block_inclusive_scan(tid, v);
        if (t < tiles) {
            sr_scan_hist[row + t] = carry + incl - v;
        }
        // The last invocation's inclusive value is the chunk total; every
        // invocation reads it through workgroup memory.
        if (tid == SR_THREADS - 1u) {
            sr_chunk_total = incl;
        }
        workgroupBarrier();
        carry = carry + sr_chunk_total;
        workgroupBarrier();
    }
    if (tid == 0u) {
        sr_scan_totals[wid.x] = carry;
    }
}

@group(0) @binding(0) var<storage, read> sr_sc_keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> sr_sc_vals_in: array<u32>;
@group(0) @binding(2) var<storage, read> sr_sc_hist: array<u32>;
@group(0) @binding(3) var<storage, read> sr_sc_totals: array<u32>;
@group(0) @binding(4) var<storage, read_write> sr_sc_keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> sr_sc_vals_out: array<u32>;
@group(0) @binding(6) var<uniform> sr_sc_params: SrRadixParams;

// Moves one tile's entries to their sorted slots. Round `r` holds tile entries
// [r * 256, r * 256 + 256), one per invocation in source order, so the slot
// order inside one digit is source order.
@compute @workgroup_size(256)
fn scatter_reduce_radix_scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let tile = sr_tile_id(wid, nwg);
    let tiles = sr_sc_params.tiles;
    if (tile >= tiles) {
        return;
    }
    let n = sr_sc_params.n;
    let shift = sr_sc_params.shift;

    // Digit base: the exclusive scan of every digit's total. SR_THREADS equals
    // SR_RADIX, so invocation `tid` owns digit `tid`.
    let total = sr_sc_totals[tid];
    let digit_base = sr_block_inclusive_scan(tid, total) - total;
    sr_offsets[tid] = digit_base + sr_sc_hist[tid * tiles + tile];

    let word = tid >> 5u;
    let below = (1u << (tid & 31u)) - 1u;
    let tile_base = tile * SR_TILE;

    for (var r = 0u; r < SR_ITEMS; r = r + 1u) {
        let round_base = tile_base + r * SR_THREADS;
        if (round_base >= n) {
            break;
        }
        for (var w = 0u; w < SR_MASK_WORDS; w = w + 1u) {
            atomicStore(&sr_masks[tid * SR_MASK_WORDS + w], 0u);
        }
        workgroupBarrier();

        let i = round_base + tid;
        let valid = i < n;
        var key = 0u;
        var digit = 0u;
        if (valid) {
            key = sr_sc_keys_in[i];
            digit = (key >> shift) & (SR_RADIX - 1u);
            atomicOr(&sr_masks[digit * SR_MASK_WORDS + word], 1u << (tid & 31u));
        }
        workgroupBarrier();

        // Rank among this round's entries with the same digit, and how many
        // such entries the round holds.
        var rank = 0u;
        var count = 0u;
        if (valid) {
            for (var w = 0u; w < SR_MASK_WORDS; w = w + 1u) {
                let m = atomicLoad(&sr_masks[digit * SR_MASK_WORDS + w]);
                count = count + countOneBits(m);
                if (w < word) {
                    rank = rank + countOneBits(m);
                } else if (w == word) {
                    rank = rank + countOneBits(m & below);
                }
            }
            let slot = sr_offsets[digit] + rank;
            sr_sc_keys_out[slot] = key;
            sr_sc_vals_out[slot] = sr_sc_vals_in[i];
        }
        workgroupBarrier();

        // The digit's first entry in this round advances its offset.
        if (valid && rank == 0u) {
            sr_offsets[digit] = sr_offsets[digit] + count;
        }
        workgroupBarrier();
    }
}
