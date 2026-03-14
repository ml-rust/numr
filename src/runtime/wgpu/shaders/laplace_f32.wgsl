// Laplace distribution sampling for f32

// PCG hash function for random number generation
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_init(seed: u32, idx: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(idx));
}

fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;
}

const WORKGROUP_SIZE: u32 = 256u;

struct LaplaceParams {
    numel: u32,
    seed: u32,
    loc: f32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: LaplaceParams;

@compute @workgroup_size(256)
fn laplace_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < params.numel {
        var state = pcg_init(params.seed, idx);
        let u = pcg_uniform(&state) - 0.5;
        let result = params.loc - params.scale * sign(u) * log(1.0 - 2.0 * abs(u));
        out[idx] = f32(result);
    }
}
