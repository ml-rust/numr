// Bernoulli distribution sampling for f32

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

struct BernoulliParams {
    numel: u32,
    seed: u32,
    p: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: BernoulliParams;

@compute @workgroup_size(256)
fn bernoulli_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < params.numel {
        var state = pcg_init(params.seed, idx);
        let u = pcg_uniform(&state);
        out[idx] = select(f32(0.0), f32(1.0), u < params.p);
    }
}
