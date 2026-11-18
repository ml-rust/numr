//! Inputs of the correctly rounded parity tests: specials, thresholds, structured values
//! and fixed-seed random bit patterns.

/// The hydr regression input.
pub const HYDR_REGRESSION: f64 = 0.156_043_060_815_255_54;

/// SplitMix64: a fixed-seed generator for reproducible samples.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The doubles `reach` units in the last place either side of `bits`, along the bit pattern.
fn around(bits: u64, reach: u64) -> impl Iterator<Item = f64> {
    (0..=2 * reach)
        .filter_map(move |k| bits.checked_add(k)?.checked_sub(reach))
        .map(f64::from_bits)
}

/// Zeros, infinities, NaNs with payloads, subnormal and normal edges, and both signs.
pub fn specials() -> Vec<f64> {
    let mut v = vec![
        0.0,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0x7ff4_0000_0000_dead),
        f64::from_bits(0x7ff8_0000_0000_beef),
        f64::from_bits(0x7fff_ffff_ffff_ffff),
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        f64::MAX,
        1.0,
        2.0,
        0.5,
        3.0,
        9.0,
        0.25,
        HYDR_REGRESSION,
    ];
    let negatives: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v
}

/// Thresholds of every branch, powers of two, values near 1 and -1, and the exp
/// subnormal-output band.
pub fn adversarial() -> Vec<f64> {
    let mut v = specials();
    // exp: overflow, zero, subnormal-result and tiny-input thresholds; log1p: its
    // series and large-argument thresholds; log: its tiny window around 1.
    for bits in [
        0x4086_2e42_fefa_39ef,
        0x4086_2e42_fefa_39f0,
        0xc087_4910_d52d_3052,
        0xc086_232b_dd7a_bcd2,
        0x3c90_0000_0000_0000,
        0x3fb0_0000_0000_0000,
        0x3f30_0000_0000_0000,
        0x3ea1_9b00_0000_0000,
        0x3ca0_0000_0000_0000,
        0x4340_0000_0000_0000,
        0x4690_0000_0000_0000,
        0x3fef_f000_0000_0000,
        0x3ff0_0800_0000_0000,
        0x3fe6_a09e_667f_3bcd,
        0x3ff6_a09e_667f_3bcd,
    ] {
        v.extend(around(bits, 8));
        v.extend(around(bits | (1 << 63), 8));
    }
    for k in 0..64 {
        let one = 1.0f64.to_bits();
        v.push(f64::from_bits(one + k));
        v.push(f64::from_bits(one - k));
        v.push(-f64::from_bits(one + k));
        v.push(-f64::from_bits(one - k));
    }
    for k in -1074..=1023 {
        let p = 2f64.powi(k);
        v.extend([p, -p]);
    }
    // Subnormal results of exp: x from -745.2 to -708.
    let mut x = -745.2f64;
    while x < -708.0 {
        v.push(x);
        x += 0.013_7;
    }
    v
}

/// Uniformly random bit patterns, and positive values spread over every binade.
pub fn dense(count: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix(seed);
    let mut v = Vec::with_capacity(2 * count);
    for _ in 0..count {
        let bits = rng.next_u64();
        v.push(f64::from_bits(bits));
        let exponent = 1 + (bits >> 52) % 0x7fe;
        v.push(f64::from_bits((exponent << 52) | (rng.next_u64() >> 12)));
    }
    v
}

/// Exponents of `cr_pow_scalar`: hydr's, the Annex F specials, integers, and exponents that
/// reach the exact, midpoint and extended branches.
pub fn exponents() -> Vec<f64> {
    vec![
        4.0 / 3.0,
        7.0 / 3.0,
        5.0 / 3.0,
        1.666_666_7,
        1.0 - 1.666_666_7,
        1.5,
        0.385,
        0.5,
        -0.5,
        0.25,
        1.0 / 3.0,
        -1.0,
        1.0,
        2.0,
        3.0,
        34.0,
        -3.0,
        1.0e6,
        4_503_599_627_370_497.0,
        -1074.0,
        -1075.0,
        1.0e300,
        2f64.powi(-60),
        2f64.powi(-1000),
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
}
