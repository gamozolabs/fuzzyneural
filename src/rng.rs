use std::cell::Cell;

/// Basic xorshift-based RNG
#[derive(Debug, Clone)]
pub struct Rng {
    /// RNG seed, wrapped in a `Cell` so we can get random numbers with `&self`
    seed: Cell<u64>,
}

impl Rng {
    /// Create a new random number generator
    pub fn new() -> Self {
        let ret = Rng { seed: Cell::new(0) };
        ret.reseed();
        ret
    }

    /// Generate a new seed for this Rng
    pub fn reseed(&self) {
        let tsc = unsafe { core::arch::x86_64::_rdtsc() as u64 };
        self.seed.set(tsc);

        // Shuffle in the TSC
        for _ in 0..128 {
            self.rand();
        }
    }

    /// Using xorshift get a new random number
    pub fn rand(&self) -> u64 {
        let mut seed = self.seed.get();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 43;
        self.seed.set(seed);
        seed
    }

    /// Generates a random floating point value in the range [min, max]
    pub fn rand_f32(&self, min: f32, max: f32) -> f32 {
        // Make sure max is larger than min
        assert!(max > min, "Invalid rand_f32 range");

        // Compute the magnitude of the range
        let magnitude = max - min;

        // Generate a random value in the range [0.0, 1.0]
        let rand_val = (self.rand() as f32) / (std::u64::MAX as f32);

        // Apply the magnitude to the random value, based on the size of our
        // range
        let rand_val = rand_val * magnitude;

        // Skew the random value WRT to the minimum value
        min + rand_val
    }
}
