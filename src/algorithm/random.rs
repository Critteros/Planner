use rand::Rng;

/// Returns a random number generator.
pub fn get_random_generator() -> impl Rng {
    rand::thread_rng()
    // let seed: [u8; 32] = [42; 32];
    // StdRng::from_seed(seed)
}
