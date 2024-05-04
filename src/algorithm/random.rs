use rand::Rng;

/// Returns a random number generator.
///
/// Uses [`rand::rngs::ThreadRng`] to get a random number generator.
/// It refreshes entropy every 64 KiB of random data and on fork.
pub fn get_random_generator() -> impl Rng {
    rand::thread_rng()
    // let seed: [u8; 32] = [42; 32];
    // StdRng::from_seed(seed)
}
