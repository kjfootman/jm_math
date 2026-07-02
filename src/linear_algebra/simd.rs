use crate::Error;
use pulp::{Arch, Simd, WithSimd};
use std::sync::OnceLock;

static ARCH: OnceLock<Arch> = OnceLock::new();

pub fn arch() -> &'static Arch {
    ARCH.get_or_init(Arch::new)
}

/// Returns the optimum chunk size.
/// The minimum chunk size is 1024.
pub fn calculate_chunk_size(len: usize) -> usize {
    let n_thread = rayon::current_num_threads();

    ((len / (n_thread * 4)).max(1024) + 7) & !7
}
