use crate::Error;
use pulp::{Arch, Simd, WithSimd};
use std::sync::OnceLock;

static ARCH: OnceLock<Arch> = OnceLock::new();

pub fn arch() -> &'static Arch {
    ARCH.get_or_init(Arch::new)
}

/// Calculates an optimized chunk size for parallel iterator processing
/// using a combination of workload oversampling and SIMD lane alignment.
///
/// This utility ensures efficient dynamic load balancing via Rayon and
/// minimizes scalar tail loop overhead during vectorized operations.
///
/// # Arguments
///
/// * `len` - The total number of elements in the collection or slice.
///
/// # Returns
///
/// An optimized chunk size that is at least 1024 and guaranteed to be a multiple of 8.
pub fn calculate_chunk_size(len: usize) -> usize {
    let n_thread = rayon::current_num_threads();

    ((len / (n_thread * 4)).max(1024) + 7) & !7
}

pub struct VectorAdd<'a>(pub &'a mut [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorAdd<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head1, tail1) = S::as_mut_simd_f64s(self.0);
        let (head2, tail2) = S::as_simd_f64s(self.1);

        head1
            .iter_mut()
            .zip(head2.iter())
            .for_each(|(v1, v2)| *v1 = simd.add_f64s(*v1, *v2));

        tail1
            .iter_mut()
            .zip(tail2.iter())
            .for_each(|(v1, v2)| *v1 += v2);
    }
}
