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
        let (head0, tail0) = S::as_mut_simd_f64s(self.0);
        let (head1, tail1) = S::as_simd_f64s(self.1);

        head0
            .iter_mut()
            .zip(head1.iter())
            .for_each(|(v1, v2)| *v1 = simd.add_f64s(*v1, *v2));

        tail0
            .iter_mut()
            .zip(tail1.iter())
            .for_each(|(v1, v2)| *v1 += v2);
    }
}

pub struct VectorSub<'a>(pub &'a mut [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorSub<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head0, tail0) = S::as_mut_simd_f64s(self.0);
        let (head1, tail1) = S::as_simd_f64s(self.1);

        head0
            .iter_mut()
            .zip(head1.iter())
            .for_each(|(v1, v2)| *v1 = simd.sub_f64s(*v1, *v2));

        tail0
            .iter_mut()
            .zip(tail1.iter())
            .for_each(|(v1, v2)| *v1 -= v2);
    }
}

pub struct VectorMul<'a>(pub &'a mut [f64], pub f64);
impl<'a> WithSimd for VectorMul<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_mut_simd_f64s(self.0);
        let scalar = self.1;
        let scalar_simd = simd.splat_f64s(scalar);

        head.iter_mut()
            .for_each(|v| *v = simd.mul_f64s(scalar_simd, *v));

        tail.iter_mut().for_each(|v| *v *= scalar);
    }
}

pub struct VectorDot<'a>(pub &'a [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorDot<'a> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head0, tail0) = S::as_simd_f64s(self.0);
        let (head1, tail1) = S::as_simd_f64s(self.1);
        let mut c = simd.splat_f64s(0.0);

        // a * b 결과를 acc 에 누적 : c = a * b + c
        head0
            .iter()
            .zip(head1.iter())
            .for_each(|(a, b)| c = simd.mul_add_f64s(*a, *b, c));

        let mut result = simd.reduce_sum_f64s(c);

        result += tail0
            .iter()
            .zip(tail1.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

        result
    }
}
