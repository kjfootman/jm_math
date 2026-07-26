use pulp::{Arch, Simd, WithSimd};
use std::sync::OnceLock;

static ARCH: OnceLock<Arch> = OnceLock::new();

pub fn arch() -> &'static Arch {
    ARCH.get_or_init(Arch::new)
}

pub fn calculate_chunk_size(len: usize) -> usize {
    let n_thread = rayon::current_num_threads();

    ((len / (n_thread * 4)).max(1024) + 7) & !7
}

pub struct VectorAdd<'a>(pub &'a mut [f64], pub &'a [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorAdd<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (a_head, a_tail) = S::as_simd_f64s(self.1);
        let (b_head, b_tail) = S::as_simd_f64s(self.2);

        out_head
            .iter_mut()
            .zip(a_head.iter())
            .zip(b_head.iter())
            .for_each(|((out, a), b)| {
                *out = simd.add_f64s(*a, *b);
            });

        out_tail
            .iter_mut()
            .zip(a_tail.iter().zip(b_tail.iter()))
            .for_each(|(out, (a, b))| {
                *out = a + b;
            });
    }
}

pub struct VectorAddAssign<'a>(pub &'a mut [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorAddAssign<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (vec_head, vec_tail) = S::as_simd_f64s(self.1);

        out_head
            .iter_mut()
            .zip(vec_head.iter())
            .for_each(|(a, b)| *a = simd.add_f64s(*a, *b));

        out_tail
            .iter_mut()
            .zip(vec_tail.iter())
            .for_each(|(a, b)| *a += b);
    }
}

pub struct VectorSub<'a>(pub &'a mut [f64], pub &'a [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorSub<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (a_head, a_tail) = S::as_simd_f64s(self.1);
        let (b_head, b_tail) = S::as_simd_f64s(self.2);

        out_head
            .iter_mut()
            .zip(a_head.iter())
            .zip(b_head.iter())
            .for_each(|((out, a), b)| {
                *out = simd.sub_f64s(*a, *b);
            });

        out_tail
            .iter_mut()
            .zip(a_tail.iter().zip(b_tail.iter()))
            .for_each(|(out, (a, b))| {
                *out = a - b;
            });
    }
}

pub struct VectorSubAssign<'a>(pub &'a mut [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorSubAssign<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (vec_head, vec_tail) = S::as_simd_f64s(self.1);

        out_head
            .iter_mut()
            .zip(vec_head.iter())
            .for_each(|(a, b)| *a = simd.sub_f64s(*a, *b));

        out_tail
            .iter_mut()
            .zip(vec_tail.iter())
            .for_each(|(a, b)| *a -= b);
    }
}

pub struct VectorScale<'a>(pub &'a mut [f64], pub f64, pub &'a [f64]);
impl<'a> WithSimd for VectorScale<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let scale = self.1;
        let (vec_head, vec_tail) = S::as_simd_f64s(self.2);
        let scale_simd = simd.splat_f64s(self.1);

        out_head.iter_mut().zip(vec_head.iter()).for_each(|(a, b)| {
            *a = simd.mul_f64s(scale_simd, *b);
        });

        out_tail
            .iter_mut()
            .zip(vec_tail.iter())
            .for_each(|(a, b)| *a = scale * b);
    }
}

pub struct VectorScaleAssign<'a>(pub &'a mut [f64], pub f64);
impl<'a> WithSimd for VectorScaleAssign<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_mut_simd_f64s(self.0);
        let scale = self.1;
        let scale_simd = simd.splat_f64s(scale);

        head.iter_mut()
            .for_each(|v| *v = simd.mul_f64s(scale_simd, *v));

        tail.iter_mut().for_each(|v| *v *= scale);
    }
}

pub struct VectorDot<'a>(pub &'a [f64], pub &'a [f64]);
impl<'a> WithSimd for VectorDot<'a> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f64s(self.0);
        let (b_head, b_tail) = S::as_simd_f64s(self.1);
        let mut c = simd.splat_f64s(0.0);

        // a * b 결과를 acc 에 누적 : c = a * b + c
        a_head
            .iter()
            .zip(b_head.iter())
            .for_each(|(a, b)| c = simd.mul_add_f64s(*a, *b, c));

        let mut result = simd.reduce_sum_f64s(c);

        result += a_tail
            .iter()
            .zip(b_tail.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

        result
    }
}

pub struct VectorNeg<'a>(pub &'a mut [f64]);
impl<'a> WithSimd for VectorNeg<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_mut_simd_f64s(self.0);

        head.iter_mut().for_each(|v| *v = simd.neg_f64s(*v));
        tail.iter_mut().for_each(|v| *v = -*v);
    }
}
