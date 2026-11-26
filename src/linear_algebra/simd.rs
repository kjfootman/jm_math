use crate::linear_algebra::vector;
use pulp::{Arch, Simd, WithSimd};
use std::sync::OnceLock;

static ARCH: OnceLock<Arch> = OnceLock::new();

pub fn arch() -> &'static Arch {
    ARCH.get_or_init(Arch::new)
}

pub fn calc_chunk_size(len: usize) -> usize {
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

pub struct VectorScaleAdd<'a>(pub &'a mut [f64], pub &'a [f64], pub f64, pub &'a [f64]);
impl<'a> WithSimd for VectorScaleAdd<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (a_head, a_tail) = S::as_simd_f64s(self.3);
        let scale = self.2;
        let (b_head, b_tail) = S::as_simd_f64s(self.1);
        let scale_simd = simd.splat_f64s(scale);

        out_head
            .iter_mut()
            .zip(a_head.iter())
            .zip(b_head.iter())
            .for_each(|((v, a), b)| {
                *v = simd.mul_add_f64s(scale_simd, *a, *b);
            });

        out_tail
            .iter_mut()
            .zip(a_tail.iter())
            .zip(b_tail.iter())
            .for_each(|((v, a), b)| {
                *v = scale * a + b;
            });
    }
}

pub struct VectorScaleAddAssign<'a>(pub &'a mut [f64], pub f64, pub &'a [f64]);
impl<'a> WithSimd for VectorScaleAddAssign<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.0);
        let (a_head, a_tail) = S::as_simd_f64s(self.2);
        let scale = self.1;
        let scale_simd = simd.splat_f64s(scale);

        out_head.iter_mut().zip(a_head.iter()).for_each(|(v, a)| {
            *v = simd.mul_add_f64s(scale_simd, *a, *v);
        });

        out_tail.iter_mut().zip(a_tail.iter()).for_each(|(v, a)| {
            *v += scale * a;
        });
    }
}

//-----------------------------------------------------------------------------//
pub struct VectorScaleAddAssignMany<'a> {
    pub out: &'a mut [f64],
    pub scales: &'a [f64],
    pub vecs: &'a [vector::Vector],
}
impl<'a> WithSimd for VectorScaleAddAssignMany<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let vecs = self.vecs;
        let scales = self.scales;
        let (out_head, out_tail) = S::as_mut_simd_f64s(self.out);
        let mut i = 0;

        while i + 4 <= vecs.len() {
            let s0 = scales[i];
            let s1 = scales[i + 1];
            let s2 = scales[i + 2];
            let s3 = scales[i + 3];

            let s0_simd = simd.splat_f64s(s0);
            let s1_simd = simd.splat_f64s(s1);
            let s2_simd = simd.splat_f64s(s2);
            let s3_simd = simd.splat_f64s(s3);

            let (v0_head, v0_tail) = S::as_simd_f64s(&vecs[i]);
            let (v1_head, v1_tail) = S::as_simd_f64s(&vecs[i + 1]);
            let (v2_head, v2_tail) = S::as_simd_f64s(&vecs[i + 2]);
            let (v3_head, v3_tail) = S::as_simd_f64s(&vecs[i + 3]);

            for j in 0..out_head.len() {
                let mut value = out_head[j];
                value = simd.mul_add_f64s(s0_simd, v0_head[j], value);
                value = simd.mul_add_f64s(s1_simd, v1_head[j], value);
                value = simd.mul_add_f64s(s2_simd, v2_head[j], value);
                value = simd.mul_add_f64s(s3_simd, v3_head[j], value);
                out_head[j] = value;
            }

            for j in 0..out_tail.len() {
                out_tail[j] +=
                    s0 * v0_tail[j] + s1 * v1_tail[j] + s2 * v2_tail[j] + s3 * v3_tail[j];
            }

            i += 4;
            // println!("i: {i}");
        }

        // println!(
        //     "vecs.len() {}, scales.len() {} i {i}",
        //     vecs.len(),
        //     scales.len()
        // );
        while i < vecs.len() {
            let scale = scales[i];
            let s0_simd = simd.splat_f64s(scale);
            let (v0_head, v0_tail) = S::as_simd_f64s(&vecs[i]);

            for j in 0..out_head.len() {
                out_head[j] = simd.mul_add_f64s(s0_simd, v0_head[j], out_head[j]);
            }

            for j in 0..out_tail.len() {
                out_tail[j] += scale * v0_tail[j];
            }

            i += 1;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::Error, linear_algebra::vector::Vector};

    // fn init() {
    //     let _ = env_logger::builder().is_test(true).try_init();
    // }

    #[test]
    fn vectorScaleAddAssignMany() -> Result<(), Error> {
        //-----------------------------------------------------------------------------//
        // test1
        let mut out = vec![1.0; 13];
        let scales = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vecs = vec![
            Vector::from(vec![4.0; 13]),
            Vector::from(vec![3.0; 13]),
            Vector::from(vec![2.0; 13]),
            Vector::from(vec![1.0; 13]),
            Vector::from(vec![4.0; 13]),
        ];

        let arch = arch();
        arch.dispatch(VectorScaleAddAssignMany {
            out: &mut out,
            scales: &scales,
            vecs: &vecs,
        });

        assert_eq!(out, vec![41.0; 13]);

        //-----------------------------------------------------------------------------//
        // test2
        let mut out = vec![1.0; 13];
        let scales = vec![1.0, 2.0, 3.0];
        let vecs = vec![
            Vector::from(vec![4.0; 13]),
            Vector::from(vec![3.0; 13]),
            Vector::from(vec![2.0; 13]),
        ];

        arch.dispatch(VectorScaleAddAssignMany {
            out: &mut out,
            scales: &scales,
            vecs: &vecs,
        });

        assert_eq!(out, vec![17.0; 13]);

        //-----------------------------------------------------------------------------//
        // test3
        let mut out = vec![1.0; 13];
        let scales = vec![1.0, 2.0, 3.0, 4.0];
        let vecs = vec![
            Vector::from(vec![4.0; 13]),
            Vector::from(vec![3.0; 13]),
            Vector::from(vec![2.0; 13]),
            Vector::from(vec![1.0; 13]),
        ];

        arch.dispatch(VectorScaleAddAssignMany {
            out: &mut out,
            scales: &scales,
            vecs: &vecs,
        });

        assert_eq!(out, vec![21.0; 13]);

        Ok(())
    }
}
