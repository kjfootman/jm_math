use super::simd;
use crate::Error;
use log::error;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, Neg, Range, Sub, SubAssign};

#[derive(Debug, PartialEq)]
pub struct Vector {
    values: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
        }
    }

    pub fn add(&mut self, a: &[f64], b: &[f64]) -> Result<(), Error> {
        let len = self.len();

        if a.len() != b.len() || len != a.len() {
            let msg = "Cannot add vectors of different dimensions";
            error!("{msg}");
            Err(Error::DimensionMismatch(msg.into()))?
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(a.par_chunks(chunk_size))
            .zip(b.par_chunks(chunk_size))
            .for_each(|((out, a), b)| {
                arch.dispatch(simd::VectorAdd(out, a, b));
            });

        Ok(())
    }

    pub fn add_assign(&mut self, vec: &[f64]) {
        let len = self.len();

        if len != vec.len() {
            error!("Cannot add vectors of different dimensions");
            panic!();
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .for_each(|(out, vec)| {
                arch.dispatch(simd::VectorAddAssign(out, vec));
            });
    }

    pub fn sub(&mut self, a: &[f64], b: &[f64]) -> Result<(), Error> {
        let len = self.len();

        if a.len() != b.len() || len != a.len() {
            let msg = "Cannot subtract vectors of different dimensions";
            error!("{msg}");
            Err(Error::DimensionMismatch(msg.into()))?
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(a.par_chunks(chunk_size))
            .zip(b.par_chunks(chunk_size))
            .for_each(|((out, a), b)| {
                arch.dispatch(simd::VectorSub(out, a, b));
            });

        Ok(())
    }

    pub fn sub_assign(&mut self, vec: &[f64]) {
        let len = self.len();

        if len != vec.len() {
            error!("Cannot subtract vectors of different dimensions");
            panic!();
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .for_each(|(out, vec)| {
                arch.dispatch(simd::VectorSubAssign(out, vec));
            });
    }

    pub fn scale(&mut self, scale: f64, vec: &[f64]) {
        let len = self.len();

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .for_each(|(out, vec)| {
                arch.dispatch(simd::VectorScale(out, scale, vec));
            });
    }

    pub fn scale_assign(&mut self, scale: f64) {
        let len = self.len();

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size).for_each(|out| {
            arch.dispatch(simd::VectorScaleAssign(out, scale));
        });
    }

    pub fn dot(&self, vec: &[f64]) -> f64 {
        let len = self.len();

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .map(|(a, b)| arch.dispatch(simd::VectorDot(a, b)))
            .sum::<f64>()
    }

    pub fn neg_assign(&mut self) {
        let len = self.len();
        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .for_each(|v| arch.dispatch(simd::VectorNeg(v)));
    }
}

impl Deref for Vector {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Index<Range<usize>> for Vector {
    type Output = [f64];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<Range<usize>> for Vector {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl From<Vec<f64>> for Vector {
    fn from(values: Vec<f64>) -> Self {
        Self { values }
    }
}

impl Neg for Vector {
    type Output = Vector;

    fn neg(mut self) -> Self::Output {
        self.neg_assign();

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const N: usize = 2_500;

    #[test]
    /// Unit test for `Vector` addition.
    fn vector_add_test() -> Result<(), Error> {
        let v1 = Vector::from(vec![1.0; N]);
        let v2 = Vector::from(vec![2.0; N]);
        let v3 = Vector::from(vec![-1.0; N]);
        let mut v = Vector::new(N);

        // v1 + v2 -> [3.0; N]
        v.add(&v1, &v2)?;

        // [3.0; N]에 v3([-1.0; N]) 3회 누적 덧셈
        for _ in 0..3 {
            v.add_assign(&v3);
        }

        assert_eq!(v, Vector::from(vec![0.0; N]));

        Ok(())
    }

    #[test]
    fn vector_sub_test() -> Result<(), Error> {
        let v1 = Vector::from(vec![-1.0; N]);
        let v2 = Vector::from(vec![2.0; N]);
        let v3 = Vector::from(vec![-1.0; N]);
        let mut v = Vector::new(N);

        // v1 - v2 -> [-3.0; N]
        v.sub(&v1, &v2)?;

        // [-3.0; N]에 v3([-1.0; N]) 3회 누적 뺄셈
        for _ in 0..3 {
            v.sub_assign(&v3);
        }

        assert_eq!(v, Vector::from(vec![0.0; N]));

        Ok(())
    }

    #[test]
    fn vector_scale_test() -> Result<(), Error> {
        let scale = 2.0;
        let v1 = Vector::from(vec![1.0; N]);
        let mut v = Vector::new(N);

        // 2.0 * v1 -> [2.0; N]
        v.scale(scale, &v1);

        // [2.0; N]에 2.0 누적 곱셈
        for _ in 0..3 {
            v.scale_assign(2.0);
        }

        assert_eq!(v, Vector::from(vec![16.0; N]));

        Ok(())
    }

    #[test]
    fn vector_dot_test() -> Result<(), Error> {
        let v1 = Vector::from(vec![1.0; N]);
        let v2 = Vector::from(vec![1.0; N]);

        assert_eq!(N as f64, v1.dot(&v2));

        Ok(())
    }

    #[test]
    fn vector_neg_test() -> Result<(), Error> {
        let mut v = Vector::from(vec![-1.0; N]);
        v.neg_assign();

        assert_eq!(N as f64, v.iter().sum::<f64>());

        Ok(())
    }
}
