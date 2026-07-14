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

    fn neg_assign(&mut self) {
        let len = self.len();
        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .for_each(|v| arch.dispatch(simd::VectorNeg(v)));
    }

    // fn dot_product(&self, rhs: &[f64]) -> f64 {
    //     let len = self.len();

    //     if len != rhs.len() {
    //         error!("Cannot multiply vectors of different dimensions");
    //         panic!();
    //     }

    //     let arch = simd::arch();
    //     let chunk_size = simd::calculate_chunk_size(len);

    //     self.par_chunks(chunk_size)
    //         .zip(rhs.par_chunks(chunk_size))
    //         .map(|(lhs, rhs)| arch.dispatch(simd::VectorDot(lhs, rhs)))
    //         .sum::<f64>()
    // }
}

// impl AsRef<[f64]> for Vector {
//     fn as_ref(&self) -> &[f64] {
//         &self.values
//     }
// }

// impl AsMut<[f64]> for Vector {
//     fn as_mut(&mut self) -> &mut [f64] {
//         &mut self.values
//     }
// }

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

// macro_rules! impl_vector_add {
//     (Vector, Vector) => {
//         impl Add<Vector> for Vector {
//             type Output = Vector;

//             fn add(mut self, rhs: Vector) -> Self::Output {
//                 self.add_assign(&rhs);
//                 self
//             }
//         }

//         impl Add<&Vector> for Vector {
//             type Output = Vector;

//             fn add(mut self, rhs: &Vector) -> Self::Output {
//                 self.add_assign(rhs);
//                 self
//             }
//         }

//         impl Add<Vector> for &Vector {
//             type Output = Vector;

//             fn add(self, mut rhs: Vector) -> Self::Output {
//                 rhs.add_assign(self);
//                 rhs
//             }
//         }
//     };
// }
// impl_vector_add!(Vector, Vector);

// macro_rules! imple_vector_add_assign {
//     (Vector) => {
//         impl AddAssign<Vector> for Vector {
//             fn add_assign(&mut self, rhs: Vector) {
//                 self.add_assign(&rhs);
//             }
//         }

//         impl AddAssign<&Vector> for Vector {
//             fn add_assign(&mut self, rhs: &Vector) {
//                 self.add_assign(rhs);
//             }
//         }
//     };
// }
// imple_vector_add_assign!(Vector);

// macro_rules! impl_vector_sub {
//     (Vector, Vector) => {
//         impl Sub<Vector> for Vector {
//             type Output = Vector;

//             fn sub(mut self, rhs: Vector) -> Self::Output {
//                 self.sub_assign(&rhs);
//                 self
//             }
//         }

//         impl Sub<&Vector> for Vector {
//             type Output = Vector;

//             fn sub(mut self, rhs: &Vector) -> Self::Output {
//                 self.sub_assign(rhs);
//                 self
//             }
//         }

//         impl Sub<Vector> for &Vector {
//             type Output = Vector;

//             fn sub(self, mut rhs: Vector) -> Self::Output {
//                 rhs.sub_assign(self);
//                 rhs
//             }
//         }
//     };
// }
// impl_vector_sub!(Vector, Vector);

// macro_rules! impl_vector_mul {
//     (f64, Vector) => {
//         impl Mul<Vector> for f64 {
//             type Output = Vector;

//             fn mul(self, mut rhs: Vector) -> Self::Output {
//                 rhs.mul_assign(self);
//                 rhs
//             }
//         }

//         impl Mul<f64> for Vector {
//             type Output = Vector;

//             fn mul(mut self, rhs: f64) -> Self::Output {
//                 self.mul_assign(rhs);
//                 self
//             }
//         }
//     };

//     (Vector, Vector) => {
//         impl Mul<Vector> for Vector {
//             type Output = f64;

//             fn mul(self, rhs: Vector) -> Self::Output {
//                 self.dot_product(&rhs)
//             }
//         }

//         impl Mul<Vector> for &Vector {
//             type Output = f64;

//             fn mul(self, rhs: Vector) -> Self::Output {
//                 rhs.dot_product(self)
//             }
//         }

//         impl Mul<&Vector> for Vector {
//             type Output = f64;

//             fn mul(self, rhs: &Vector) -> Self::Output {
//                 self.dot_product(rhs)
//             }
//         }

//         impl Mul<&Vector> for &Vector {
//             type Output = f64;

//             fn mul(self, rhs: &Vector) -> Self::Output {
//                 self.dot_product(rhs)
//             }
//         }
//     };
// }
// impl_vector_mul!(f64, Vector);
// impl_vector_mul!(Vector, Vector);

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
}
