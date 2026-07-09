use super::simd;
use log::error;
use rayon::prelude::*;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Range, Sub};

#[derive(Debug)]
pub struct Vector {
    values: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
        }
    }

    fn add_assign(&mut self, rhs: &[f64]) {
        let len = self.len();

        if len != rhs.len() {
            error!("Cannot add vectors of different dimensions");
            panic!();
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(rhs.par_chunks(chunk_size))
            .for_each(|(lhs, rhs)| arch.dispatch(simd::VectorAdd(lhs, rhs)));
    }

    fn sub_assign(&mut self, rhs: &[f64]) {
        let len = self.len();

        if len != rhs.len() {
            error!("Cannot subtract vectors of different dimensions");
            panic!();
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(rhs.par_chunks(chunk_size))
            .for_each(|(lhs, rhs)| arch.dispatch(simd::VectorSub(lhs, rhs)));
    }

    fn mul_assign(&mut self, rhs: f64) {
        let len = self.len();
        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .for_each(|v| arch.dispatch(simd::VectorMul(v, rhs)));
    }

    fn dot(&self, rhs: &[f64]) -> f64 {
        let len = self.len();

        if len != rhs.len() {
            error!("Cannot multiply vectors of different dimensions");
            panic!();
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks(chunk_size)
            .zip(rhs.par_chunks(chunk_size))
            .map(|(lhs, rhs)| arch.dispatch(simd::VectorDot(lhs, rhs)))
            .sum::<f64>()
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

macro_rules! impl_vector_add {
    (Vector, Vector) => {
        impl Add<Vector> for Vector {
            type Output = Vector;

            fn add(mut self, rhs: Vector) -> Self::Output {
                self.add_assign(&rhs);
                self
            }
        }

        impl Add<&Vector> for Vector {
            type Output = Vector;

            fn add(mut self, rhs: &Vector) -> Self::Output {
                self.add_assign(rhs);
                self
            }
        }

        impl Add<Vector> for &Vector {
            type Output = Vector;

            fn add(self, mut rhs: Vector) -> Self::Output {
                rhs.add_assign(self);
                rhs
            }
        }
    };
}
impl_vector_add!(Vector, Vector);

macro_rules! impl_vector_sub {
    (Vector, Vector) => {
        impl Sub<Vector> for Vector {
            type Output = Vector;

            fn sub(mut self, rhs: Vector) -> Self::Output {
                self.sub_assign(&rhs);
                self
            }
        }

        impl Sub<&Vector> for Vector {
            type Output = Vector;

            fn sub(mut self, rhs: &Vector) -> Self::Output {
                self.sub_assign(rhs);
                self
            }
        }

        impl Sub<Vector> for &Vector {
            type Output = Vector;

            fn sub(self, mut rhs: Vector) -> Self::Output {
                rhs.sub_assign(self);
                rhs
            }
        }
    };
}
impl_vector_sub!(Vector, Vector);

macro_rules! impl_vector_mul {
    (f64, Vector) => {
        impl Mul<Vector> for f64 {
            type Output = Vector;

            fn mul(self, mut rhs: Vector) -> Self::Output {
                rhs.mul_assign(self);
                rhs
            }
        }

        impl Mul<f64> for Vector {
            type Output = Vector;

            fn mul(mut self, rhs: f64) -> Self::Output {
                self.mul_assign(rhs);
                self
            }
        }
    };

    (Vector, Vector) => {
        impl Mul<Vector> for Vector {
            type Output = f64;

            fn mul(self, rhs: Vector) -> Self::Output {
                self.dot(&rhs)
            }
        }

        impl Mul<Vector> for &Vector {
            type Output = f64;

            fn mul(self, rhs: Vector) -> Self::Output {
                rhs.dot(self)
            }
        }

        impl Mul<&Vector> for Vector {
            type Output = f64;

            fn mul(self, rhs: &Vector) -> Self::Output {
                self.dot(rhs)
            }
        }

        impl Mul<&Vector> for &Vector {
            type Output = f64;

            fn mul(self, rhs: &Vector) -> Self::Output {
                self.dot(rhs)
            }
        }
    };
}
impl_vector_mul!(f64, Vector);
impl_vector_mul!(Vector, Vector);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_test() {
        let v1 = Vector::from(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::from(vec![3.0, 2.0, 1.0]);

        // let v = v1 - &v2;
        // let v = v1 - v2;
        // let v = 3.0 * v1;
        // let v = v1 * 3.0;
        let v = &v1 * &v2;
        println!("{v:#?}");
    }

    // fn test(mut a: f64) -> f64 {
    //     a += 3.0;

    //     a
    // }
}
