use super::simd;
use log::error;
use rayon::prelude::*;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Range};

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

    /// Performs in-place vector addition (`self += rhs`) using Rayon for multi-threading
    /// and Pulp for runtime-dispatched SIMD vectorization.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `self` and `rhs` are not equal.
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

// impl Add for Vector {
//     type Output = Vector;

//     /// `Vector` addition
//     fn add(self, rhs: Self) -> Self::Output {
//         assert_eq!(
//             self.len(),
//             rhs.len(),
//             "Cannot add vectors of different dimensions"
//         );

//         let v: Vec<f64> = self
//             .par_iter()
//             .zip(rhs.par_iter())
//             .map(|(v1, v2)| v1 + v2)
//             .collect();

//         Vector::from(v)
//     }
// }

macro_rules! impl_vector_add {
    (Vector, Vector) => {
        impl Add<Vector> for Vector {
            type Output = Vector;

            fn add(mut self, rhs: Vector) -> Self::Output {
                self.add_assign(&rhs);
                self
            }
        }
    };
}

impl_vector_add!(Vector, Vector);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_test() {
        let v1 = Vector::from(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::from(vec![3.0, 2.0, 1.0]);

        let v = v1 + v2;

        println!("{v:#?}");
    }
}
