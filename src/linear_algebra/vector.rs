use super::simd;
use crate::linear_algebra::CSRMatrix;
use crate::{error::Error, linear_algebra::Matrix};
use log::error;
use rayon::prelude::*;
use std::ops::{Deref, DerefMut, Index, IndexMut, Neg, Range};

#[derive(Debug, PartialEq)]
pub struct Vector {
    values: Vec<f64>,
}

impl Vector {
    // Returns the zero vector of size `size`.
    // - size: the length of vector
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

    pub fn add_assign(&mut self, vec: &[f64]) -> Result<(), Error> {
        let len = self.len();

        if len != vec.len() {
            let msg = "Cannot add vectors of different dimensions";
            error!("{msg}");
            return Err(Error::DimensionMismatch(msg.into()));
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .for_each(|(out, vec)| {
                arch.dispatch(simd::VectorAddAssign(out, vec));
            });

        Ok(())
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

    pub fn sub_assign(&mut self, vec: &[f64]) -> Result<(), Error> {
        let len = self.len();

        if len != vec.len() {
            let msg = "Cannot subtract vectors of different dimensions";
            error!("{msg}");
            Err(Error::DimensionMismatch(msg.into()))?
        }

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .for_each(|(out, vec)| {
                arch.dispatch(simd::VectorSubAssign(out, vec));
            });

        Ok(())
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

    // todo: Result 타입 반환
    pub fn dot(&self, vec: &[f64]) -> f64 {
        let len = self.len();

        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks(chunk_size)
            .zip(vec.par_chunks(chunk_size))
            .map(|(a, b)| arch.dispatch(simd::VectorDot(a, b)))
            .sum::<f64>()
    }

    pub fn csr_spmv(&mut self, matrix: &CSRMatrix, vec: &Vector) -> Result<(), Error> {
        let (m, n) = (matrix.rows(), matrix.cols());
        let ia = matrix.row_ptr();
        let ja = matrix.col_indices();
        let aa = matrix.values();

        if n != vec.len() {
            let msg = format!(
                "Dimension mismatch for SpMV (Columns of matrix: {}, length of vector: {}",
                n,
                vec.len()
            );
            error!("{msg}");
            return Err(Error::DimensionMismatch(msg));
        }

        // todo: chunk_size로 최적화
        self.par_iter_mut().enumerate().for_each(|(i, v)| {
            let start = ia[i];
            let end = ia[i + 1];

            *v = (start..end)
                .into_iter()
                .map(|j| aa[j] * vec[ja[j]])
                .sum::<f64>();
        });

        Ok(())
    }

    pub fn neg_assign(&mut self) {
        let len = self.len();
        let arch = simd::arch();
        let chunk_size = simd::calculate_chunk_size(len);

        self.par_chunks_mut(chunk_size)
            .for_each(|v| arch.dispatch(simd::VectorNeg(v)));
    }

    pub fn magnitude(&self) -> f64 {
        self.dot(self).sqrt()
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
    use crate::linear_algebra::CSRMatrixArgs;
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
            v.add_assign(&v3)?;
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
            v.sub_assign(&v3)?;
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

        for _ in 0..3 {
            v.neg_assign();
        }

        assert_eq!(N as f64, -v.iter().sum::<f64>());

        Ok(())
    }

    #[test]
    fn vector_magnitude_test() -> Result<(), Error> {
        let v = Vector::from(vec![1.0; N]);

        assert_eq!(v.magnitude(), 50.0);

        Ok(())
    }

    #[test]
    fn vector_csr_spmxv_test() {
        let (rows, cols) = (5, 5);
        let row_ptr = vec![0, 2, 5, 9, 11, 12];
        let col_indices = vec![0, 3, 0, 1, 3, 0, 2, 3, 4, 2, 3, 4];
        let values = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        // 행렬의 각 행 요소 합
        let row_sum_vec: Vec<f64> = row_ptr
            .windows(2)
            .map(|range| values[range[0]..range[1]].iter().sum())
            .collect();
        let row_sum_vec = Vector::from(row_sum_vec);

        // CSRMatrix 생성 및 1.0 벡터와 곱
        // 1.0   0.0   0.0   2.0   0.0   |   1.0
        // 3.0   4.0   0.0   5.0   0.0   |   1.0
        // 6.0   0.0   7.0   8.0   9.0   |   1.0
        // 0.0   0.0  10.0  11.0   0.0   |   1.0
        // 0.0   0.0   0.0   0.0  12.0   |   1.0
        let matrix = CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        });
        let vec = Vector::from(vec![1.0; cols]);
        let mut result = Vector::new(cols);

        result.csr_spmv(&matrix, &vec);

        // 결과 비교
        assert_eq!(result, row_sum_vec);
    }
}
