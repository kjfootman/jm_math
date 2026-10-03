use pulp::WithSimd;

use crate::{
    error::Error,
    linear_algebra::{matrix::csr::CSRMatrix, preconditioner::Preconditioner, simd},
};

pub struct Jacobi<'a> {
    matrix: &'a CSRMatrix,
}

// Jacobi preconditioning simd 적용을 위한 구조체
// struct DiagonalDivision<'a> {
//     inplace: &'a mut [f64],
//     matrix: &'a CSRMatrix,
//     vector: &'a [f64],
// }
//
// impl<'a> WithSimd for DiagonalDivision<'a> {
//     type Output = ();
//
//     #[inline(always)]
//     fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
//         let (out_head, out_tail) = S::as_mut_simd_f64s(self.inplace);
//         let (v_head, v_out) = S::as_simd_f64s(self.vector);
//
//     }
// }

impl<'a> Jacobi<'a> {
    pub fn new(matrix: &'a CSRMatrix) -> Self {
        Jacobi { matrix }
    }
}

impl<'a> Preconditioner for Jacobi<'a> {
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        let diag_ptr = self
            .matrix
            .diag_ptr()
            .ok_or_else(|| Error::ValueError("Failed to get the diag_ptr".into()))?;
        let values = self.matrix.values();

        // method1: without simd
        inplace
            .iter_mut()
            .zip(diag_ptr)
            .zip(v)
            .for_each(|((inplace, diag_ptr), v)| *inplace = *v / values[*diag_ptr as usize]);

        // method2: with auto simd
        // let arch = simd::arch();
        // arch.dispatch(|| {
        //     inplace
        //         .iter_mut()
        //         .zip(diag_ptr)
        //         .zip(v)
        //         .for_each(|((inplace, diag_ptr), v)| *inplace = *v / values[*diag_ptr as usize]);
        // });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    use super::{Jacobi, Preconditioner};
    use crate::prelude::*;

    #[test]
    fn preconditioner_jacobi() -> Result<(), Error> {
        let (rows, cols) = (3, 3);
        let coordinates = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 0, 3.0),
            (1, 1, 4.0),
            (2, 2, 5.0),
        ];
        let M = CSRMatrix::from_coordinates(rows, cols, coordinates);
        let v = Vector::from(vec![1.0, 4.0, 5.0]);
        let mut z = Vector::new(v.len());
        let jacobi = Jacobi::new(&M);

        jacobi.preconditioning(&v, &mut z)?;

        assert_eq!(z, Vector::from(vec![1.0; 3]));

        Ok(())
    }
}
