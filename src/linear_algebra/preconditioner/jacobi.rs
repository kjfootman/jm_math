use crate::{
    error::Error,
    linear_algebra::{matrix::csr::CSRMatrix, preconditioner::Preconditioner, simd},
};

pub struct Jacobi<'a> {
    matrix: &'a CSRMatrix,
}

struct DiagonalDivision;

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

        // let arch = simd::arch();
        // diag_ptr.iter().zip(v.iter_mut()).for_each(|(diag_ptr, v)| {
        //     *v /= values[*diag_ptr as usize];
        // });

        // method1: without simd
        inplace
            .iter_mut()
            .zip(diag_ptr)
            .zip(v)
            .for_each(|((inplace, diag_ptr), v)| *inplace = *v / values[*diag_ptr as usize]);

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
