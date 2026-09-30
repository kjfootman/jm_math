use super::Preconditioner;
use crate::error::Error;
use crate::linear_algebra::{CSRMatrix, simd};

pub struct Jacobi<'a> {
    matrix: &'a CSRMatrix,
}

impl<'a> Jacobi<'a> {
    pub fn new(matrix: &'a CSRMatrix) -> Self {
        Jacobi { matrix }
    }
}

impl<'a> Preconditioner for Jacobi<'a> {
    fn preconditioning(&self, v: &mut [f64]) -> Result<(), Error> {
        let diag_ptr = self
            .matrix
            .diag_ptr()
            .ok_or_else(|| Error::ValueError("Failed to get the diag_ptr".into()))?;
        let values = self.matrix.values();

        // let arch = simd::arch();
        diag_ptr.iter().zip(v.iter_mut()).for_each(|(diag_ptr, v)| {
            *v /= values[*diag_ptr as usize];
        });

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
        let mut v = Vector::from(vec![1.0, 4.0, 5.0]);
        let pc = Jacobi::new(&M);

        pc.preconditioning(&mut v)?;

        assert_eq!(v, Vector::from(vec![1.0; 3]));

        Ok(())
    }
}
