use crate::{
    error::Error,
    linear_algebra::{CSRMatrix, MSolver, Matrix, Vector, csr},
};
use std::{cell::RefCell, ops::DivAssign};

#[derive(Debug)]
pub struct GaussSeidel {
    residual: RefCell<f64>,
    tolerance: f64,
    iter_max: usize,
    iter: RefCell<usize>,
}

pub struct GaussSeidelBuilder {
    tolerance: Option<f64>,
    iter_max: Option<usize>,
}

impl GaussSeidel {
    pub fn iter(&self) -> usize {
        self.iter.take()
    }

    pub fn residual(&self) -> f64 {
        self.residual.take()
    }
}

impl Default for GaussSeidel {
    fn default() -> Self {
        GaussSeidel {
            residual: RefCell::new(f64::MAX),
            tolerance: 1E-7,
            iter_max: 500,
            iter: RefCell::new(0),
        }
    }
}

impl GaussSeidelBuilder {
    pub fn new() -> Self {
        Self {
            tolerance: None,
            iter_max: None,
        }
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    pub fn with_max_iter(mut self, iter_max: usize) -> Self {
        self.iter_max = Some(iter_max);
        self
    }

    pub fn build(self) -> GaussSeidel {
        GaussSeidel {
            tolerance: self.tolerance.unwrap_or_default(),
            iter_max: self.iter_max.unwrap_or_default(),
            ..Default::default()
        }
    }
}

impl MSolver for GaussSeidel {
    /// Solves the systems of euqations with Gauss-Seidel method.
    fn solve(&self, matrix: &CSRMatrix, b: &Vector) -> Result<Vector, Error> {
        let (m, n) = (matrix.rows(), matrix.cols());
        let mut iter = self.iter.borrow_mut();
        let mut residual = self.residual.borrow_mut();
        let tol = self.tolerance;
        let iter_max = self.iter_max;

        let b_mag = b.magnitude()?;
        let mut Ax = Vector::new(n);

        let A = matrix;
        let ia = A.row_ptr();
        let ja = A.col_indices();
        let da = A
            .diag_ptr()
            .ok_or_else(|| Error::ValueError("Diagonal pointer is not initialized".into()))?;
        let aa = A.values();
        let mut x = Vector::new(n);
        let mut r = Vector::new(n);

        // calculate residual vector r - Ax
        Ax.csr_spmv(matrix, &x)?;
        r.sub(b, &Ax)?;
        // relative calculate residual
        *residual = r.magnitude()?.abs() / b_mag;

        // main iteration
        while *residual > tol && *iter < iter_max {
            unsafe {
                for i in 0..n {
                    let start = *ia.get_unchecked(i);
                    let end = *ia.get_unchecked(i + 1);
                    let diag_idx = *da.get_unchecked(i);

                    let mut sum = *b.get_unchecked(i);

                    let aa_slice = aa.get_unchecked(start..diag_idx);
                    let ja_slice = ja.get_unchecked(start..diag_idx);
                    for (&a_val, &col_idx) in aa_slice.iter().zip(ja_slice.iter()) {
                        sum -= a_val * x.get_unchecked(col_idx);
                        // sum = (-a_val).mul_add(*x.get_unchecked(col_idx), sum);
                    }

                    let aa_slice = aa.get_unchecked(diag_idx + 1..end);
                    let ja_slice = ja.get_unchecked(diag_idx + 1..end);
                    for (&a_val, &col_idx) in aa_slice.iter().zip(ja_slice.iter()) {
                        sum -= a_val * x.get_unchecked(col_idx);
                        // sum = (-a_val).mul_add(*x.get_unchecked(col_idx), sum);
                    }

                    *x.get_unchecked_mut(i) = sum / *aa.get_unchecked(diag_idx);
                }
            }

            // calculate residual vector r - Ax
            Ax.csr_spmv(matrix, &x)?;
            r.sub(b, &Ax)?;
            // relative calculate residual
            *residual = r.magnitude()?.abs() / b_mag;

            *iter += 1;

            log::debug!("iter: {}, residual: {:.2E}", *iter, *residual);
            if *iter >= iter_max {
                return Err(Error::Convergence(*iter, *residual));
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::CSRMatrixArgs;

    #[test]
    fn gauss_seidel_test() -> Result<(), Error> {
        let (rows, cols) = (4, 4);
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let diag_ptr = csr::find_diag_ptr(&row_ptr, &col_indices).ok();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let M = CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr,
            diag_ptr,
            col_indices,
            values,
        });
        let b = Vector::from(vec![6.0, 15.0, 15.0, 9.0]);

        let gs = GaussSeidelBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            .build();

        let x = gs.solve(&M, &b)?;

        println!(
            "iter: {}, residual: {:.2E}, sol: {:#.4?}",
            gs.iter(),
            gs.residual(),
            x
        );

        Ok(())
    }
}
