use crate::{
    error::Error,
    linear_algebra::{CSRMatrix, MSolver, Matrix, Vector},
};

#[derive(Debug)]
pub struct ConjugateGradient {
    residual: f64,
    tolerance: f64,
    max_iter: usize,
    iter: usize,
    workspace: Workspace,
}

#[derive(Debug, Default)]
struct Workspace {
    pub Ap: Vector,
    pub p: Vector,
    pub r: Vector,
}

impl Workspace {
    pub fn set_workspace(&mut self, size: usize) {
        if self.r.len() < size {
            self.r.resize(size, 0.0);
        }
    }
}

impl MSolver for ConjugateGradient {
    fn iter(&self) -> usize {
        self.iter
    }

    fn residual(&self) -> f64 {
        self.residual
    }

    fn solve(&mut self, matrix: &CSRMatrix, b: &Vector, x: &mut Vector) -> Result<(), Error> {
        let (m, n) = (matrix.rows(), matrix.cols());
        let iter = &mut self.iter;
        let residual = &mut self.residual;
        let tol = self.tolerance;
        let max_iter = self.max_iter;
        let b_mag = b.magnitude()?;
        let A = matrix;
        let ia = A.row_ptr();
        let ja = A.col_indices();
        // let da = A
        //     .diag_ptr()
        //     .ok_or_else(|| Error::ValueError("Diagonal pointer is not initialized".into()))?;
        let aa = A.values();

        let workspace = &mut self.workspace;
        workspace.set_workspace(m);

        let Ap = &mut workspace.Ap;
        let r = &mut workspace.r;
        let mut p;

        // 1. calculate r0 = b - Ax0
        // 1.1 Ap = A * x0
        Ap.csr_spmv2(A, x)?;
        // 1.2 r0 = b - Ap = b - A * x0
        r.sub(b, Ap)?;

        // p0 = r0
        p = &mut *r;

        while *residual > tol && *iter < max_iter {
            *iter += 1;
        }

        p = &mut workspace.p;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::{CSRMatrix, csr};

    #[test]
    fn conjugate_gradient_test() -> Result<(), Error> {
        Ok(())
    }
}
