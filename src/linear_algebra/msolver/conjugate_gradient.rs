use crate::{
    error::Error,
    linear_algebra::{CSRMatrix, MSolver, Matrix, Vector, simd},
};

#[derive(Debug)]
pub struct ConjugateGradient {
    residual: f64,
    tolerance: f64,
    max_iter: usize,
    iter: usize,
    workspace: Workspace,
}

#[derive(Debug)]
pub struct ConjugateGradientBuilder {
    tolerance: Option<f64>,
    max_iter: Option<usize>,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        ConjugateGradient {
            residual: f64::MAX,
            tolerance: 1E-7,
            max_iter: 500,
            iter: 0,
            workspace: Workspace::default(),
        }
    }
}

#[derive(Debug, Default)]
struct Workspace {
    pub Ap: Vector,
    pub p: Vector,
    pub r: Vector,
}

impl Workspace {
    pub fn set_workspace(&mut self, size: usize) {
        if self.Ap.len() < size {
            self.Ap.resize(size, 0.0);
        }

        if self.r.len() < size {
            self.r.resize(size, 0.0);
        }

        if self.p.len() < size {
            self.p.resize(size, 0.0);
        }
    }
}

impl ConjugateGradientBuilder {
    pub fn new() -> Self {
        Self {
            tolerance: None,
            max_iter: None,
        }
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    pub fn build(self) -> ConjugateGradient {
        ConjugateGradient {
            tolerance: self.tolerance.unwrap_or_default(),
            max_iter: self.max_iter.unwrap_or_default(),
            ..Default::default()
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
        let mut r_squared;

        let workspace = &mut self.workspace;
        workspace.set_workspace(m);

        let Ap = &mut workspace.Ap;
        let r = &mut workspace.r;
        let p = &mut workspace.p;

        // 1. calculate r0 = b - Ax0
        // 1.1 Ap = A * x0
        Ap.csr_spmv2(A, x)?;
        // 1.2 r0 = b - Ap = b - A * x0
        // r.sub(b, Ap)?;
        r.calc_residual(b, A, x)?;

        // p0 = r0
        *p = r.clone();

        while *residual > tol && *iter < max_iter {
            // 1. alpha = r * r / Ap * p
            r_squared = r.dot(r)?;
            // 1.1 Ap = A * p;
            Ap.csr_spmv2(A, p)?;
            let alpha = r_squared / Ap.dot(p)?;

            // 2. x = x + alpha * p
            // p.scale_assign(alpha);
            // x.add_assign(p)?;
            x.scale_add_assign(alpha, p)?;

            // 3. r = r - alpha * Ap
            // Ap.scale_assign(alpha);
            // r.sub_assign(Ap)?;
            r.scale_add_assign(-alpha, Ap)?;

            // 4. beta = r(j + 1) * r(j + 1) / r(j) * r(j)
            let beta = r.dot(r)? / r_squared;

            // 5. p = r + beta * p;
            // Ap.scale(beta / alpha, p);

            // Ap.scale(beta, p);
            // p.add(r, Ap)?;

            let arch = simd::arch();
            arch.dispatch(|| {
                p.iter_mut().zip(r.iter()).for_each(|(p, r)| {
                    *p = beta * *p + r;
                });
            });

            // relative calculate residual
            *residual = r.magnitude()?.abs() / b_mag;

            *iter += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::{CSRMatrix, CSRMatrixArgs, csr};

    #[test]
    fn conjugate_gradient() -> Result<(), Error> {
        let (rows, cols) = (4, 4);
        let row_ptr = vec![0, 3, 5, 9, 12];
        let col_indices = vec![0, 2, 3, 1, 2, 0, 1, 2, 3, 0, 2, 3];
        let diag_ptr = csr::find_diag_ptr(&row_ptr, &col_indices).ok();
        let values = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 4.0];

        let M = CSRMatrix::from_args(CSRMatrixArgs {
            rows,
            cols,
            row_ptr,
            diag_ptr,
            col_indices,
            values,
        });
        let b = Vector::from(vec![6.0, 3.0, 7.0, 8.0]);

        let mut cg = ConjugateGradientBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            .build();
        let mut x = Vector::new(rows);
        cg.solve(&M, &b, &mut x)?;

        println!(
            "iter: {}, residual: {:.2E}, sol: {:#.4?}",
            cg.iter(),
            cg.residual(),
            x
        );

        Ok(())
    }
}
