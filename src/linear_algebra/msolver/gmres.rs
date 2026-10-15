use crate::{
    error::Error,
    linear_algebra::{
        givens::{self, compute_givens_rotation},
        matrix::{
            Matrix,
            csr::CSRMatrix,
            dense::{DenseMatrix, DenseMatrixArgs},
        },
        msolver::MSolver,
        preconditioner as pc, simd,
        vector::Vector,
    },
};

#[derive(Debug)]
pub struct Gmres {
    kind: String,
    residual: f64,
    tolerance: f64,
    max_iter: usize,
    restart: usize,
    iter: usize,
    workspace: Workspace,
}

impl Default for Gmres {
    fn default() -> Self {
        Gmres {
            kind: "GMRES".to_string(),
            residual: f64::MAX,
            tolerance: 1E-7,
            max_iter: 500,
            restart: 0,
            iter: 0,
            workspace: Workspace::default(),
        }
    }
}

impl MSolver for Gmres {
    fn iter(&self) -> usize {
        self.iter
    }

    fn residual(&self) -> f64 {
        self.residual
    }

    fn label(&self) -> &str {
        &self.kind
    }

    fn solve<'a, T: pc::Preconditioner>(
        &mut self,
        matrix: &'a CSRMatrix,
        preconditioner: &'a T,
        b: &Vector,
        x: &mut Vector,
    ) -> Result<(), Error> {
        let A = matrix;
        let M = preconditioner;
        let (m, n) = (A.rows(), A.cols());
        let iter = &mut self.iter;
        let residual = &mut self.residual;
        let restart = &mut self.restart;
        let tol = self.tolerance;
        let max_iter = self.max_iter;
        let b_mag = b.magnitude()?;

        // TODO: restart > m 일 경우 처리
        if *restart > m {
            log::error!("Restart number must be less then the number of {m}");
        }

        let workspace = &mut self.workspace;
        workspace.set_workspace(m, *restart);

        // let mut h;
        let r = &mut workspace.r;
        let w = &mut workspace.w;
        let g = &mut workspace.g;
        let V = &mut workspace.V;
        let H = &mut workspace.H;
        let cos = &mut workspace.cos;
        let sin = &mut workspace.sin;

        // 1. r0 = b - A * x0
        r.calc_residual(b, A, x)?;

        // 2. beta = |r0|
        let beta = r.magnitude()?;

        // 3. v0 = r0 / beta
        let mut v = &mut V[0];
        v.scale(beta.recip(), r);

        // 4. g = beta * e0
        g.fill(0.0);
        g[0] = b_mag;

        while *residual > tol && *iter < max_iter {
            for j in 0..*restart {
                // 4. w = A * M^-1 * v[j]
                v = &mut V[j];
                // 4.1 r = M^-1 * v[j] (결과를 r에 덮어쓰기)
                M.preconditioning(v, r)?;
                // 4.2 w = A * r
                w.csr_spmv2(A, r)?;

                // 5. Arnoldi process
                let h = H.column_mut(j);
                for i in 0..j + 1 {
                    let v = &V[i];
                    h[i] = w.dot(v)?;
                    w.scale_sub_assign(h[i], v)?;
                }

                h[j + 1] = w.magnitude()?;

                // H[j] 에 given's rotation 적용
                for i in 0..j {
                    let h0 = h[i];
                    let h1 = h[i + 1];
                    let c = cos[i];
                    let s = sin[i];

                    (h[i], h[i + 1]) = givens::apply_givens_rotation(h0, h1, c, s);
                }

                (cos[j], sin[j]) = compute_givens_rotation(h[j], h[j + 1]);

                let g0 = g[j];
                g[j] = g0 * cos[j];

                if h[j + 1].abs() < tol {
                    log::debug!("lucky breakdown");
                    *restart = j + 1;
                    break;
                }

                v = &mut V[j + 1];
                v.scale(h[j + 1].recip(), w);

                (h[j], h[j + 1]) = givens::apply_givens_rotation(h[j], h[j + 1], cos[j], sin[j]);
                g[j + 1] = -g0 * sin[j];
            }

            // solve the lower triangular

            // update solution

            *iter += 1;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct GmresBuilder {
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    restart: Option<usize>,
}

impl GmresBuilder {
    pub fn new() -> Self {
        Self {
            ..Default::default()
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

    pub fn with_restart(mut self, restart: usize) -> Self {
        self.restart = Some(restart);
        self
    }

    pub fn build(self) -> Result<Gmres, Error> {
        let restart = self.restart.ok_or_else(|| {
            Error::ValueError("with_restart method is missing for GMRES solver".into())
        })?;

        let gmres = Gmres {
            kind: format!("GMRES(restart)"),
            tolerance: self.tolerance.unwrap_or_default(),
            max_iter: self.max_iter.unwrap_or_default(),
            restart,
            ..Default::default()
        };

        Ok(gmres)
    }
}

#[derive(Debug, Default)]
struct Workspace {
    pub r: Vector,
    pub g: Vector,
    pub w: Vector,
    pub V: Vec<Vector>,
    pub H: DenseMatrix,
    pub cos: Vector,
    pub sin: Vector,
}

impl Workspace {
    pub fn set_workspace(&mut self, rows: usize, restart: usize) {
        if self.r.len() < rows {
            self.r.resize(rows, 0.0);
        }

        if self.w.len() < rows {
            self.w.resize(rows, 0.0);
        }

        if self.V.len() < restart + 1 {
            self.V.resize(restart + 1, Vector::new(rows));
        }

        if self.g.len() < restart + 1 {
            self.g.resize(restart + 1, 0.0);
        }

        let H_len = (restart + 1) * restart;
        if self.H.values().len() < H_len {
            self.H = DenseMatrix::from(DenseMatrixArgs {
                rows: restart + 1,
                cols: restart,
                values: vec![0.0; H_len],
            });
        }

        if self.cos.len() < restart {
            self.cos.resize(restart, 0.0);
        }

        if self.sin.len() < restart {
            self.sin.resize(restart, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::{
        matrix::csr::{self, CSRMatrix, CSRMatrixArgs},
        preconditioner as pc,
    };

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn gmres() -> Result<(), Error> {
        init();
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
        let mut x = Vector::new(rows);

        let mut gmres = GmresBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            .with_restart(4)
            .build()?;

        gmres.solve(&M, &pc::Idendity, &b, &mut x)?;

        Ok(())
    }
}
