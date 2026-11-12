use crate::{
    error::Error,
    linear_algebra::{
        givens, householder,
        matrix::{
            Matrix,
            csr::CSRMatrix,
            dense::{DenseMatrix, DenseMatrixArgs},
        },
        msolver::MSolver,
        preconditioner as pc, triangular,
        vector::Vector,
    },
};

#[derive(Debug)]
pub struct Hgmres {
    kind: String,
    residual: f64,
    tolerance: f64,
    max_iter: usize,
    restart: usize,
    iter: usize,
    workspace: Workspace,
}

impl Default for Hgmres {
    fn default() -> Self {
        Self {
            kind: "HGMRES".to_string(),
            residual: f64::MAX,
            tolerance: 1E-7,
            max_iter: 500,
            restart: 10,
            iter: 0,
            workspace: Workspace::default(),
        }
    }
}

impl MSolver for Hgmres {
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

        *iter = 0;
        *residual = f64::MAX;

        if *restart > m {
            log::warn!("Restart number must be less than {m}. Setting restart number to {m}.");
            *restart = m;
        }

        if n != x.len() {
            return Err(Error::DimensionMismatch(
                "Matrix and solution vector dimensions do not match".to_string(),
            ));
        }

        let workspace = &mut self.workspace;
        workspace.set_workspace(m, *restart);

        let v = &mut workspace.v;
        let w = &mut workspace.w;
        let z = &mut workspace.z;
        let g = &mut workspace.g;
        let y = &mut workspace.y;
        let U = &mut workspace.U;
        let V = &mut workspace.V;
        let H = &mut workspace.H;
        let cos = &mut workspace.cos;
        let sin = &mut workspace.sin;

        // let mut beta;
        while *residual > tol && *iter < max_iter {
            // 1. z = b - A * x0
            // 2. z = r0
            z.calc_residual(b, A, x)?;

            // g.fill(0.0);

            for j in 0..*restart + 1 {
                // 3. calculate householder vector u of z
                let u = &mut U[j];
                householder::calc_householder_vec(&z[j..m], &mut u[j..m]);

                // 4. calculate h[j-1] = P[j] * z;
                let h = if j != 0 {
                    // j > 0 조건일 경우만 H[j-1]에 저장
                    H.column_mut(j - 1)
                } else {
                    H.column_mut(j)
                };

                let sigma = -2.0 * z.dot(u)?;

                for i in 0..(j + 1).min(z.len()) {
                    h[i] = z[i] + sigma * u[i];
                }

                // 6. j != 0일때 H[j] 에 given's rotation 적용
                if j != 0 {
                    for i in 0..j - 1 {
                        let h0 = h[i];
                        let h1 = h[i + 1];
                        let c = cos[i];
                        let s = sin[i];
                        (h[i], h[i + 1]) = givens::apply_givens_rotation(h0, h1, c, s);
                    }

                    if h[j].abs() < tol {
                        log::debug!("lucky breakdown");
                        *restart = j;
                        break;
                    }

                    (cos[j - 1], sin[j - 1]) = givens::calc_givens_rotation(h[j - 1], h[j]);
                    (h[j - 1], h[j]) =
                        givens::apply_givens_rotation(h[j - 1], h[j], cos[j - 1], sin[j - 1]);

                    g[j] = -g[j - 1] * sin[j - 1];
                    g[j - 1] *= cos[j - 1];
                } else {
                    g[0] = h[0]
                }

                // if j > *restart + 1 {
                if j >= *restart {
                    println!("max dim: {}", iter);
                    break;
                }

                // 5. calculate v[j] = P[0] * P[1] * ... P[j] * e[j]
                v.fill(0.0);
                v[j] = 1.0;
                for i in (0..j + 1).rev() {
                    let u = &U[i];
                    // householder::transform(u, v);
                    householder::transform(&u[i..m], &mut v[i..m]);
                }

                //---------------------------------------------------------------------------------------//
                V[j].copy_from_slice(v);
                //---------------------------------------------------------------------------------------//

                // 7. calculate z[j] = P[j] * P[j-1] * ... * P[0] * A * v[j]
                M.preconditioning(v, w)?;
                z.csr_spmv2(A, w)?;

                // U.iter()
                //     .take(j + 1)
                //     .for_each(|u| householder::transform(u, z));
                U.iter().take(j + 1).enumerate().for_each(|(i, u)| {
                    householder::transform(&u[i..m], &mut z[i..m]);
                });
            }

            //---------------------------------------------------------------------------------------//
            println!("HGMRES---------");
            for i in 0..V.len() {
                let vi = &V[i];

                for j in 0..i {
                    print!("{:15}", "");
                }

                for j in i..V.len() {
                    let vj = &V[j];
                    let dot = vi.dot(vj)?;
                    print!("{:<15.4E}", dot);
                }
                println!();
            }
            //---------------------------------------------------------------------------------------//

            // solve the upper triangular
            triangular::solve_upper_triangular(H.values(), g, y)?;

            // update solution
            z.fill(0.0);
            v.fill(0.0);
            for i in (0..y.len()).rev() {
                let u = &U[i];
                // set unit vector
                // v.fill(0.0);
                v[i] = 1.0;

                // z = eta[j] * e[j] + z
                z.scale_add_assign(y[i], v)?;

                // z = P[j] * z
                // householder::transform(u, z);
                householder::transform(&u[i..m], &mut z[i..m]);
                v[i] = 0.0;
            }

            // x = x0 + z
            M.preconditioning(z, w)?;
            x.add_assign(w)?;

            *iter += 1;
            *residual = g[*restart].abs() / b_mag;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct HgmresBuilder {
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    restart: Option<usize>,
}

impl HgmresBuilder {
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

    pub fn build(self) -> Hgmres {
        let mut hgmres = Hgmres {
            tolerance: self.tolerance.unwrap_or_default(),
            max_iter: self.max_iter.unwrap_or_default(),
            ..Default::default()
        };

        if let Some(restart) = self.restart {
            hgmres.restart = restart
        }

        hgmres.kind = format!("HGMRES({})", hgmres.restart);

        hgmres
    }
}

#[derive(Debug, Default)]
struct Workspace {
    pub v: Vector,
    pub w: Vector,
    pub g: Vec<f64>,
    pub z: Vector,
    pub y: Vec<f64>,
    pub U: Vec<Vector>,
    pub V: Vec<Vector>,
    pub H: DenseMatrix,
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
}

impl Workspace {
    pub fn set_workspace(&mut self, rows: usize, restart: usize) {
        if self.v.len() < rows {
            self.v.resize(rows, 0.0);
        }

        if self.w.len() < rows {
            self.w.resize(rows, 0.0);
        }

        if self.z.len() < rows {
            self.z.resize(rows, 0.0);
        }

        if self.U.len() < restart + 1 {
            self.U.resize(restart + 1, Vector::new(rows));
        }

        if self.V.len() < restart + 1 {
            self.V.resize(restart, Vector::new(rows));
        }

        if self.g.len() < restart + 1 {
            self.g.resize(restart + 1, 0.0);
        }

        if self.y.len() < restart {
            self.y.resize(restart, 0.0);
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
    fn hgmres() -> Result<(), Error> {
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

        let mut hgmres = HgmresBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            // .with_restart(4)
            .with_restart(3)
            .build();

        hgmres.solve(&M, &pc::Idendity, &b, &mut x)?;

        log::debug!(
            "iter: {}, residual: {:.2E}",
            hgmres.iter(),
            hgmres.residual()
        );

        Ok(())
    }
}
