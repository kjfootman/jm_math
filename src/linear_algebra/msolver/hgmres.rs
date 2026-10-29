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
        preconditioner as pc, simd, triangular,
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
            kind: "GMRES".to_string(),
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

        // let r = &mut workspace.r;
        let z = &mut workspace.z;
        let g = &mut workspace.g;
        let y = &mut workspace.y;
        let U = &mut workspace.U;
        let H = &mut workspace.H;
        let cos = &mut workspace.cos;
        let sin = &mut workspace.sin;

        let mut beta;

        while *residual > tol && *iter < max_iter {
            // 1. z = b - A * x0
            // 2. z = r0
            z.calc_residual(b, A, x)?;

            for j in 0..*restart + 2 {
                // 3. calculate householder vector u of z
                let u = &mut U[j];
                householder::calc_householder_vec(&z[j..m], &mut u[j..m]);

                // 4. calculate h[j-1] = P[j] * z;
                let h = H.column_mut(j);
                householder::transform(u, z, h);

                if j == 1 {
                    beta = h[0];
                }

                println!();
            }

            *iter += 1;
        }

        todo!()
    }
}

// fn calc_householder_vec(v: &[f64], inplace: &mut [f64]) {
//     debug_assert_eq!(v.len(), inplace.len());
//
//     let arch = simd::arch();
//     let mut magnitude = arch.dispatch(simd::VectorDot(v, v)).sqrt();
//
//     if magnitude == 0.0 {
//         return;
//     }
//
//     let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
//
//     inplace[0] = v[0] + sign * v[0] * magnitude;
//     magnitude = 2.0 * magnitude * (magnitude + v[0].abs());
//     magnitude = magnitude.sqrt();
//
//     arch.dispatch(simd::VectorScale(
//         &mut inplace[1..],
//         magnitude.recip(),
//         &v[1..],
//     ));
//
//     inplace[0] *= magnitude.recip();
// }

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

        hgmres.kind = format!("GMRES({})", hgmres.restart);

        hgmres
    }
}

#[derive(Debug, Default)]
struct Workspace {
    // pub r: Vector,
    pub g: Vec<f64>,
    pub z: Vector,
    pub y: Vec<f64>,
    pub U: Vec<Vector>,
    pub H: DenseMatrix,
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
}

impl Workspace {
    pub fn set_workspace(&mut self, rows: usize, restart: usize) {
        // if self.r.len() < rows {
        //     self.r.resize(rows, 0.0);
        // }

        if self.z.len() < rows {
            self.z.resize(rows, 0.0);
        }

        if self.U.len() < restart + 2 {
            self.U.resize(restart + 2, Vector::new(rows));
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
