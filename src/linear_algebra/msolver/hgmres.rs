use crate::{
    error::Error,
    linear_algebra::{
        givens, householder,
        matrix::{
            Matrix,
            csr::CSRMatrix,
            dense::{DenseMatrix, DenseMatrixArgs, DenseMatrixRef, Layout},
        },
        msolver::MSolver,
        preconditioner as pc, triangular,
        vector::{self, Vector},
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
        let mut restart = self.restart;
        let tol = self.tolerance;
        let max_iter = self.max_iter;
        let b_mag = b.magnitude()?;

        *iter = 0;
        *residual = f64::MAX;

        if restart > m {
            log::warn!("Restart number must be less than {m}. Setting restart number to {m}.");
            restart = m;
        }

        if n != x.len() {
            return Err(Error::DimensionMismatch(
                "Matrix and solution vector dimensions do not match".to_string(),
            ));
        }

        let workspace = &mut self.workspace;
        workspace.set_workspace(m, restart);

        let v = &mut workspace.v;
        let w = &mut workspace.w;
        let z = &mut workspace.z;
        let g = &mut workspace.g;
        let c = &mut workspace.c;
        let d = &mut workspace.d;
        let y = &mut workspace.y;
        let U = &mut workspace.U;
        let H = &mut workspace.H;
        let L = &mut workspace.L;
        let cos = &mut workspace.cos;
        let sin = &mut workspace.sin;

        // let mut beta;
        while *residual > tol && *iter < max_iter {
            // 1. z = b - A * x0
            // 2. z = r0
            z.calc_residual(b, A, x)?;

            // g.fill(0.0);

            for j in 0..restart + 1 {
                // 3. calculate householder vector u of z
                let u = &mut U[j][j..m];
                householder::calc_householder_vec(&z[j..m], u);

                // 4. calculate h[j-1] = P[j] * z;
                let h = if j != 0 {
                    // j > 0 조건일 경우만 H[j-1]에 저장
                    H.column_mut(j - 1)?
                } else {
                    H.column_mut(j)?
                };

                let u = &U[j];
                let sigma = -2.0 * vector::dot(&u[j..m], &z[j..m]);

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
                        #[cfg(debug_assertions)]
                        log::debug!("lucky breakdown");

                        restart = j;
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

                if j >= restart {
                    break;
                }

                // Lower matrix for variation method
                //-----------------------------------------------------------------------------//
                let l = L.row_mut(j)?;
                l[j] = 1.0;

                for i in 0..j {
                    let ui = &U[i];
                    let uj = &U[j];
                    l[i] = 2.0 * vector::dot(&ui[j..m], &uj[j..m]);
                }
                //-----------------------------------------------------------------------------//

                // 5. calculate v[j] = P[0] * P[1] * ... P[j] * e[j]
                v.fill(0.0);
                v[j] = 1.0;
                for i in 0..j + 1 {
                    c[i] = -2.0 * U[i][j];
                }

                // FIX: L일 때와 L^T일 때 triangular::solver_lower_triangular 솔버 구분 필요
                // L: row major 메모리 구조
                // - L^T 일 때 역방향 메인루프 (0..m).rev() -> 메모리 연속성 X
                // - L 일 때 정방향 메인루프 (0..m) -> 메모리 연속성 O
                triangular::solve_upper(&L.transpose(), c, &mut d[0..j + 1])?;

                // for i in 0..j + 1 {
                //     v.scale_add_assign(d[i], &U[i])?;
                // }
                v.scale_add_assign_many(&d[0..j + 1], &U[0..j + 1])?;

                // 7. calculate z[j] = P[j] * P[j-1] * ... * P[0] * A * M^-1 * v[j]
                M.preconditioning(v, w)?;
                z.csr_spmv2(A, w)?;

                // -2.0 * U ^ T * v
                for i in 0..j + 1 {
                    let u = &U[i];
                    c[i] = -2.0 * vector::dot(&u[i..m], &z[i..m]);
                }

                triangular::solve_lower(L, c, &mut d[0..j + 1])?;

                // for i in 0..j + 1 {
                //     z.scale_add_assign(d[i], &U[i])?;
                // }
                z.scale_add_assign_many(&d[0..j + 1], &U[0..j + 1])?;
            }

            triangular::solve_upper(H, g, y)?;

            //-----------------------------------------------------------------------------//
            // update solution
            z.fill(0.0);
            for i in (0..y.len()).rev() {
                let u = &U[i];

                z[i] += y[i];
                householder::transform(&u[i..m], &mut z[i..m]);
            }
            //-----------------------------------------------------------------------------//

            // x = x0 + M^-1 * z
            M.preconditioning(z, w)?;
            x.add_assign(w)?;

            *iter += 1;
            *residual = g[restart].abs() / b_mag;

            if *iter == max_iter {
                log::warn!("Maximum iteration exceeded");
            }

            #[cfg(debug_assertions)]
            log::debug!("iter: {}, residual: {:10.6E}", iter, residual);
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
    pub c: Vec<f64>,
    pub d: Vec<f64>,
    pub z: Vector,
    pub y: Vec<f64>,
    pub U: Vec<Vector>,
    pub H: DenseMatrix,
    pub L: DenseMatrix,
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
        // if self.U.len() < restart {
        //     self.U.resize(restart, Vector::new(rows));
        // }

        if self.g.len() < restart + 1 {
            self.g.resize(restart + 1, 0.0);
        }

        if self.c.len() < restart {
            self.c.resize(restart, 0.0);
        }

        if self.d.len() < restart {
            self.d.resize(restart, 0.0);
        }

        if self.y.len() < restart {
            self.y.resize(restart, 0.0);
        }

        let H_len = (restart + 1) * restart;
        if self.H.values().len() < H_len {
            self.H = DenseMatrix::from(DenseMatrixArgs {
                rows: restart + 1,
                cols: restart,
                layout: Layout::ColMajor,
                values: vec![0.0; H_len],
            });
        }

        let L_len = restart * restart;
        if self.L.values().len() < L_len {
            self.L = DenseMatrix::from(DenseMatrixArgs {
                rows: restart,
                cols: restart,
                layout: Layout::RowMajor,
                values: vec![0.0; L_len],
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
    use crate::{linear_algebra::preconditioner as pc, test_utils};
    use approx::assert_relative_eq;

    #[test]
    fn hh_gmres() -> Result<(), Error> {
        test_utils::init_env_logger();

        let M = test_utils::symm_4x4_csr()?;
        let b = test_utils::source_for_unit_solution(&M);
        let mut x = Vector::new(M.cols());

        let mut hgmres = HgmresBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            // restart 최대 4까지 가능 --> basis 벡터 4개
            // .with_restart(4)
            .with_restart(3)
            .build();

        // hgmres 솔버
        hgmres.solve(&M, &pc::Idendity, &b, &mut x)?;

        // 수렴성 테스트
        assert!(
            hgmres.residual() < 1E-7,
            "{} failed to converge: iter={}, residual={:.2E}",
            hgmres.label(),
            hgmres.iter(),
            hgmres.residual()
        );

        // 수치해와 해석해의 근사 일치 검증
        assert_relative_eq!(
            x,
            Vector::from(vec![1.0; 4]),
            epsilon = 1E-10,
            max_relative = 1E-6
        );

        Ok(())
    }
}
