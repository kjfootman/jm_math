use crate::{
    error::Error,
    linear_algebra::{
        matrix::{Matrix, csr::CSRMatrix},
        msolver::MSolver,
        preconditioner as pc, simd,
        vector::Vector,
    },
};

#[derive(Debug)]
pub struct ConjugateGradient {
    ty: &'static str,
    residual: f64,
    tolerance: f64,
    max_iter: usize,
    iter: usize,
    workspace: Workspace,
}

#[derive(Debug, Default)]
pub struct ConjugateGradientBuilder {
    tolerance: Option<f64>,
    max_iter: Option<usize>,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        ConjugateGradient {
            ty: "Conjugate Gradient",
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
    pub z: Vector,
}

impl Workspace {
    pub fn set_workspace(&mut self, size: usize) {
        if self.Ap.len() < size {
            self.Ap.resize(size, 0.0);
        }

        if self.p.len() < size {
            self.p.resize(size, 0.0);
        }

        if self.r.len() < size {
            self.r.resize(size, 0.0);
        }

        if self.z.len() < size {
            self.z.resize(size, 0.0);
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

    fn label(&self) -> &str {
        self.ty
    }

    fn solve<'a, T: pc::Preconditioner>(
        &mut self,
        matrix: &'a CSRMatrix,
        preconditioner: &'a T,
        b: &Vector,
        x: &mut Vector,
    ) -> Result<(), Error> {
        let (m, n) = (matrix.rows(), matrix.cols());
        let iter = &mut self.iter;
        let residual = &mut self.residual;
        let tol = self.tolerance;
        let max_iter = self.max_iter;
        let b_mag = b.magnitude()?;
        let A = matrix;
        let M = preconditioner;

        *iter = 0;
        *residual = f64::MAX;

        if n != x.len() {
            return Err(Error::DimensionMismatch(
                "Matrix and solution vector dimensions do not match".to_string(),
            ));
        }

        let workspace = &mut self.workspace;
        workspace.set_workspace(m);

        let Ap = &mut workspace.Ap;
        let p = &mut workspace.p;
        let r = &mut workspace.r;
        let z = &mut workspace.z;

        // 1. calculate r0 = b - Ax0
        r.calc_residual(b, A, x)?;

        // 2. calculate z0 = M^-1 * r0;
        M.preconditioning(r, z)?;

        // rz = r0 * z0
        let mut rz = r.dot(z)?;

        // b 가 0벡터로 입력되거나 잔차가 0인 경우
        if b_mag == 0.0 || rz <= 0.0 {
            *residual = 0.0;
            *iter = 0;
            return Ok(());
        }

        // msolver 반복 사용을 위해 초기 residual, iter 설정
        *residual = rz.sqrt() / b_mag;
        *iter = 0;

        // 초기값이 이미 tol을 만족하는 경우
        if *residual < tol {
            return Ok(());
        }

        // 3. clone p0 as r0
        p.copy_from_slice(z);

        while *residual > tol && *iter < max_iter {
            // 1. calculate alpha
            // 1.2 Ap = A * p;
            Ap.csr_spmv2(A, p)?;
            // 1.3 alpha = r * r / Ap * p
            let alpha = rz / Ap.dot(p)?;

            // 2. x = x + alpha * p
            x.scale_add_assign(alpha, p)?;

            // 3. r(j + 1) = r(j) - alpha * Ap
            r.scale_add_assign(-alpha, Ap)?;

            // 4. z(j + 1) = M^-1 * r(j + 1)
            M.preconditioning(r, z)?;

            // new_rz = r(j + 1) * z(j + 1)
            let new_rz = r.dot(z)?;

            // 4. beta = r(j + 1) * z(j + 1) / r(j) * r(j)
            let beta = new_rz / rz;

            // 5. p = z + beta * p;
            let arch = simd::arch();
            arch.dispatch(|| {
                p.iter_mut().zip(z.iter()).for_each(|(p, z)| {
                    *p = beta * *p + z;
                });
            });

            // relative calculate residual
            // *residual = r.magnitude()?.abs() / b_mag;
            *residual = new_rz.sqrt() / b_mag;
            rz = new_rz;

            *iter += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear_algebra::preconditioner as pc, test_utils};
    use approx::assert_relative_eq;

    #[test]
    fn conjugate_gradient() -> Result<(), Error> {
        test_utils::init_env_logger();

        let M = test_utils::symm_4x4_csr()?;
        let b = test_utils::source_for_unit_solution(&M);
        let mut x = Vector::new(M.cols());

        let mut cg = ConjugateGradientBuilder::new()
            .with_max_iter(50)
            .with_tolerance(1E-7)
            .build();

        // cg.solve(&M, &pc::NoPreconditioner, &b, &mut x)?;
        cg.solve(&M, &pc::Jacobi::new(&M)?, &b, &mut x)?;

        // 수렴성 테스트
        assert!(
            cg.residual() < 1E-7,
            "{} failed to converge: iter={}, residual={:.2E}",
            cg.label(),
            cg.iter(),
            cg.residual()
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
