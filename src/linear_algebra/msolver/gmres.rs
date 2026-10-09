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
        let restart = self.restart;
        let tol = self.tolerance;
        let max_iter = self.max_iter;
        let b_mag = b.magnitude()?;

        let workspace = &mut self.workspace;
        workspace.set_workspace(m, restart);

        let r = &mut workspace.r;
        let w = &mut workspace.w;
        let V = &mut workspace.V;
        let H = &mut workspace.H;
        let cos = &mut workspace.cos;
        let sin = &mut workspace.sin;

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
    pub w: Vector,
    pub V: Vec<Vector>,
    pub H: Vec<f64>,
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

        if self.V.len() < restart {
            self.V.resize(restart, Vector::new(rows));
        }

        let H_len = rows * (rows + 1);
        if self.H.len() < H_len {
            self.H.resize(H_len, 0.0);
        }

        if self.cos.len() < restart {
            self.cos.resize(restart, 0.0);
        }

        if self.sin.len() < restart {
            self.cos.resize(restart, 0.0);
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

    #[test]
    fn gmres() -> Result<(), Error> {
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
