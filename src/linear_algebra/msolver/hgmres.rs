use crate::{
    error::Error,
    linear_algebra::{
        givens,
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
        todo!()
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
}

#[derive(Debug, Default)]
struct Workspace {
    pub r: Vector,
    pub g: Vec<f64>,
    pub w: Vector,
    pub y: Vec<f64>,
    pub U: Vec<Vector>,
    pub H: DenseMatrix,
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
}
