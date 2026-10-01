pub mod conjugate_gradient;
pub mod gauss_seidel;
pub mod gmres;

use crate::{
    error::Error,
    linear_algebra::{matrix::csr::CSRMatrix, preconditioner as pc, vector::Vector},
};

pub trait MSolver {
    fn iter(&self) -> usize;
    fn residual(&self) -> f64;
    fn solve<'a, T: pc::Preconditioner>(
        &mut self,
        matrix: &'a CSRMatrix,
        preconditioner: &'a T,
        b: &Vector,
        x: &mut Vector,
    ) -> Result<(), Error>;
}
