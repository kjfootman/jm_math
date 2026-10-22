mod cg;
mod gauss_seidel;
mod gmres;
mod hgmres;

pub use cg::ConjugateGradientBuilder;
pub use gauss_seidel::GaussSeidelBuilder;
pub use gmres::GmresBuilder;

use crate::{
    error::Error,
    linear_algebra::{matrix::csr::CSRMatrix, preconditioner as pc, vector::Vector},
};

pub trait MSolver {
    fn iter(&self) -> usize;
    fn residual(&self) -> f64;
    fn label(&self) -> &str;

    fn solve<'a, T: pc::Preconditioner>(
        &mut self,
        matrix: &'a CSRMatrix,
        preconditioner: &'a T,
        b: &Vector,
        x: &mut Vector,
    ) -> Result<(), Error>;
}
