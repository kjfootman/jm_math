mod conjugate_gradient;
mod gauss_seidel;
mod gmres;

pub use conjugate_gradient::ConjugateGradientBuilder;
pub use gauss_seidel::GaussSeidelBuilder;

use crate::{
    error::Error,
    linear_algebra::{matrix::csr::CSRMatrix, preconditioner as pc, vector::Vector},
};

// #[derive(Debug)]
// enum MSolverKind {
//     GS(&'static str),
//     SOR(&'static str, f32),
//     CG(&'static str),
//     GMRES(&'static str, u16),
// }

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
