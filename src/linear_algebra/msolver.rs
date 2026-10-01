mod conjugate_gradient;
mod gauss_seidel;
mod gmres;

use crate::error::Error;
use crate::linear_algebra::preconditioner::Preconditioner;
use crate::linear_algebra::{CSRMatrix, Vector};
pub use conjugate_gradient::ConjugateGradientBuilder;
pub use gauss_seidel::GaussSeidelBuilder;

pub trait MSolver {
    fn iter(&self) -> usize;
    fn residual(&self) -> f64;
    fn solve<'a, T: Preconditioner>(
        &mut self,
        matrix: &'a CSRMatrix,
        pc: &'a T,
        b: &Vector,
        x: &mut Vector,
    ) -> Result<(), Error>;
}
