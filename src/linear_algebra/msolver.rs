mod conjugate_gradient;
mod gauss_seidel;
mod gmres;

use crate::error::Error;
use crate::linear_algebra::{CSRMatrix, Vector};
pub use gauss_seidel::{GaussSeidel, GaussSeidelBuilder};

pub trait MSolver {
    fn iter(&self) -> usize;
    fn residual(&self) -> f64;
    fn solve(&mut self, matrix: &CSRMatrix, b: &Vector, x: &mut Vector) -> Result<(), Error>;
}
