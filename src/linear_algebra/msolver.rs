mod gauss_seidel;
mod gmres;

use crate::error::Error;
use crate::linear_algebra::{CSRMatrix, Vector};
pub use gauss_seidel::{GaussSeidel, GaussSeidelBuilder};

pub trait MSolver {
    fn solve(&mut self, matrix: &CSRMatrix, b: &Vector) -> Result<Vector, Error>;
}
