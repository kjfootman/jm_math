mod gauss_seidel;
mod gmres;

use crate::{CSRMatrix, Vector};
pub use gauss_seidel::GaussSeidel;

pub trait MSolver {
    fn solve(&mut self, matrix: &CSRMatrix, b: &Vector);
}
