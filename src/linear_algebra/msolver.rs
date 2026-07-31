mod gauss_seidel;
mod gmres;

use crate::linear_algebra::{CSRMatrix, Vector};
pub use gauss_seidel::GaussSeidel;

pub trait MSolver {
    fn solve(&self, matrix: &CSRMatrix, b: &Vector);
}
