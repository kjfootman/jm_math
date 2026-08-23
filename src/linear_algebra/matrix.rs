pub mod csr;
mod dense;

pub use csr::{CSRMatrix, CSRMatrixArgs};

pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}
