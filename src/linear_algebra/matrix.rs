mod csr;
mod dense;

pub use csr::CSRMatrix;

pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}
