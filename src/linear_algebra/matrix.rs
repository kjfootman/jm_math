pub mod csr;
mod dense;

pub use csr::{CSRMatrix, CSRMatrixArgs};

pub trait Matrix {
    /// Return the number of rows.
    fn rows(&self) -> usize;

    /// Return the number of columns.
    fn cols(&self) -> usize;
}
