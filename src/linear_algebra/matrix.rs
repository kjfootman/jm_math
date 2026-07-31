mod csr;
mod dense;

pub use csr::{CSRMatrix, CSRMatrixArgs};

pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_csr_test() {
        // CSRMatrix 선언
        let csr = CSRMatrix::from_args(CSRMatrixArgs {
            rows: 4,
            cols: 4,
            row_ptr: vec![0, 3, 6, 8, 9],
            diag_ptr: None,
            col_indices: vec![0, 2, 3, 0, 1, 3, 2, 3, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        });
    }
}
