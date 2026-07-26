use crate::Error;

#[derive(Debug)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CSRMatrix {
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        CSRMatrix {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl Matrix for CSRMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }
}

pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}
