use super::Matrix;

#[derive(Debug, Default)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    diag_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub struct CSRMatrixArgs {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub diag_ptr: Option<Vec<usize>>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CSRMatrix {
    pub fn from_args(args: CSRMatrixArgs) -> CSRMatrix {
        // todo: 대각 성분 포인터 찾기
        let diag_ptr = args.diag_ptr.unwrap_or_default();

        CSRMatrix {
            rows: args.rows,
            cols: args.cols,
            row_ptr: args.row_ptr,
            diag_ptr,
            col_indices: args.col_indices,
            values: args.values,
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

    // pub fn rows(&self) -> usize
}

impl Matrix for CSRMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }
}
