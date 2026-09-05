use super::Matrix;

pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Matrix for DenseMatrix {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        self.rows
    }
}
