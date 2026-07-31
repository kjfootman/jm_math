use super::Matrix;

#[derive(Debug)]
pub struct CSRMatrix {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    diag_ptr: Vec<usize>,
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
        // let diag_ptr: Vec<Option<usize>> = row_ptr
        //     .par_windows(2)
        //     .enumerate()
        //     .map(|(row, range)| {
        //         let start = range[0];
        //         let end = range[1];

        //         if col_indices[start..end].contains(&row) {
        //             Some(row)
        //         } else {
        //             None
        //         }

        //         col_indices[start..end].iter().find(|j|)
        //     })
        //     .collect();

        // println!("{diag_ptr:#?}");

        CSRMatrix {
            rows,
            cols,
            row_ptr,
            diag_ptr: vec![],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_test() {
        let (rows, cols) = (4, 4);
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let values = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let matrix = CSRMatrix::new(rows, cols, row_ptr, col_indices, values);
    }
}
