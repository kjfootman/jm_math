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
    // pub fn new(
    //     rows: usize,
    //     cols: usize,
    //     row_ptr: Vec<usize>,
    //     col_indices: Vec<usize>,
    //     values: Vec<f64>,
    // ) -> Self {
    //     // let diag_ptr: Vec<Option<usize>> = row_ptr
    //     //     .par_windows(2)
    //     //     .enumerate()
    //     //     .map(|(row, range)| {
    //     //         let start = range[0];
    //     //         let end = range[1];

    //     //         if col_indices[start..end].contains(&row) {
    //     //             Some(row)
    //     //         } else {
    //     //             None
    //     //         }

    //     //         col_indices[start..end].iter().find(|j|)
    //     //     })
    //     //     .collect();

    //     // println!("{diag_ptr:#?}");

    //     CSRMatrix {
    //         rows,
    //         cols,
    //         row_ptr,
    //         diag_ptr: vec![],
    //         col_indices,
    //         values,
    //     }
    // }

    pub fn from_args(args: CSRMatrixArgs) -> CSRMatrix {
        CSRMatrix {
            rows: args.rows,
            ..Default::default()
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
