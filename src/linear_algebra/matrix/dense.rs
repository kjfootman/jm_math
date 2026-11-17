use super::Matrix;

#[derive(Debug, Default)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    layout: Layout,
    values: Vec<f64>,
}

#[derive(Debug, Default)]
pub enum Layout {
    #[default]
    RowMajor,
    ColMajor,
}

impl DenseMatrix {
    pub fn column(&self, idx: usize) -> &[f64] {
        let m = self.rows;
        &self.values[idx * m..(idx + 1) * m]
    }

    pub fn column_mut(&mut self, idx: usize) -> &mut [f64] {
        let m = self.rows;
        &mut self.values[idx * m..(idx + 1) * m]
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl From<DenseMatrixArgs> for DenseMatrix {
    fn from(args: DenseMatrixArgs) -> Self {
        Self {
            rows: args.rows,
            cols: args.cols,
            layout: args.layout,
            values: args.values,
        }
    }
}

// impl Default for DenseMatrix {
//     fn default() -> Self {
//         Self {
//             rows: 0,
//             cols: 0,
//             layout: Layout::ColMajor,
//             values: vec![],
//         }
//     }
// }

impl Matrix for DenseMatrix {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Debug)]
pub struct DenseMatrixArgs {
    pub rows: usize,
    pub cols: usize,
    pub layout: Layout,
    pub values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn denseMatrix() -> Result<(), Error> {
        let args = DenseMatrixArgs {
            rows: 5,
            cols: 4,
            layout: Layout::ColMajor,
            values: vec![
                1.0, 5.0, 0.0, 0.0, 0.0, 2.0, 6.0, 9.0, 0.0, 0.0, 3.0, 7.0, 10.0, 12.0, 0.0, 4.0,
                8.0, 11.0, 13.0, 14.0,
            ],
        };
        // 1.0  2.0   3.0   4.0
        // 5.0  6.0   7.0   8.0
        // 0.0  9.0  10.0  11.0
        // 0.0  0.0  12.0  13.0
        // 0.0  0.0   0.0  14.0
        let A = DenseMatrix::from(args);

        assert_eq!(A.column(0), &[1.0, 5.0, 0.0, 0.0, 0.0]);
        assert_eq!(A.column(1), &[2.0, 6.0, 9.0, 0.0, 0.0]);
        assert_eq!(A.column(2), &[3.0, 7.0, 10.0, 12.0, 0.0]);
        assert_eq!(A.column(3), &[4.0, 8.0, 11.0, 13.0, 14.0]);

        Ok(())
    }
}
