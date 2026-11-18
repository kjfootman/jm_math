use super::Matrix;
use crate::error::Error;

#[derive(Debug, Default)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    layout: Layout,
    values: Vec<f64>,
}

#[derive(Debug, Default, PartialEq)]
pub enum Layout {
    #[default]
    RowMajor,
    ColMajor,
}

#[derive(Debug)]
pub struct DenseMatrixView<'a> {
    rows: usize,
    cols: usize,
    layout: Layout,
    values: &'a [f64],
}

pub trait DenseMatrixRef: Matrix {
    fn layout(&self) -> &Layout;
    fn values(&self) -> &[f64];
}

impl DenseMatrix {
    pub fn column(&self, idx: usize) -> Result<&[f64], Error> {
        let m = self.rows;

        if self.layout != Layout::ColMajor {
            return Err(Error::ValueError(
                "Column extraction is not supported for a row-major Densematrix".into(),
            ));
        }

        Ok(&self.values[idx * m..(idx + 1) * m])
    }

    pub fn column_mut(&mut self, idx: usize) -> Result<&mut [f64], Error> {
        let m = self.rows;

        if self.layout != Layout::ColMajor {
            return Err(Error::ValueError(
                "Column extraction is not supported for a row-major Densematrix".into(),
            ));
        }

        Ok(&mut self.values[idx * m..(idx + 1) * m])
    }

    pub fn row(&self, idx: usize) -> Result<&[f64], Error> {
        let n = self.cols;

        if self.layout != Layout::RowMajor {
            return Err(Error::ValueError(
                "Row extraction is not supported for a col-major Densematrix".into(),
            ));
        }

        Ok(&self.values[idx * n..(idx + 1) * n])
    }

    pub fn row_mut(&mut self, idx: usize) -> Result<&mut [f64], Error> {
        let n = self.cols;

        if self.layout != Layout::RowMajor {
            return Err(Error::ValueError(
                "Row extraction is not supported for a col-major Densematrix".into(),
            ));
        }

        Ok(&mut self.values[idx * n..(idx + 1) * n])
    }

    pub fn transpose(&self) -> DenseMatrixView<'_> {
        DenseMatrixView {
            rows: self.cols,
            cols: self.rows,
            layout: match self.layout {
                Layout::RowMajor => Layout::ColMajor,
                Layout::ColMajor => Layout::RowMajor,
            },
            values: &self.values,
        }
    }
}

impl DenseMatrixRef for DenseMatrix {
    #[inline]
    fn values(&self) -> &[f64] {
        &self.values
    }

    #[inline]
    fn layout(&self) -> &Layout {
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

impl Matrix for DenseMatrix {
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
}

impl DenseMatrixRef for DenseMatrixView<'_> {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn values(&self) -> &[f64] {
        self.values
    }
}

impl Matrix for DenseMatrixView<'_> {
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

        assert_eq!(A.column(0)?, &[1.0, 5.0, 0.0, 0.0, 0.0]);
        assert_eq!(A.column(1)?, &[2.0, 6.0, 9.0, 0.0, 0.0]);
        assert_eq!(A.column(2)?, &[3.0, 7.0, 10.0, 12.0, 0.0]);
        assert_eq!(A.column(3)?, &[4.0, 8.0, 11.0, 13.0, 14.0]);

        Ok(())
    }

    #[test]
    fn get_row() -> Result<(), Error> {
        let (rows, cols) = (3, 3);
        let A = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::RowMajor,
            #[rustfmt::skip]
            values: vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            ],
        });

        let row = A.row(0)?;
        assert_eq!(row, vec![1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn get_column() -> Result<(), Error> {
        let (rows, cols) = (3, 3);
        let A = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::ColMajor,
            #[rustfmt::skip]
            values: vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            ],
        });

        let col = A.column(0)?;
        assert_eq!(col, vec![1.0, 2.0, 3.0]);

        Ok(())
    }
}
