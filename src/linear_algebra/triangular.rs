use crate::{
    error::Error,
    linear_algebra::matrix::dense::{DenseMatrixRef, Layout},
};

pub fn solve_upper_triangular(matrix: &[f64], source: &[f64], x: &mut [f64]) -> Result<(), Error> {
    let mut m = source.len();
    // Leading Dimension Array of A
    let lda = m;

    if x.len() != source.len() {
        m -= 1;
    }

    #[cfg(debug_assertions)]
    {
        assert_eq!(
            matrix.len(),
            lda * m,
            "Matrix dimensions do not match source length."
        );
    }

    for i in (0..m).rev() {
        let mut sum = source[i];
        let diag = matrix[i + lda * i];

        sum -= (i + 1..m).map(|j| matrix[i + lda * j] * x[j]).sum::<f64>();

        if diag.abs() < 1e-14 {
            return Err(Error::MissingDiagonal(i));
        }

        x[i] = sum / diag;
    }

    Ok(())
}

pub fn solve_upper<T: DenseMatrixRef>(
    matrix: &T,
    source: &[f64],
    x: &mut [f64],
) -> Result<(), Error> {
    // Leading Dimension Array of A
    match matrix.layout() {
        Layout::RowMajor => {
            let lda = matrix.cols();
            solve_upper_row_major(matrix.values(), lda, source, x);
        }
        Layout::ColMajor => {
            let lda = matrix.rows();
            solve_upper_col_major(matrix.values(), lda, source, x);
        }
    }

    Ok(())
}

pub fn solve_lower<T: DenseMatrixRef>(
    matrix: &T,
    source: &[f64],
    x: &mut [f64],
) -> Result<(), Error> {
    match matrix.layout() {
        Layout::RowMajor => {
            let lda = matrix.cols();
            solve_lower_row_major(matrix.values(), lda, source, x);
        }
        Layout::ColMajor => {
            let lda = matrix.rows();
            solve_lower_col_major(matrix.values(), lda, source, x);
        }
    }

    Ok(())
}

fn solve_upper_row_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    let m = x.len();

    for i in (0..m).rev() {
        let row = &matrix[lda * i..lda * (i + 1)];
        let mut sum = source[i];

        for j in i + 1..m {
            sum -= row[j] * x[j];
        }

        x[i] = sum / row[i];
    }
}

fn solve_upper_col_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    let m = x.len();

    for i in (0..m).rev() {
        let mut sum = source[i];
        let diag = matrix[i + lda * i];

        // for j in i + 1..m {
        //     sum -= matrix[i + lda * j] * x[j];
        // }
        sum -= (i + 1..m).map(|j| matrix[i + lda * j] * x[j]).sum::<f64>();

        x[i] = sum / diag;
    }
}

fn solve_lower_row_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    let m = x.len();

    for i in 0..m {
        let row = &matrix[lda * i..lda * (i + 1)];
        let mut sum = source[i];

        for j in 0..i {
            sum -= row[j] * x[j];
        }

        x[i] = sum / row[i];
    }
}

fn solve_lower_col_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    let m = x.len();

    log::debug!("lda: {lda}");
    for i in 0..m {
        let mut sum = source[i];
        let diag = matrix[i + lda * i];

        // for j in 0..i {
        //     sum -= matrix[i + lda * j] * x[j];
        // }
        sum -= (0..i).map(|j| matrix[i + lda * j] * x[j]).sum::<f64>();

        x[i] = sum / diag;
    }
}

pub fn solve_lower_triangular(
    matrix: &[f64],
    lda: usize,
    source: &[f64],
    x: &mut [f64],
) -> Result<(), Error> {
    let m = source.len();

    for i in 0..m {
        let row = &matrix[i * lda..(i + 1) * lda];
        let mut sum = source[i];

        for j in 0..i {
            sum -= row[j] * x[j];
        }

        x[i] = sum / row[i];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::{
        matrix::dense::{DenseMatrix, DenseMatrixArgs, Layout},
        triangular::solve_upper,
    };

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn solve_upper_row_major() -> Result<(), Error> {
        // 1  2  4   7 | 1 | 14
        // 0  3  5   8 | 1 | 16
        // 0  0  6   9 | 1 | 15
        // 0  0  0  10 | 1 | 10
        // 0  0  0   0 |   |  0
        let (rows, cols) = (5, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::RowMajor,
            #[rustfmt::skip]
            // row-major DenseMatrix
            values: vec![
                1.0, 2.0, 4.0,  7.0,
                0.0, 3.0, 5.0,  8.0,
                0.0, 0.0, 6.0,  9.0,
                0.0, 0.0, 0.0, 10.0,
                0.0, 0.0, 0.0,  0.0,
            ],
        });
        let b = vec![14.0, 16.0, 15.0, 10.0, 0.0];
        let mut x = vec![0.0; cols];

        solve_upper(&M, &b, &mut x)?;

        assert_eq!(x, vec![1.0; cols]);

        Ok(())
    }

    #[test]
    fn solve_upper_transpose_row_major() -> Result<(), Error> {
        // [ 1  0  0   0 ] ^ T | 1 | 14
        // [ 2  3  0   0 ]     | 1 | 16
        // [ 4  5  6   0 ]     | 1 | 15
        // [ 7  8  9  10 ]     | 1 | 10

        let (rows, cols) = (4, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::RowMajor,
            #[rustfmt::skip]
            // row-major DenseMatrix
            values: vec![
                1.0, 0.0, 0.0,  0.0,
                2.0, 3.0, 0.0,  0.0,
                4.0, 5.0, 6.0,  0.0,
                7.0, 8.0, 9.0, 10.0,
            ],
        });
        let b = vec![14.0, 16.0, 15.0, 10.0];
        let mut x = vec![0.0; cols];

        let transpose = M.transpose();
        solve_upper(&transpose, &b, &mut x)?;

        assert_eq!(x, vec![1.0; cols]);

        Ok(())
    }

    #[test]
    fn solve_upper_col_major() -> Result<(), Error> {
        // 1  2  4   7 | 1 | 14
        // 0  3  5   8 | 1 | 16
        // 0  0  6   9 | 1 | 15
        // 0  0  0  10 | 1 | 10
        // 0  0  0   0 |   |  0

        let (rows, cols) = (5, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::ColMajor,
            #[rustfmt::skip]
            // column-major DenseMatrix
            values: vec![
                1.0, 0.0, 0.0,  0.0, 0.0,
                2.0, 3.0, 0.0,  0.0, 0.0,
                4.0, 5.0, 6.0,  0.0, 0.0,
                7.0, 8.0, 9.0, 10.0, 0.0,
            ],
        });
        let b = vec![14.0, 16.0, 15.0, 10.0, 0.0];
        let mut x = vec![0.0; cols];

        solve_upper(&M, &b, &mut x)?;

        assert_eq!(x, vec![1.0; cols]);

        Ok(())
    }

    #[test]
    fn solve_lower_row_major() -> Result<(), Error> {
        // 1  0  0   0 | 1 |  1
        // 2  3  0   0 | 1 |  5
        // 4  5  6   0 | 1 | 15
        // 7  8  9  10 | 1 | 34
        // 0  0  0   0 |   |  0
        let (rows, cols) = (5, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::RowMajor,
            #[rustfmt::skip]
            // row-major DenseMatrix
            values: vec![
                1.0, 0.0, 0.0,  0.0,
                2.0, 3.0, 0.0,  0.0,
                4.0, 5.0, 6.0,  0.0,
                7.0, 8.0, 9.0, 10.0,
                0.0, 0.0, 0.0,  0.0,
            ],
        });
        let b = vec![1.0, 5.0, 15.0, 34.0, 0.0];
        let mut x = vec![0.0; cols];

        solve_lower(&M, &b, &mut x)?;

        assert_eq!(x, vec![1.0; cols]);

        //--------------------------------------------------------------------------------//
        let (rows, cols) = (4, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::RowMajor,
            #[rustfmt::skip]
            values: vec![
                1.0, 0.0, 0.0, 0.0,
                2.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ],
        });
        let b = vec![1.0, 5.0, 0.0, 0.0];
        let mut x = vec![0.0; rows];

        solve_lower(&M, &b, &mut x[0..2])?;
        assert_eq!(x, vec![1.0, 1.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn solve_lower_col_major() -> Result<(), Error> {
        // 1  0  0   0 | 1 |  1
        // 2  3  0   0 | 1 |  5
        // 4  5  6   0 | 1 | 15
        // 7  8  9  10 | 1 | 34
        // 0  0  0   0 |   |  0
        init();

        let (rows, cols) = (5, 4);
        let M = DenseMatrix::from(DenseMatrixArgs {
            rows,
            cols,
            layout: Layout::ColMajor,
            #[rustfmt::skip]
            // column-major DenseMatrix
            values: vec![
                1.0, 2.0, 4.0,  7.0, 0.0,
                0.0, 3.0, 5.0,  8.0, 0.0,
                0.0, 0.0, 6.0,  9.0, 0.0,
                0.0, 0.0, 0.0, 10.0, 0.0,
            ],
        });
        let b = vec![1.0, 5.0, 15.0, 34.0, 0.0];
        let mut x = vec![0.0; cols];

        solve_lower(&M, &b, &mut x)?;

        assert_eq!(x, vec![1.0; cols]);

        Ok(())
    }
}
