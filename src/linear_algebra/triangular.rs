use crate::{
    error::Error,
    linear_algebra::matrix::{
        Matrix,
        dense::{DenseMatrix, Layout},
    },
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

pub fn solve_upper(matrix: &DenseMatrix, source: &[f64], x: &mut [f64]) -> Result<(), Error> {
    let lda = matrix.rows();

    match matrix.layout() {
        Layout::RowMajor => {}
        Layout::ColMajor => {
            solve_upper_col_major(matrix.values(), lda, source, x);
        }
    }
    todo!()
}

pub fn solve_lower(matrix: &DenseMatrix, source: &[f64], x: &mut [f64]) -> Result<(), Error> {
    match matrix.layout() {
        Layout::RowMajor => {}
        Layout::ColMajor => {}
    }
    todo!()
}

fn solve_upper_row_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    todo!()
}

fn solve_upper_col_major(matrix: &[f64], lda: usize, source: &[f64], x: &mut [f64]) {
    let m = x.len();

    for i in (0..m).rev() {
        let mut sum = source[i];
        let diag = matrix[i + lda * i];
        for j in i + 1..m {
            sum -= matrix[i + lda * j] * x[j];
        }

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

    #[test]
    fn upper_triangular() -> Result<(), Error> {
        // 1.0,  2.0,  4.0,  7.0,
        // 0.0,  3.0,  5.0,  8.0,
        // 0.0,  0.0,  6.0,  9.0,
        // 0.0,  0.0,  0.0, 10.0,
        let M = vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 10.0,
        ];
        let b = vec![14.0, 16.0, 15.0, 10.0];
        let mut x = vec![0.0, 0.0, 0.0, 0.0];
        let m = b.len();

        if b.len() != x.len() {
            return Err(Error::DimensionMismatch(format!(
                "b: {}, x: {}",
                b.len(),
                x.len()
            )));
        }

        solve_upper_triangular(&M, &b, &mut x)?;
        solve_upper_col_major(&M, m, &b, &mut x);

        assert_eq!(x, vec![1.0; m]);

        Ok(())
    }

    #[test]
    fn lower_triangular() -> Result<(), Error> {
        // 1.0,  0.0,  0.0,  0.0,
        // 2.0,  5.0,  0.0,  0.0,
        // 3.0,  6.0,  8.0,  0.0,
        // 4.0,  7.0,  9.0, 10.0,
        let M = vec![
            1.0, 1.0, 1.0, 1.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 8.0, 0.0, 4.0, 7.0, 9.0, 10.0,
        ];
        let b = vec![1.0, 7.0, 17.0, 30.0];
        let mut x = vec![0.0, 0.0, 0.0, 0.0];
        let m = b.len();

        if b.len() != x.len() {
            return Err(Error::DimensionMismatch(format!(
                "b: {}, x: {}",
                b.len(),
                x.len()
            )));
        }

        solve_lower_triangular(&M, m, &b, &mut x)?;

        assert_eq!(x, vec![1.0; m]);

        Ok(())
    }
}
