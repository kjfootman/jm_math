use crate::error::Error;

pub fn solve_upper_triangular(matrix: &[f64], source: &[f64], x: &mut [f64]) -> Result<(), Error> {
    let mut m = source.len();
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

        assert_eq!(x, vec![1.0; m]);

        Ok(())
    }
}
