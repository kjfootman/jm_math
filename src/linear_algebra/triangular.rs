use crate::error::Error;

pub fn solve_upper_triangular(matrix: &[f64], source: &[f64], x: &mut [f64]) -> Result<(), Error> {
    let m = source.len();

    for i in (0..m).rev() {
        let mut sum = source[i];
        let diag = matrix[i * (m + 2)];

        sum -= (i + 1..m)
            .map(|j| matrix[i + (m + 1) * j] * x[j])
            .sum::<f64>();

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
    fn triangular() -> Result<(), Error> {
        // 1.0,  0.0,  0.0,  0.0,
        // 2.0,  3.0,  0.0,  0.0,
        // 4.0,  5.0,  6.0,  0.0,
        // 7.0,  8.0,  9.0, 10.0,
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
