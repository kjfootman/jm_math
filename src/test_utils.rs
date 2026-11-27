use crate::{
    error::Error,
    linear_algebra::{
        matrix::csr::{self, CSRMatrix, CSRMatrixArgs},
        vector::Vector,
    },
};
use approx::{AbsDiffEq, RelativeEq};

/// initializes the env_logger
pub(crate) fn init_env_logger() {
    let _ = env_logger::builder()
        .format_timestamp_secs()
        .is_test(true)
        .try_init();
}

/// returns symmetric 4x4 test matrix
pub(crate) fn symm_4x4_csr() -> Result<CSRMatrix, Error> {
    //  1  0  2  3
    //  0  2  1  0
    //  2  1  3  1
    //  3  0  1  4
    let (rows, cols) = (4, 4);
    let row_ptr = vec![0, 3, 5, 9, 12];
    let col_indices = vec![0, 2, 3, 1, 2, 0, 1, 2, 3, 0, 2, 3];
    let diag_ptr = csr::find_diag_ptr(&row_ptr, &col_indices).ok();
    #[rustfmt::skip]
    let values = vec![
        1.0, 2.0, 3.0,
        2.0, 1.0, 2.0,
        1.0, 3.0, 1.0,
        3.0, 1.0, 4.0
    ];

    let M = CSRMatrix::from_args(CSRMatrixArgs {
        rows,
        cols,
        row_ptr,
        diag_ptr,
        col_indices,
        values,
    });

    Ok(M)
}

/// Computes `b = Ax` for the unit solution vector `x = [1, ..., 1]^T`.
pub(crate) fn source_for_unit_solution(csr: &CSRMatrix) -> Vector {
    let ia = csr.row_ptr();
    let aa = csr.values();

    let arr = ia
        .windows(2)
        .map(|range| {
            let start = range[0] as usize;
            let end = range[1] as usize;

            aa[start..end].iter().sum::<f64>()
        })
        .collect::<Vec<_>>();

    Vector::from(arr)
}

impl AbsDiffEq for Vector {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        !self.abs_diff_eq(other, epsilon)
    }
}

impl RelativeEq for Vector {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        !self.relative_eq(other, epsilon, max_relative)
    }
}
