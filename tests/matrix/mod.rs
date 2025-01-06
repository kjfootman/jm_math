use jm_math::linear_algebra::Matrix;
use jm_math::JmError;
use std::error::Error;

pub fn get_matrix(case: usize) -> Result<Matrix, Box<dyn Error>> {
    match case {
        0 => {
            // 4 x 4 test matrix
            Matrix::from_rows([
                [1.0, 0.0, 2.0, 3.0],
                [4.0, 5.0, 0.0, 6.0],
                [0.0, 0.0, 7.0, 8.0],
                [0.0, 0.0, 0.0, 9.0],
            ])
        }
        1 => {
            // test for lower matrix
            let IA = [0, 1, 3, 5, 7, 9, 12, 15, 18, 20, 23, 26, 29];
            let JA = [
                0, 0, 1, 1, 2, 2, 3, 0, 4, 1, 4, 5, 2, 5, 6, 3, 6, 7, 4, 8, 5, 8, 9, 6, 9, 10, 7,
                10, 11,
            ];
            let AA = [
                3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0,
                1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0,
            ];

            Matrix::from_csr(IA, JA, AA)
        }
        2 => {
            // from matrix market
            // unsymmetric
            // 760 x 760
            Matrix::from_matrix_market(
                "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/fs_760_1.mtx",
            )
        }
        3 => {
            // from matrix market
            // symmetric
            // Cannot be solved without preconditioner
            // Even cannot be solved by CG with Jacobi preconditioner
            // 2003 x 2003
            Matrix::from_matrix_market(
                "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/bcsstk13.mtx",
            )
        }
        4 => {
            // unsymmetric
            // 765 x 765
            // Cannot solve without ILU0 preconditioner
            Matrix::from_matrix_market("matrix_makert/mcfe.mtx")
        }
        5 => {
            // test for Gauss Eliminatiton
            Matrix::from_rows([
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 10.0, 12.0],
                [13.0, 14.0, 15.0, 15.0],
            ])
        }
        6 => {
            // test for Conjugate Gradient method
            Matrix::from_rows([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        }
        _ => {
            let err_msg = String::from("Not available option.");
            let err = JmError::NotAvailable(err_msg);
            Err(Box::new(err))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn get_matrix_test() -> Result<(), Box<dyn Error>> {
        const CASE: usize = 4;
        let A = get_matrix(CASE)?;
        let (m, _) = A.dim();

        if m < 5 {
            println!("{A:.4}");
        }

        Ok(())
    }
}
