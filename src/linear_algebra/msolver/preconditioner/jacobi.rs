use crate::linear_algebra::Matrix;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::error::Error;

pub fn Jacobi(A: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    //! Returns the Jacobi preconditioner.
    let (m, _) = A.dim();
    let uptr = A.get_ptr_diagonal()?;
    let ja_ref = A.JA();
    let aa_ref = A.AA();
    let IA = (0..m + 1).collect::<Vec<_>>();
    let JA = uptr.par_iter().map(|ia| ja_ref[*ia]).collect::<Vec<_>>();
    let AA = uptr
        .par_iter()
        .map(|ia| 1f64 / aa_ref[*ia])
        .collect::<Vec<_>>();

    Matrix::from_csr(IA, JA, AA)
}
