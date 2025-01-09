use crate::linear_algebra::Matrix;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::error::Error;

pub fn SOR(A: &Matrix, w: f64) -> Result<Matrix, Box<dyn Error>> {
    //! Returns the Gauss-Seidel preconditioner. Lower part of the matrix.
    let (m, _) = A.dim();
    let uptr = A.get_ptr_diagonal()?;
    let IA = A.IA();
    let JA = A.JA();
    let AA = A.AA();

    // Each component of pure lower part is devided by its diagonal component.(pivot)
    // diagonal components is muliplied by w.
    let AA = (0..m)
        .into_par_iter()
        .map(|i| {
            let n = (IA[i]..uptr[i] + 1).len();
            let mut t1;
            let mut jrow;
            let mut local_AA = Vec::with_capacity(n);

            for j in IA[i]..uptr[i] {
                jrow = JA[j];
                t1 = w * AA[j] / AA[uptr[jrow]];
                local_AA.push(t1);
            }

            local_AA.push(w * AA[uptr[i]].recip());
            local_AA
        })
        .flatten()
        .collect::<Vec<_>>();

    let JA = (0..m)
        .into_par_iter()
        .map(|i| JA[IA[i]..uptr[i] + 1].to_vec())
        .flatten()
        .collect::<Vec<_>>();

    //? possible to paralellize?
    let IA = std::iter::once(0)
        .chain((0..m).scan(0, |acc, i| {
            *acc += (IA[i]..uptr[i] + 1).count();
            Some(*acc)
        }))
        .collect::<Vec<_>>();

    Matrix::from_csr(IA, JA, AA)
}

// #[cfg(test)]
// mod tests {
//     use crate::prelude::Vector;

//     use super::*;

//     #[test]
//     fn SOR_test() -> Result<(), Box<dyn Error>> {
//         let A = Matrix::from_rows([
//             [4.0 / 1.2, 0.0, 0.0, 0.0],
//             [2.0, 5.0 / 1.2, 0.0, 0.0],
//             [0.0, 0.0, 1.0 / 1.2, 0.0],
//             [2.0, 0.0, 0.0, 6.0 / 1.2],
//         ])?;

//         let x = SOR(&A, 1.2)?;
//         println!("{x:.4}");

//         Ok(())
//     }
// }
