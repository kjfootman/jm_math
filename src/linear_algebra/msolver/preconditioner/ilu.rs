use crate::linear_algebra::Matrix;
use crate::JmError;
use std::error::Error;

pub fn ILU(A: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    //! Returns the ILU(0) preconditioner.
    let (m, n) = A.dim();
    let uptr = A.get_ptr_diagonal()?;
    let mut IA = Vec::new();
    let mut JA = Vec::new();
    let mut luval = Vec::new();

    // copy csr
    rayon::scope(|s| {
        s.spawn(|_| {
            IA = A.IA().clone();
        });

        s.spawn(|_| {
            JA = A.JA().clone();
        });

        s.spawn(|_| {
            luval = A.AA().clone();
        });
    });

    for k in 0..m {
        let mut iw = vec![0; n];
        let mut t1;
        let mut jrow;

        for j in IA[k]..IA[k + 1] {
            iw[JA[j]] = j;
        }

        for j in IA[k]..uptr[k] + 1 {
            jrow = JA[j];

            // diagonal components
            if jrow == k {
                let value = luval[uptr[k]].recip();

                if value.is_infinite() {
                    let err_msg = format!("The diagonal component of {}th row is zero.", jrow);
                    let err = JmError::ValueErr(err_msg);
                    return Err(Box::new(err));
                }

                luval[uptr[k]] = value;
                continue;
            }

            t1 = luval[j] * luval[uptr[jrow]];
            luval[j] = t1;

            for jj in uptr[jrow] + 1..IA[jrow + 1] {
                let jw = iw[JA[jj]];
                if jw != 0 {
                    luval[jw] -= t1 * luval[jj]
                }
            }
        }
    }

    Matrix::from_csr(IA, JA, luval)
}
