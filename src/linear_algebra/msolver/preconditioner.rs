mod ilu;
mod jacobi;
mod sor;

use crate::linear_algebra::{Matrix, Vector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::error::Error;

#[derive(Debug, Clone, Copy)]
pub enum PreconType {
    Jacobi,
    ILU0,
    SOR(f64),
}

pub fn get_preconditioner(A: &Matrix, pType: PreconType) -> Result<Preconditioner, Box<dyn Error>> {
    //! Constructs a preconditioner associated with each preconditioner type.
    let M = match pType {
        PreconType::Jacobi => jacobi::Jacobi(A),
        PreconType::ILU0 => ilu::ILU(A),
        PreconType::SOR(w) => sor::SOR(A, w),
    }?;

    get_level_scheduled_preconditioner(M)
}

#[derive(Debug)]
/// Struct for Preconditioner.
pub struct Preconditioner {
    M: Matrix,
    l_q: Vec<usize>,
    l_lev: Vec<usize>,
    u_q: Vec<usize>,
    u_lev: Vec<usize>,
    uptr: Vec<usize>,
}

fn get_level_scheduled_preconditioner(M: Matrix) -> Result<Preconditioner, Box<dyn Error>> {
    //! Returns the Preconditioner with the information for level scheduling.
    let uptr = M.get_ptr_diagonal()?;
    let (l_depth, u_depth) = get_depth(&M, &uptr)?;
    let (l, u) = rayon::join(|| get_level(l_depth), || get_level(u_depth));
    let (l_q, l_lev) = l?;
    let (u_q, u_lev) = u?;

    let preconditioner = Preconditioner {
        M,
        l_q,
        l_lev,
        u_q,
        u_lev,
        uptr,
    };

    Ok(preconditioner)
}

impl std::fmt::Display for Preconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);
        let precision = f.precision().unwrap_or(2);

        writeln!(f, "{:>width$.precision$}", self.M)
    }
}

pub fn preconditioning(P: &Preconditioner, v: &Vector) -> Result<Vector, Box<dyn Error>> {
    //! Preconditioning `Vector` v with level scheduling.
    //! P * x = v
    let n = v.len();
    let uptr = &P.uptr;
    let M = &P.M;
    let IA = M.IA();
    let JA = M.JA();
    let AA = M.AA();

    // foward sweep
    let mut q = &P.l_q;
    let mut level = &P.l_lev;
    let mut nlev = level.len() - 1;
    let mut y = vec![0.0; n];

    for j in 0..nlev {
        let y_local = &y;
        let subset = (level[j]..level[j + 1])
            .into_par_iter()
            .map(|k| {
                let i = q[k];
                let mut sum = v[i];
                for ia in IA[i]..uptr[i] {
                    sum -= AA[ia] * y_local[JA[ia]];
                }

                sum
            })
            .collect::<Vec<_>>();

        for (i, k) in (level[j]..level[j + 1]).enumerate() {
            y[q[k]] = subset[i];
        }
    }

    // backward sweep
    q = &P.u_q;
    level = &P.u_lev;
    nlev = level.len() - 1;

    // let mut x = vec![0.0; n];
    let mut x = y;
    for j in 0..nlev {
        let x_local = &x;
        let subset = (level[j]..level[j + 1])
            .into_par_iter()
            .map(|k| {
                let i = q[k];
                let mut sum = x[i];
                // let mut sum = y[i];
                for ia in uptr[i] + 1..IA[i + 1] {
                    sum -= AA[ia] * x_local[JA[ia]];
                }

                sum * AA[uptr[i]]
            })
            .collect::<Vec<_>>();

        for (i, k) in (level[j]..level[j + 1]).enumerate() {
            x[q[k]] = subset[i]
        }
    }

    Ok(Vector::from_iter(x))
}

fn get_depth(M: &Matrix, uptr: &[usize]) -> Result<(Vec<usize>, Vec<usize>), Box<dyn Error>> {
    //! Returns the tuple of lower, upper depth of a `Matrix` M.
    let (m, _) = M.dim();
    // let uptr = M.get_ptr_diagonal()?;
    let IA = M.IA();
    let JA = M.JA();

    let (l_depth, u_depth) = rayon::join(
        || {
            // compute lower depth
            let mut l_depth = vec![0; m];
            for i in 0..m {
                l_depth[i] = JA[IA[i]..=uptr[i]].iter().map(|ja| l_depth[*ja]).max()? + 1;
            }

            Some(l_depth)
        },
        || {
            // compute uppper depth
            let mut u_depth = vec![0; m];
            for i in (0..m).rev() {
                u_depth[i] = JA[uptr[i]..IA[i + 1]].iter().map(|ja| u_depth[*ja]).max()? + 1;
            }

            Some(u_depth)
        },
    );

    Ok((
        l_depth.ok_or("Failed to compute lower depth.")?,
        u_depth.ok_or("Failed to compute upper depth.")?,
    ))
}

fn get_level(depth: Vec<usize>) -> Result<(Vec<usize>, Vec<usize>), String> {
    //! Returns the tuple for level scheduling of permutation array and level.

    // compute a perputation array
    let mut q = (0..depth.len()).collect::<Vec<_>>();
    q.par_sort_by(|i, j| depth[*i].cmp(&depth[*j]));

    // compute the array of levels
    let chunk_size = q
        .par_chunk_by(|i, j| depth[*i] == depth[*j])
        .map(|chunk| chunk.len())
        .collect::<Vec<_>>();

    let level = std::iter::once(0)
        .chain(chunk_size.iter().scan(0, |acc, &size| {
            *acc += size;
            Some(*acc)
        }))
        .collect();

    Ok((q, level))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JmError;
    use std::time::Instant;

    #[test]
    fn lev_schedule_test() -> Result<(), Box<dyn Error>> {
        let M = get_test_matrix(0)?;
        let start = Instant::now();
        let (lev, q) = M.level_scheduling()?;

        println!("elapsed: {:.4} sec", start.elapsed().as_secs_f32());
        println!("Length of levels: {}", lev.len());
        println!("Lenght of q: {}", q.len());

        assert_eq!(lev, vec![0, 1, 3, 6, 9, 11, 12]);
        assert_eq!(q, vec![0, 1, 4, 2, 5, 8, 3, 6, 9, 7, 10, 11]);

        Ok(())
    }

    #[test]
    fn get_depth_test() -> Result<(), Box<dyn Error>> {
        let M = get_test_matrix(4)?;
        let uptr = M.get_ptr_diagonal()?;
        let (ldepth, udepth) = get_depth(&M, &uptr)?;

        assert_eq!(ldepth, vec![1, 2, 1, 1]);
        assert_eq!(udepth, vec![3, 2, 2, 1]);

        Ok(())
    }

    #[test]
    fn get_level_test() -> Result<(), Box<dyn Error>> {
        let M = get_test_matrix(0)?;
        let uptr = M.get_ptr_diagonal()?;
        let (l_depth, u_depth) = get_depth(&M, &uptr)?;
        let (l_q, l_level) = get_level(l_depth)?;
        let (u_q, u_level) = get_level(u_depth)?;

        assert_eq!(l_q, vec![0, 1, 4, 2, 5, 8, 3, 6, 9, 7, 10, 11]);
        assert_eq!(l_level, vec![0, 1, 3, 6, 9, 11, 12]);

        assert_eq!(u_q, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(u_level, vec![0, 12]);

        Ok(())
    }

    #[test]
    fn preconditioning_test() -> Result<(), Box<dyn Error>> {
        let A = get_test_matrix(5)?;
        let y = Vector::from_iter(vec![1.0, 2.0, 5.0]);

        // Test for Jacobi preconditioner.
        let P = get_preconditioner(&A, PreconType::Jacobi)?;
        let v = preconditioning(&P, &y)?;
        assert_eq!(v, Vector::from_iter(vec![1.0, 1.0, 1.0]));

        // Test for SOR(w) preconditioner.
        let P = get_preconditioner(&A, PreconType::SOR(1.0))?;
        let v = preconditioning(&P, &y)?;
        assert_eq!(v, Vector::from_iter(vec![1.0, 1.0, -0.4]));

        // Test for ILU(0) preconditioner.
        let P = get_preconditioner(&A, PreconType::ILU0)?;
        let v = preconditioning(&P, &y)?;
        assert_eq!(v, Vector::from_iter(vec![0.0, 0.0, 1.0]));

        Ok(())
    }

    fn get_test_matrix(case: u8) -> Result<Matrix, Box<dyn Error>> {
        match case {
            0 => {
                // test matrix in text book
                let IA = [0, 1, 3, 5, 7, 9, 12, 15, 18, 20, 23, 26, 29];
                let JA = [
                    0, 0, 1, 1, 2, 2, 3, 0, 4, 1, 4, 5, 2, 5, 6, 3, 6, 7, 4, 8, 5, 8, 9, 6, 9, 10,
                    7, 10, 11,
                ];
                let AA = [
                    3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0,
                    1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0,
                ];

                Matrix::from_csr(IA, JA, AA)
            }
            1 => {
                // matrix market
                Matrix::from_matrix_market(
                    "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/fs_760_1.mtx",
                )
            }
            2 => {
                // matrix market; symmetric
                Matrix::from_matrix_market(
                    "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/bcsstk13.mtx",
                )
            }
            3 => {
                // matrix market; unsymmetric
                Matrix::from_matrix_market(
                    "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/e40r5000.mtx",
                )
            }
            4 => {
                // 4 x 4 matrix
                Matrix::from_rows([
                    [1.0, 0.0, 2.0, 3.0],
                    [4.0, 5.0, 0.0, 6.0],
                    [0.0, 0.0, 7.0, 8.0],
                    [0.0, 0.0, 0.0, 9.0],
                ])
            }
            5 => {
                // 3 x 3 matrix
                Matrix::from_rows([[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [3.0, 4.0, 5.0]])
            }
            6 => {
                // 3 x 3 matrix
                Matrix::from_rows([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
            }
            _ => {
                let err_msg = String::from("Not available option.");
                let err = JmError::NotAvailable(err_msg);
                Err(Box::new(err))
            }
        }
    }
}
