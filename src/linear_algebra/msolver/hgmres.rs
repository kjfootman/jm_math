use crate::linear_algebra;
use crate::linear_algebra::msolver::{preconditioner as pc, MSolver};
use crate::linear_algebra::Vector;
use crate::JmError;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::error::Error;

pub fn solve(restart: usize, ms: &MSolver) -> Result<Vector, Box<dyn Error>> {
    let A = ms.A;
    let b = ms.b;
    let b_norm = b.magnitude();
    let (_, n) = A.dim();
    let iMax = ms.iMax;
    let mut iter = 0;
    let mut res = f64::MAX;
    let tol = ms.tol;
    let mut restart = restart;
    let mut x = Vector::from_iter(vec![0f64; b.len()]);

    // set preconditioner
    let P = match ms.preconditioner {
        Some(pType) => pc::get_preconditioner(A, pType).ok(),
        None => None,
    };

    // *iter = 0;
    // *res = f64::MAX;

    while res > tol && iter < iMax {
        let r = b - A * &x;
        let beta = -r[0].signum() * r.magnitude();
        let mut U = Vec::with_capacity(restart);
        let mut H = Vec::with_capacity(restart);
        let mut g = vec![0f64; restart + 1];
        let mut z = r;

        g[0] = beta;

        // Householder normalization
        for j in 0..restart + 1 {
            // Compute the Househoder unit vector
            U.push(linear_algebra::cal_Householder_vec(&z[j..n]));

            // compute H[j] = Pj * z;
            let mut h = vec![0f64; j + 1];
            z.iter().enumerate().take(j + 1).for_each(|(i, value)| {
                //* parallelization not needed because j + 1 << n *//
                h[i] = *value;
            });

            h[j] += (j..n)
                .into_par_iter()
                .map(|i| -2.0 * U[j][0] * U[j][i - j] * z[i])
                .sum::<f64>();

            if j != 0 {
                H.push(h);

                // lucky breakdown
                if H[j - 1][j].abs() < tol {
                    println!("HGMRES({restart}) lucky breakdown");
                    restart = j;
                    break;
                }
            }

            // compute basis vector vj = P0 * P1 * P2 * Pj * ej
            z = Vector::from_iter(vec![0f64; n]);
            z[j] = 1f64;

            for (i, u) in U.iter().enumerate().take(j + 1).rev() {
                let sigma = (i..n)
                    .into_par_iter()
                    .map(|k| -2f64 * u[k - i] * z[k])
                    .sum::<f64>();

                z[i..n].par_iter_mut().enumerate().for_each(|(k, value)| {
                    *value += sigma * u[k];
                });
            }

            // compute z = Pj * P2 * P1 * P0 * A * vj
            // preconditioning
            // z = A * z;
            z = match &P {
                Some(P) => A * pc::preconditioning(P, &z)?,
                None => A * z,
            };

            for (i, u) in U.iter().enumerate().take(j + 1) {
                let sigma = (i..n)
                    .into_par_iter()
                    .map(|k| -2f64 * u[k - i] * z[k])
                    .sum::<f64>();

                z[i..n]
                    // .iter_mut()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(k, value)| *value += sigma * U[i][k]);
            }
        }

        // Givens roatation
        linear_algebra::Givens_rotation(&mut H, &mut g);

        let y = linear_algebra::solve_upper_triangular(&H, &g);

        // x = x0 + y0 * v0 + y1 * v1 + ... + ym-1 * vm-1
        z = Vector::from_iter(vec![0f64; n]);

        for (i, u) in U.iter().enumerate().take(restart).rev() {
            z[i] += y[i];
            let sigma = (i..n)
                .into_par_iter()
                .map(|k| -2f64 * u[k - i] * z[k])
                .sum::<f64>();

            z[i..n]
                .par_iter_mut()
                // .iter_mut()
                .enumerate()
                .for_each(|(k, value)| *value += sigma * U[i][k]);
        }

        // preconditioning
        z = match &P {
            Some(P) => pc::preconditioning(P, &z)?,
            None => z,
        };

        // update solution
        x += &z;

        res = g[restart].abs() / b_norm;
        iter += 1;

        log::info!("HGMRES({restart}) iteration: {iter}, residual: {res:.4E}");
    }

    if iter == iMax {
        let err_msg = format!("HGMRES({restart}): Maximum iteration exceeded.");
        let err = JmError::ConvErr(err_msg);
        return Err(Box::new(err));
    }

    log::info!("HGMRES({restart}) iteration: {iter}, residual: {res:.4E}");

    Ok(x)
}
