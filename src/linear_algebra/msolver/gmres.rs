use crate::linear_algebra;
use crate::linear_algebra::msolver::{preconditioner as pc, MSolver};
use crate::linear_algebra::Vector;
use crate::JmError;
use std::error::Error;

pub fn solve(restart: usize, ms: &MSolver) -> Result<Vector, Box<dyn Error>> {
    //! A linear system of equations is solved using GMRES.
    //! The method returns the solution if the relative residual satisfies the convergence criterion.
    //!
    //! # Examples
    //!
    //! ```
    //! # #![allow(non_snake_case)]
    //! # use std::error::Error;
    //! # use jm_math::prelude::{Matrix, Vector};
    //! # fn main() -> Result<(), Box<dyn Error>> {
    //! # Ok(())
    //! # }
    //! ```
    let A = ms.A;
    let b = ms.b;
    let b_norm = b.magnitude();
    let iMax = ms.iMax;
    let tol = ms.tol;
    let mut iter = 0;
    let mut res = f64::MAX;
    let mut restart = restart;
    let mut x = Vector::from_iter(vec![0f64; b.len()]);

    // set preconditioner
    let P = match ms.preconditioner {
        Some(pType) => pc::get_preconditioner(A, pType).ok(),
        None => None,
    };

    // let P = ms.get_preconditioner();

    // *iter = 0;
    // *res = f64::MAX;

    while res > tol && iter < iMax {
        let r = b - A * &x;
        let beta = r.magnitude();
        let mut V = Vec::with_capacity(restart);
        let mut H = Vec::with_capacity(restart);
        let mut g = vec![0f64; restart + 1];

        g[0] = beta;
        V.push(r / beta);

        // Arnoldi process
        for j in 0..restart {
            // preconditioning
            let mut w = match &P {
                Some(P) => A * pc::preconditioning(P, &V[j])?,
                None => A * &V[j],
            };
            let mut h = vec![0f64; j + 2];

            for (i, v) in V.iter().enumerate().take(j + 1) {
                h[i] = &w * v;
                w -= h[i] * v;
            }

            h[j + 1] = w.magnitude();
            H.push(h);

            // lucky breakdown
            if H[j][j + 1].abs() < tol {
                println!("GMRES({restart}) lucky breakdown");
                restart = j + 1;
                break;
            }

            V.push(w / H[j][j + 1]);
        }

        // Givens rotation
        linear_algebra::Givens_rotation(&mut H, &mut g);

        let y = linear_algebra::solve_upper_triangular(&H, &g);
        let mut z = Vector::from_iter(vec![0.0; b.len()]);

        for (i, v) in V.iter().enumerate().take(restart) {
            z += y[i] * v;
        }

        // preconditioning
        z = match &P {
            Some(P) => pc::preconditioning(P, &z)?,
            None => z,
        };

        // update solution
        x += z;

        res = g[restart].abs() / b_norm;
        iter += 1;

        log::info!("GMRES({restart}) iteration: {iter}, residual: {res:.4E}");
    }

    if iter == iMax {
        let err_msg = format!("GMRES({restart}): Maximum iteration exceeded.");
        let err = JmError::ConvErr(err_msg);
        return Err(Box::new(err));
    }

    log::info!("GMRES({restart}) iteration: {iter}, residual: {res:.4E}");

    Ok(x)
}
