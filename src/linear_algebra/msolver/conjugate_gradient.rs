use crate::linear_algebra::msolver::{preconditioner as pc, MSolver};
use crate::linear_algebra::Vector;
use crate::JmError;
use std::error::Error;

pub fn solve(ms: &MSolver) -> Result<Vector, Box<dyn Error>> {
    let A = ms.A;
    let b = ms.b;
    let b_norm = b.magnitude();
    let iMax = ms.iMax;
    let tol = ms.tol;
    let mut iter = 0;
    let mut res = f64::MAX;
    let mut x = Vector::from_iter(vec![0f64; b.len()]);
    let mut r = b - A * &x;
    let mut alpha;
    // let mut rsold = &r * &r;
    // let mut Ap;
    let mut Aq;

    // set preconditioner
    let P = match ms.preconditioner {
        Some(pType) => pc::get_preconditioner(A, pType).ok(),
        None => None,
    };

    let mut q = match &P {
        Some(P) => pc::preconditioning(P, &r)?,
        None => r.clone(),
    };

    let mut z = match &P {
        Some(P) => pc::preconditioning(P, &r)?,
        None => r.clone(),
    };

    let mut rsold = &z * &r;

    while res > tol && iter < iMax {
        // Ap = A * &p;
        // alpha = (&r * &r) / (&Ap * &p);
        Aq = A * &q;
        alpha = (&z * &r) / (&Aq * &q);

        // x += alpha * &p;
        // r -= alpha * &Ap;
        x += alpha * &q;
        r -= alpha * &Aq;
        z = match &P {
            Some(P) => pc::preconditioning(P, &r)?,
            None => z,
        };

        // let rsnew = &r * &r;
        let rsnew = &z * &r;

        // p = &r + (rsnew / rsold) * &p;
        // rsold = rsnew;
        q = &z + (rsnew / rsold) * &q;
        rsold = rsnew;

        res = rsnew.sqrt() / b_norm;
        iter += 1;
    }

    if iter == iMax {
        let err_msg = String::from("CG: Maximum iteration exceeded.");
        let err = JmError::ConvErr(err_msg);
        return Err(Box::new(err));
    }

    log::info!("CG iteration: {iter}, residual: {res:.4E}");

    Ok(x)
}
