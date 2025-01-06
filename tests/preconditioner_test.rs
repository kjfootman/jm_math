#![allow(non_snake_case)]

mod matrix;
mod utils;

use jm_math::linear_algebra::{MSolver, Matrix, PreconType, Vector};
use std::error::Error;

fn build_source(A: &Matrix) -> Vector {
    let (_, n) = A.dim();
    let b = Vector::from_iter(vec![1.0; n]);

    A * b
}

#[test]
fn preconditioner_test() -> Result<(), Box<dyn Error>> {
    // initialize log4rs
    log4rs::init_file("log4rs.yaml", Default::default())?;

    const CASE: usize = 3;
    const DIGIT: i32 = 4;
    const TOL: f64 = 1E-12;

    let m = 10;
    let A = matrix::get_matrix(CASE)?;
    // let P = PreconType::Jacobi;
    let P = PreconType::SOR(1.2);
    // let P = PreconType::ILU0;
    let b = build_source(&A);
    let ms = MSolver::build(&A, &b)
        .preconditioner(P)
        .tolerance(TOL)
        .finish();

    // Test for GMRES(m)
    let x = ms
        .GMRES(m)?
        .iter()
        .map(|x| utils::round(*x, DIGIT))
        .collect::<Vec<_>>();

    assert_eq!(*x, vec![1.0; b.len()]);

    // Test for HGMRES(m)
    let x = ms
        .HGMRES(m)?
        .iter()
        .map(|x| utils::round(*x, DIGIT))
        .collect::<Vec<_>>();
    assert_eq!(*x, vec![1.0; b.len()]);

    let x = ms
        .CG()?
        .iter()
        .map(|x| utils::round(*x, DIGIT))
        .collect::<Vec<_>>();
    assert_eq!(*x, vec![1.0; b.len()]);

    Ok(())
}
