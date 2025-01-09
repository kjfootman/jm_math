#![allow(non_snake_case)]
mod matrix;

use jm_math::linear_algebra::{MSolver, Vector};
use std::error::Error;

#[test]
fn hgmres_verification_test() -> Result<(), Box<dyn Error>> {
    log4rs::init_file("log4rs.yaml", Default::default())?;

    let A = matrix::get_matrix(0)?;
    let b = Vector::from_iter([6.0, 15.0, 15.0, 9.0]);
    let ms = MSolver::new(&A, &b);
    let x = ms.HGMRES(4)?;

    println!("{x:.4}");

    Ok(())
}
