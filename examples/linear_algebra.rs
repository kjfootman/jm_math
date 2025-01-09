#![allow(non_snake_case)]
use std::error::Error;
use jm_math::linear_algebra::{MSolver, Matrix, Vector, PreconType};

fn main() -> Result<(), Box<dyn Error>>{
    log4rs::init_file("log4rs.yaml", Default::default())?;

    // coefficient matrix
    let A = Matrix::from_rows([
        [5.0, 3.0, 0.0, 1.0],
        [2.0, 6.0, 3.0, 0.0],
        [0.0, 0.0, 7.0, 1.0],
        [2.0, 0.0, 0.0, 8.0],
    ])?;

    // source vector
    let b = Vector::from_iter([9.0, 11.0, 8.0, 10.0]);

    // set sovler
    let ms = MSolver::build(&A, &b)
        .max_iter(100)
        .tolerance(1e-10)
        .preconditioner(PreconType::SOR(1.2))
        .finish();

    // solve the system of equstions with GMRES(m)
    let x = ms.GMRES(2)?;
    println!("{x:.4}");

    Ok(())
}
