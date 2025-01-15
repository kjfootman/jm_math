#![allow(non_snake_case)]
use jm_math::prelude::{Matrix, Vector};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Matrix initialization from rows
    let M0 = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]])?;

    // initialization from simple coordinates
    let M1 = Matrix::from_simple_coordinate([(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)])?;
    assert_eq!(M0, M1);

    // Matrix vector multiplication
    let b = Vector::from_iter([1.0, 1.0]);
    assert_eq!(&M0 * &b, Vector::from_iter([3.0, 7.0]));

    Ok(())
}
