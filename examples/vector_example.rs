use jm_math::prelude::Vector;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let v0 = Vector::from_iter([-1.0, 0.0, 1.0, 3.0]);
    let v1 = Vector::from_iter([3.0, 1.0, 0.0, -1.0]);

    // addition
    assert_eq!(&v0 + &v1, Vector::from_iter([2.0, 1.0, 1.0, 2.0]));
    // subtraction
    assert_eq!(&v0 - &v1, Vector::from_iter([-4.0, -1.0, 1.0, 4.0]));
    // dot product
    assert_eq!(&v0 * &v1, -6.0);
    // multiplication
    assert_eq!(2.0 * &v0, Vector::from_iter([-2.0, 0.0, 2.0, 6.0]));
    // division
    assert_eq!(&v0 / 2.0, Vector::from_iter([-0.5, 0.0, 0.5, 1.5]));
    // negation
    assert_eq!(-&v0, Vector::from_iter([1.0, 0.0, -1.0, -3.0]));
    // magnitude
    assert_eq!(v0.magnitude(), f64::sqrt(11.0));

    Ok(())
}
