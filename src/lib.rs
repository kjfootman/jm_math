#![allow(non_snake_case)]
mod error;
mod linear_algebra;

pub use error::Error;
pub use linear_algebra::msolver;
pub use linear_algebra::{CSRMatrix, Vector};
