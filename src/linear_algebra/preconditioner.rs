mod jacobi;

use crate::error::Error;
pub use jacobi::Jacobi;

pub struct NoPreconditioner;

pub trait Preconditioner {
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error>;
}

impl Preconditioner for NoPreconditioner {
    fn preconditioning(&self, _: &[f64], _: &mut [f64]) -> Result<(), Error> {
        Ok(())
    }
}
