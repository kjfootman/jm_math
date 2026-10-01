pub mod jacobi;

use crate::error::Error;
pub use jacobi::Jacobi;

pub struct NoPreconditioner;

pub trait Preconditioner {
    fn preconditioning(&self, v: &mut [f64]) -> Result<(), Error>;
}

impl Preconditioner for NoPreconditioner {
    fn preconditioning(&self, _: &mut [f64]) -> Result<(), Error> {
        Ok(())
    }
}
