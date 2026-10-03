mod jacobi;

use crate::error::Error;
pub use jacobi::Jacobi;

pub struct NoPreconditioner;

pub trait Preconditioner {
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error>;
}

impl Preconditioner for NoPreconditioner {
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        inplace.copy_from_slice(v);
        Ok(())
    }
}
