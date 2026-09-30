mod jacobi;

use crate::error::Error;
pub use jacobi::Jacobi;

pub struct None;

pub trait Preconditioner {
    fn preconditioning(&self, v: &mut [f64]) -> Result<(), Error>;
}

impl Preconditioner for None {
    fn preconditioning(&self, _: &mut [f64]) -> Result<(), Error> {
        Ok(())
    }
}
