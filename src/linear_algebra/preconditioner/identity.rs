use crate::{error::Error, linear_algebra::preconditioner::Preconditioner};

pub struct Idendity;

impl Preconditioner for Idendity {
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        inplace.copy_from_slice(v);
        Ok(())
    }
}
