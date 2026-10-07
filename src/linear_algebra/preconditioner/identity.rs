use crate::{error::Error, linear_algebra::preconditioner::Preconditioner};

#[derive(Default)]
pub struct Idendity;

impl Idendity {
    pub fn new() -> Self {
        Idendity
    }
}

impl Preconditioner for Idendity {
    fn label(&self) -> &'static str {
        "without preconditioner"
    }

    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        inplace.copy_from_slice(v);
        Ok(())
    }
}
