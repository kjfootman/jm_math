use crate::{
    error::Error,
    linear_algebra::preconditioner::{PreconType, Preconditioner},
};

pub struct Idendity {
    ty: PreconType,
}

impl Idendity {
    pub fn new() -> Self {
        Self {
            ty: PreconType::Identity("without preconditioner"),
        }
    }
}

impl Default for Idendity {
    fn default() -> Self {
        Self {
            ty: PreconType::Identity("without preconditioner"),
        }
    }
}

impl Preconditioner for Idendity {
    fn label(&self) -> &'static str {
        match self.ty {
            PreconType::Identity(value) => value,
            _ => "without preconditioner",
        }
    }

    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        inplace.copy_from_slice(v);
        Ok(())
    }
}
