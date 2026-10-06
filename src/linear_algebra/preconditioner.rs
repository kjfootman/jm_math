mod identity;
mod jacobi;

use crate::error::Error;
pub use identity::Idendity;
pub use jacobi::Jacobi;

#[allow(clippy::upper_case_acronyms)]
enum PreconType {
    Identity(&'static str),
    Jacobi(&'static str),
    SOR(&'static str, f32),
    SSOR(&'static str, f32),
    ILU(&'static str),
}

pub trait Preconditioner {
    fn label(&self) -> &'static str;
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error>;
}
