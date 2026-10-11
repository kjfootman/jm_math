mod identity;
mod jacobi;

use crate::error::Error;
pub use identity::Idendity;
pub use jacobi::Jacobi;

pub trait Preconditioner {
    fn label(&self) -> &'static str;
    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error>;
}
