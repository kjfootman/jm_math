mod matrix;
mod msolver;
pub mod preconditioner;
mod simd;
mod vector;

pub use matrix::{CSRMatrix, CSRMatrixArgs, Matrix, csr};
pub use msolver::{ConjugateGradientBuilder, GaussSeidelBuilder, MSolver};
pub use vector::Vector;
