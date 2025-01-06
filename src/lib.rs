#![allow(non_snake_case)]
//! Fast and easy library for mathematics.
//!
//! Provides some modules for linear algebra, geometry, ...
//!
//! # First section
//!
//!
//! # Second section

mod error;
pub mod linear_algebra;
pub mod prelude;

#[doc(hidden)]
pub use error::JmError;

// #[doc(inline)]
// pub use linear_algebra::{MSolver, Matrix, PreconType, Vector};
