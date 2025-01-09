//! A module for solving systems of equations.
//!
//! This module provides numerical methods for solving systems of equations.
//! It also offers preconditioners to speed up the convergence rate of the numerical methods.

mod conjugate_gradient;
mod gauss_seidel;
mod gmres;
mod hgmres;
mod preconditioner;

use crate::linear_algebra::{Matrix, Vector};
pub use preconditioner::PreconType;
use std::error::Error;

/// A struct for solving systems of equations.
///
/// The `MSolver` is a struct designed to solve systems of equations.
/// It allows you to apply various numerical methods.
/// The following methods are available for you to use.
///
/// **iterative solvers**
/// - [x] Conjugate Gradient method for symmetric matrix - CG
/// - [x] Gernalized Minimum RESidual method with restart m - GMRES(m)
/// - [x] Householder version of GMRES(m) - HMGRES(m)
///
/// **preconditioners**
/// - Jacobi
/// - Successive Over Relaxation - SOR(w)
/// - Incomplete Lower Upper factorization - ILU0
///
/// The methods mentioned above return a `Vector` of solutions
/// when the relative residual L2-norm converges under the specified tolerance.
///
/// # Usage
///
/// The `MSolver` can be initialized with `build()` method.
/// And then the maxium iteration number, tolerance and preconditioner can be specified with it.
/// You can choose a solver depending on your problem.
///
/// **Examples**
///
/// ```rust
/// # #![allow(non_snake_case)]
/// # use std::error::Error;
/// use jm_math::prelude::{MSolver, Matrix, PreconType, Vector};
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// // coeffcient matrix
/// let A = Matrix::from_rows([
///     [4.0, 3.0, 0.0, 1.0],
///     [2.0, 5.0, 3.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0],
///     [2.0, 0.0, 0.0, 6.0],
/// ])?;
///
/// // source vector
/// let b = Vector::from_iter([8.0, 10.0, 2.0, 8.0]);
///
/// // initialize MSolver
/// let ms = MSolver::build(&A, &b)
///     .max_iter(100)
///     .preconditioner(PreconType::ILU0)
///     .tolerance(1e-8)
///     .finish();
///
/// // It's also possible to initialize to its default state.
/// // let ms = MSolver::new(&A, &b);
///
/// // solve with GMRES(m)
/// let x = ms.GMRES(2)?;
/// println!("{x:.4}");
///
/// // error
/// let error = &x - Vector::from_iter([1.0, 1.0, 1.0, 1.0]);
/// assert!(error.magnitude() < 1e-5);
///
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct MSolver<'a> {
    iMax: usize,
    tol: f64,
    A: &'a Matrix,
    b: &'a Vector,
    preconditioner: Option<PreconType>,
}

#[derive(Clone, Copy)]
pub struct MSolverBuilder<'a> {
    ms: MSolver<'a>,
}

impl<'a> MSolver<'a> {
    pub fn new(A: &'a Matrix, b: &'a Vector) -> Self {
        //! Returns a default `MSolver`.
        MSolver {
            iMax: 1000,
            tol: 1E-7,
            A,
            b,
            preconditioner: None,
        }
    }

    pub fn build(A: &'a Matrix, b: &'a Vector) -> MSolverBuilder<'a> {
        //! Returns an `MSolverBuilder` struct, which is used to build an `MSolver`.
        let ms = MSolver {
            iMax: 1000,
            tol: 1E-7,
            A,
            b,
            preconditioner: None,
        };

        MSolverBuilder { ms }
    }

    pub fn GMRES(&self, restart: usize) -> Result<Vector, Box<dyn Error>> {
        //! Solves systems of euqations using GMRES(m).
        //!
        //! # Examples
        //!
        //! ```rust
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! use jm_math::prelude::{MSolver, Matrix, PreconType, Vector};
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! # let A = Matrix::from_rows([
        //! #     [5.0, 3.0, 0.0, 1.0],
        //! #     [2.0, 6.0, 3.0, 0.0],
        //! #     [0.0, 0.0, 1.0, 1.0],
        //! #     [2.0, 0.0, 0.0, 6.0],
        //! # ])?;
        //! # let b = Vector::from_iter([9.0, 11.0, 2.0, 8.0]);
        //!
        //! # // initialize MSolver
        //! let ms = MSolver::build(&A, &b)
        //!     .max_iter(100)
        //!     .preconditioner(PreconType::SOR(1.2))
        //!     .tolerance(1e-8)
        //!     .finish();
        //!
        //! // solve with GMRES(m)
        //! let x = ms.GMRES(2)?;
        //! println!("{x:.4}");
        //!
        //! # Ok(())
        //! # }
        //! ```
        gmres::solve(restart, self)
    }

    pub fn HGMRES(&self, restart: usize) -> Result<Vector, Box<dyn Error>> {
        //! Solves systems of euqations using HGMRES(m).
        //!
        //! # Examples
        //!
        //! ```rust
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! use jm_math::prelude::{MSolver, Matrix, PreconType, Vector};
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! # let A = Matrix::from_rows([
        //! #     [5.0, 3.0, 0.0, 1.0],
        //! #     [2.0, 6.0, 3.0, 0.0],
        //! #     [0.0, 0.0, 1.0, 1.0],
        //! #     [2.0, 0.0, 0.0, 6.0],
        //! # ])?;
        //! # let b = Vector::from_iter([9.0, 11.0, 2.0, 8.0]);
        //!
        //! # // initialize MSolver
        //! let ms = MSolver::build(&A, &b)
        //!     .max_iter(100)
        //!     .preconditioner(PreconType::SOR(1.2))
        //!     .tolerance(1e-8)
        //!     .finish();
        //!
        //! // solve with HGMRES(m)
        //! let x = ms.HGMRES(2)?;
        //! println!("{x:.4}");
        //!
        //! # Ok(())
        //! # }
        //! ```
        hgmres::solve(restart, self)
    }

    pub fn CG(&self) -> Result<Vector, Box<dyn Error>> {
        //! Solves systems of euqations using CG for symmetric matrix.
        //!
        //! # Examples
        //!
        //! ```rust
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! use jm_math::prelude::{MSolver, Matrix, PreconType, Vector};
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! # let A = Matrix::from_rows([
        //! #     [5.0, 0.0, 1.0, 0.0],
        //! #     [0.0, 6.0, 2.0, 3.0],
        //! #     [1.0, 2.0, 9.0, 4.0],
        //! #     [0.0, 3.0, 4.0, 8.0],
        //! # ])?;
        //! # let b = Vector::from_iter([6.0, 11.0, 16.0, 15.0]);
        //!
        //! # // initialize MSolver
        //! let ms = MSolver::build(&A, &b)
        //!     .max_iter(100)
        //!     .preconditioner(PreconType::Jacobi)
        //!     .tolerance(1e-8)
        //!     .finish();
        //!
        //! // solve with CG
        //! let x = ms.CG()?;
        //! println!("{x:.4}");
        //!
        //! # Ok(())
        //! # }
        //! ```
        conjugate_gradient::solve(self)
    }
}

impl<'a> MSolverBuilder<'a> {
    pub fn max_iter(&mut self, iMax: usize) -> Self {
        //! Sets maximum iteration number for `MSolver`.
        self.ms.iMax = iMax;
        *self
    }

    pub fn tolerance(&mut self, tol: f64) -> Self {
        //! Sets tolerance for convergence.
        self.ms.tol = tol;
        *self
    }

    pub fn preconditioner(&mut self, M: PreconType) -> Self {
        //! Sets `Preconditioner`.
        self.ms.preconditioner = Some(M);

        *self
    }

    pub fn finish(&self) -> MSolver<'a> {
        //! Returns a built `MSolver`.
        self.ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msolver_test() -> Result<(), Box<dyn Error>> {
        let A = Matrix::from_rows([
            [1.0, 0.0, 2.0, 3.0],
            [4.0, 5.0, 0.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 9.0],
        ])?;
        let b = Vector::from_iter([6.0, 15.0, 15.0, 9.0]);

        // GMRES
        let m = 3;
        let ms = MSolver::build(&A, &b).finish();
        let x = ms.GMRES(m)?;
        let x_exact = Vector::from_iter([1.0, 1.0, 1.0, 1.0]);
        let e = x - x_exact;
        // println!("GMRES({}) iter: {}, residual: {:.4E}", m, ms.iter, ms.res);
        assert!(e.magnitude() < 10E-6);

        // HGMRES
        let x = ms.HGMRES(m)?;
        let x_exact = Vector::from_iter([1.0, 1.0, 1.0, 1.0]);
        let e = x - x_exact;
        // println!("GMRES({}) iter: {}, residual: {:.4E}", m, ms.iter, ms.res);
        assert!(e.magnitude() < 10E-6);

        Ok(())
    }

    #[test]
    fn msover_matrix_market_test() -> Result<(), Box<dyn Error>> {
        let A = Matrix::from_matrix_market(
            "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/fs_760_1.mtx",
        )?;
        // let A = Matrix::from_matrix_market(
        //     "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/bcsstk13.mtx",
        // )?;
        // let A = Matrix::from_matrix_market(
        //     "/Users/h1007185/workspace/Rust/jm_math/matrix_makert/e40r5000.mtx",
        // )?;
        let (_, n) = A.dim();
        let b = Vector::from_iter(vec![1.0; n]);
        let b = &A * b;
        let restart = 20;

        let ms = MSolver::build(&A, &b).finish();
        let x = ms.GMRES(restart)?;
        // println!("GMRES({}) iter: {}, residual: {}", restart, ms.iter, ms.res);
        println!("{:.4}", x.iter().sum::<f64>());

        let x = ms.HGMRES(restart)?;
        // println!(
        //     "HGMRES({}) iter: {}, residual: {}",
        //     restart, ms.iter, ms.res
        // );
        println!("{:.4}", x.iter().sum::<f64>());
        Ok(())
    }

    // #[test]
    // fn MSolverBuilder_test() -> Result<(), Box<dyn Error>> {
    //     let A = Matrix::from_rows([
    //         [1.0, 0.0, 2.0, 3.0],
    //         [4.0, 5.0, 0.0, 6.0],
    //         [0.0, 0.0, 7.0, 8.0],
    //         [0.0, 0.0, 0.0, 9.0],
    //     ])?;
    //     let b = Vector::from_iter([6.0, 15.0, 15.0, 9.0]);

    //     let mut ms = MSolverBuilder::build(&A, &b).finish();
    //     let x = ms.GMRES(3)?;
    //     print!("{x:.4}");

    //     Ok(())
    // }

    #[test]
    fn preconditioner_test() -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
