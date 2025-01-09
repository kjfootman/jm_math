//! module for linear algebra.
//!
//! The `linear_algebra` module provides various structs
//! and functions for performing linear algebra operations.
//!
//! The `Matrix` and `Vector` structs are the primary data structures
//! for matrix and vector arithmetic.
//! And the `MSolver` struct is used to compute solutions for systems of equations.
//!
//! ```rust
#![doc = include_str!("../examples/linear_algebra.rs")]
//! ```

mod matrix;
mod msolver;
mod vector;

pub use matrix::Matrix;
pub use msolver::{MSolver, PreconType};
pub use vector::Vector;

pub(in crate::linear_algebra) fn cal_Householder_vec<T: AsRef<[f64]>>(v: T) -> Vector {
    //! Calcuate Householder vector.
    let v = v.as_ref().to_vec(); //* copy occurs */
    let mut u = Vector::from_iter(v); //* zero time cost */
    let mag = u.magnitude();

    if let Some(value) = u.get_mut(0) {
        *value += value.signum() * mag;
    }

    u.unit_vec()
}

pub(in crate::linear_algebra) fn Givens_rotation(H: &mut [Vec<f64>], g: &mut [f64]) {
    //! Applies Givens rotation to Hessenberg matrix and vector.
    let n = H.len();

    for i in 0..n {
        let r = H[i][i];
        let h = H[i][i + 1];
        let l = (r.powi(2) + h.powi(2)).sqrt();
        let c = r / l;
        let s = -h / l;

        // m << n
        for col in H.iter_mut().take(n).skip(i) {
            let hji = col[i];

            col[i] = col[i] * c - col[i + 1] * s;
            col[i + 1] = hji * s + col[i + 1] * c;
        }

        g[i + 1] = s * g[i];
        g[i] *= c;
    }
}

pub(in crate::linear_algebra) fn solve_upper_triangular(U: &[Vec<f64>], g: &[f64]) -> Vec<f64> {
    //! Solves an upper triangular system of equations.
    let n = U.len();
    let mut y = vec![0f64; n];

    for i in (0..n).rev() {
        let mut sum = g[i];
        for j in i + 1..n {
            sum -= U[j][i] * y[j];
        }
        y[i] = sum / U[i][i];
    }

    y
}

#[cfg(test)]
mod test {
    use super::*;

    // todo: test for Givens_rotation

    #[test]
    fn upper_triangular_solve_test() {
        let H = [
            vec![1.0, 0.0, 0.0],
            vec![2.0, 4.0, 0.0],
            vec![3.0, 5.0, 6.0],
        ];
        let g = [14.0, 23.0, 18.0];
        let x_exact = Vector::from_iter([1.0, 2.0, 3.0]);

        let x = Vector::from_iter(solve_upper_triangular(&H, &g));
        let err = (x - x_exact).magnitude();

        assert!(err < 1.0E-5);
    }
}
