use rayon::iter::*;
use std::{
    fmt::Display,
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg,
        Range, Sub, SubAssign,
    },
};

/// A struct for vector arithmetic.
///
/// The `Vector` struct is a wrapper for vector arithmetic.
/// It implements some trait, such as `Add`, `Sub`, `Mul`, ..., for vector arithmetics.
/// Because the `Vector` also implements `Deref` and `Index`, users can easily manipulate it like a `Vec`.
///
/// # `Vecter` initialization
///
/// An empty vector is defined using the [Vector::new()] method.
/// Then, some elements are added to it.
///
/// Another method to define a new vector is by using the [Vector::from_iter()] with a input array.
///
/// ```rust
/// use jm_math::prelude::Vector;
///
/// // Create a new empty Vector.
/// let mut v1 = Vector::new();
///
/// // Add some elements.
/// v1.push(1.0);
/// v1.push(2.0);
/// v1.push(3.0);
///
/// // Define another Vector from array.
/// let v2 = Vector::from_iter([1.0, 2.0, 3.0]);
/// assert_eq!(v1, v2);
/// ```
///
/// # `Vector` arithmetic
///
/// ```rust
/// use jm_math::prelude::Vector;
///
/// let v1 = Vector::from_iter([1.0, -2.0]);
/// let v2 = Vector::from_iter([2.0, 1.0]);
///
/// // addition
/// assert_eq!(&v1 + &v2, Vector::from_iter([3.0, -1.0]));
///
/// // subtraction
/// assert_eq!(&v1 - &v2, Vector::from_iter([-1.0, -3.0]));
///
/// // dot product
/// assert_eq!(&v1 * &v2, 0.0);
///
/// // scalar multiplication
/// assert_eq!(2.0 * &v1, Vector::from_iter([2.0, -4.0]));
///
/// // scalar division
/// assert_eq!(&v1 / 2.0, Vector::from_iter([0.5, -1.0]));
///
/// // negation
/// assert_eq!(-&v1, Vector::from_iter([-1.0, 2.0]));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vector {
    /// vector elements
    AA: Vec<f64>,
}

impl Vector {
    pub fn new() -> Self {
        //! Returns an empty `Vector`.

        Vector { AA: Vec::new() }
    }

    pub fn dim(&self) -> usize {
        //! Returns the dimension of a `Vector`.
        //!
        //! # Examples
        //!
        //! ```
        //! # use jm_math::prelude::Vector;
        //! let v = Vector::from_iter([1.0, 2.0, 3.0]);
        //! assert_eq!(v.dim(), 3);
        //! ```

        self.AA.len()
    }

    pub fn magnitude(&self) -> f64 {
        //! Returns the magnitude of a `Vector`.
        //!
        //! # Examples
        //!
        //! ```
        //! # use jm_math::prelude::Vector;
        //! let v = Vector::from_iter([3.0, 4.0]);
        //! assert_eq!(v.magnitude(), 5.0);
        //! ```

        let squared_sum: f64 = self.AA.par_iter().map(|v| v * v).sum();

        squared_sum.sqrt()
    }

    pub fn unit_vec(&self) -> Vector {
        //! Returns the unit vector of a `Vector`.
        //!
        //! # Examples
        //!
        //! ```
        //! # use jm_math::prelude::Vector;
        //! let v = Vector::from_iter([3.0, 4.0]);
        //!
        //! // unit vector
        //! let unit = v.unit_vec();
        //!
        //! // error vector
        //! let e = unit - Vector::from_iter([0.6, 0.8]);
        //! assert!(e.magnitude() < 10E-5);
        //! ```

        let mag = 1.0 / self.magnitude();
        let AA = self.AA.par_iter().map(|value| value * mag).collect();

        Self { AA }
    }
}

impl<'a> FromParallelIterator<&'a f64> for Vector {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = &'a f64>,
    {
        let AA = par_iter.into_par_iter().copied().collect();

        Self { AA }
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let AA = iter.into_iter().collect(); //* zero time cost */
        Self { AA }
    }
}

impl Deref for Vector {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.AA
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.AA
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);
        let precision = f.precision().unwrap_or(2);

        for (idx, value) in self.AA.iter().enumerate() {
            writeln!(
                f,
                "v[{idx}] = {value:>width$.precision$}",
                width = width,
                precision = precision
            )?;
        }

        Ok(())
    }
}

// impl PartialEq for Vector {
//     fn eq(&self, other: &Self) -> bool {
//         self.AA == other.AA
//     }
// }

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        // &self.AA[index]

        match self.AA.get(index) {
            Some(value) => value,
            None => {
                let err_msg = format!(
                    "Index out of bounds. The size is {}, but the index is {}",
                    self.dim(),
                    index
                );
                log::error!("{err_msg}");
                panic!();
            }
        }
    }
}

impl Index<Range<usize>> for Vector {
    type Output = [f64];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.AA[index.start..index.end]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let dim = self.dim();

        // self.AA.get_mut(index).unwrap_or_else(|| {
        //     let err_msg = format!(
        //         "Index out of bounds.\n
        //         The lenght of a vector: {}\n
        //         The index is {}.",
        //         dim, index
        //     );
        //     panic!("{}", err_msg);
        // })

        match self.AA.get_mut(index) {
            Some(value) => value,
            None => {
                let err_msg = format!(
                    "Index out of bounds. The size is {}, but the index is {}",
                    dim, index
                );
                log::error!("{err_msg}");
                panic!();
            }
        }
    }
}

impl IndexMut<Range<usize>> for Vector {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.AA[index.start..index.end]
    }
}

impl AsRef<[f64]> for Vector {
    fn as_ref(&self) -> &[f64] {
        &self.AA
    }
}

// impl std::borrow::Borrow<[f64]> for Vector {
//     fn borrow(&self) -> &[f64] {
//         &self.AA
//     }
// }

macro_rules! impl_add {
    ($($t:ty),*) => {
        $(
            impl<T: AsRef<[f64]>> Add<T> for $t {
                type Output = Vector;

                fn add(self, rhs: T) -> Self::Output {
                    let rhs = rhs.as_ref();

                    // dimension check
                    if self.dim() != rhs.len() {
                        let err_msg = "The dimensions of the two vectors for addition are not equal.";
                        log::error!("{err_msg}");
                        panic!();
                    }

                    let AA = self.AA.par_iter().enumerate()
                        .map(|(i, value)| value + rhs[i])
                        .collect::<Vec<_>>();

                    Vector {
                        AA
                    }
                }
            }
        )*
    };
}

impl_add!(Vector, &Vector);

impl<T: AsRef<[f64]>> AddAssign<T> for Vector {
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.as_ref();

        // dimension check
        if self.dim() != rhs.len() {
            let err_msg = "The dimensions of the two vectors for add_assign are not equal.";
            log::error!("{err_msg}");
            panic!();
        }

        let AA = self
            .AA
            .par_iter()
            .enumerate()
            .map(|(i, value)| value + rhs[i])
            .collect::<Vec<_>>();

        *self = Self { AA }
    }
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<T: AsRef<[f64]>> Sub<T> for $t {
                type Output = Vector;

                fn sub(self, rhs: T) -> Self::Output {
                    let rhs = rhs.as_ref();

                    // dimension check
                    if self.dim() != rhs.len() {
                        let err_msg = "The dimensions of the two vectors for subtraction are not equal.";
                        log::error!("{err_msg}");
                        panic!();
                    }

                    let AA = self.AA.par_iter().enumerate()
                        .map(|(i, value)| value - rhs[i])
                        .collect::<Vec<_>>();

                    Vector {
                        AA
                    }
                }
            }
        )*
    };
}

impl_sub!(Vector, &Vector);

impl<T: AsRef<[f64]>> SubAssign<T> for Vector {
    fn sub_assign(&mut self, rhs: T) {
        let rhs = rhs.as_ref();

        // dimension check
        if self.dim() != rhs.len() {
            let err_msg = "The dimensions of the two vectors for sub_assign are not equal.";
            log::error!("{err_msg}");
            panic!();
        }

        let AA = self
            .AA
            .par_iter()
            .enumerate()
            .map(|(i, value)| value - rhs[i])
            .collect::<Vec<_>>();

        *self = Self { AA }
    }
}

macro_rules! impl_dot {
    ($($t:ty),*) => {
        $(
            impl<T: AsRef<[f64]>> Mul<T> for $t {
                type Output = f64;

                fn mul(self, rhs: T) -> Self::Output {
                    let rhs = rhs.as_ref();

                    // dimension check
                    if self.dim() != rhs.len() {
                        // let err_msg = "The dimensions of the two vectors for dot_product are not equal.";
                        let err_msg = format!(
                            "The dimensions of the two vectors for dot_product are not equal. \
                            left: {} right: {}",
                            self.dim(),
                            rhs.len()
                        );
                        log::error!("{err_msg}");
                        panic!();
                    }

                    self.AA.par_iter().enumerate()
                        .map(|(i, value)| value * rhs[i])
                        .sum()
                }
            }
        )*
    };
}

impl_dot!(Vector, &Vector);

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(
            impl Mul<$t> for f64 {
                type Output = Vector;

                fn mul(self, rhs: $t) -> Self::Output {
                    let AA = rhs.AA.par_iter()
                        .map(|value| self * value)
                        .collect::<Vec<_>>();

                    Vector {
                        AA
                    }
                }
            }
        )*
    };
}

impl_mul!(Vector, &Vector);

impl MulAssign<f64> for Vector {
    fn mul_assign(&mut self, rhs: f64) {
        let AA = self
            .AA
            .par_iter()
            .map(|value| rhs * value)
            .collect::<Vec<_>>();

        *self = Self { AA }
    }
}

macro_rules! impl_div {
    ($($t:ty),*) => {
        $(
            impl Div<f64> for $t {
                type Output = Vector;

                fn div(self, rhs: f64) -> Self::Output {
                    let AA = self.AA.par_iter()
                        .map(|value| value / rhs)
                        .collect::<Vec<_>>();

                    Vector {
                        AA
                    }
                }
            }
        )*
    };
}

impl_div!(Vector, &Vector);

impl DivAssign<f64> for Vector {
    fn div_assign(&mut self, rhs: f64) {
        let AA = self
            .AA
            .par_iter()
            .map(|value| value / rhs)
            .collect::<Vec<_>>();

        *self = Self { AA }
    }
}

macro_rules! impl_neg {
    ($($t:ty),*) => {
        $(
            impl Neg for $t {
                type Output = Vector;

                fn neg(self) -> Self::Output {
                    let AA = self.AA.par_iter()
                        .map(|value| -value)
                        .collect::<Vec<_>>();

                    Vector {
                        AA
                    }
                }
            }
        )*
    };
}

impl_neg!(Vector, &Vector);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_definition() {
        let v1 = Vector::from_iter([1.0, 2.0, 3.0]);
        let v2 = Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(v1, v2);
    }

    #[test]
    fn vector_calculation() {
        // addition
        let v1 = Vector::from_iter([1.0, 2.0, 3.0]);
        let v2 = Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(v1 + v2, Vector::from_iter([2.0, 4.0, 6.0]));

        // add_assign
        let mut v = Vector::from_iter([1.0, 2.0, 3.0]);
        v += Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(v, Vector::from_iter([2.0, 4.0, 6.0]));

        // subtraction
        let v1 = Vector::from_iter([1.0, 2.0, 3.0]);
        let v2 = Vector::from_iter([3.0, 2.0, 1.0]);

        assert_eq!(v2 - v1, Vector::from_iter([2.0, 0.0, -2.0]));

        // sub_assign
        let mut v = Vector::from_iter([1.0, 2.0, 3.0]);
        v -= Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(v, Vector::from_iter([0.0, 0.0, 0.0]));

        // dot product
        let v1 = Vector::from_iter([1.0, 2.0, 3.0]);
        let v2 = Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(&v1 * &v2, 14.0);
        assert_eq!(v1 * v2, 14.0);

        // multiplicatoin
        let v1 = Vector::from_iter([1.0, 2.0, 3.0]);
        let v2 = [1.0, 2.0, 3.0];

        assert_eq!(2.0 * &v1, Vector::from_iter([2.0, 4.0, 6.0]));
        assert_eq!(&v1 * v2, 14.0);
        assert_eq!(2.0 * v1, Vector::from_iter([2.0, 4.0, 6.0]));

        // mul_assign
        let mut v = Vector::from_iter([1.0, 2.0, 3.0]);
        for _i in 0..2 {
            v *= 2.0;
        }

        assert_eq!(v, Vector::from_iter([4.0, 8.0, 12.0]));

        // div
        let v = Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(&v / 2.0, Vector::from_iter([0.5, 1.0, 1.5]));
        assert_eq!(v / 2.0, Vector::from_iter([0.5, 1.0, 1.5]));

        // div_assign
        let mut v = Vector::from_iter([4.0, 8.0, 12.0]);
        for _i in 0..2 {
            v /= 2.0;
        }

        assert_eq!(v, Vector::from_iter([1.0, 2.0, 3.0]));

        // neg
        let v = Vector::from_iter([1.0, 2.0, 3.0]);

        assert_eq!(-v, Vector::from_iter([-1.0, -2.0, -3.0]));

        // maginitude
        let v = Vector::from_iter([3.0, 4.0]);

        assert_eq!(v.magnitude(), 5.0);
    }

    #[test]
    fn vector_manipulation() {
        let mut v = Vector::from_iter([1.0, 2.0, 3.0]);

        v[2] = 5.0;
        v.push(2.0);

        assert_eq!(v.dim(), 4);
        assert_eq!(v[2], 5.0);

        // unit vector
        let v = Vector::from_iter([3.0, 4.0]);
        let unit = v.unit_vec();
        let err = &unit - Vector::from_iter([0.6, 0.8]);

        let mut rms: f64 = err.par_iter().map(|value| value * value).sum();
        rms = (rms / err.len() as f64).sqrt();

        assert_eq!(unit.magnitude(), 1.0);
        assert!(rms < 10E-7);
    }

    #[test]
    #[ignore = "takes too much time"]
    fn parallel_test() {
        const BUF_SIZE: usize = 100_000_000;
        let mut start;
        let a = std::rc::Rc::new(vec![1f64; BUF_SIZE]);
        let b = a.clone();

        // single
        start = std::time::Instant::now();
        let _ = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();
        println!(
            "Elapsed time (single): {:.4} sec",
            start.elapsed().as_secs_f32()
        );

        // parallel
        start = std::time::Instant::now();
        let _ = a
            .par_iter()
            .zip(b.par_iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();
        println!(
            "Elapsed time (parallel): {:.4} sec",
            start.elapsed().as_secs_f32()
        );
    }

    #[test]
    #[ignore]
    fn vector_initialize_Test() {
        const BUF_SIZE: usize = 500_000_000;
        let mut start;
        let mut AA = Vector::from_iter(vec![0f64; BUF_SIZE]);

        start = std::time::Instant::now();
        AA.fill(0f64);
        println!(
            "Elapsed time (fill 0): {:.4} sec",
            start.elapsed().as_secs_f32()
        );

        start = std::time::Instant::now();
        Vector::from_iter(vec![0f64; BUF_SIZE]);
        println!(
            "Elapsed time (re-define): {:.4} sec",
            start.elapsed().as_secs_f32()
        );
    }

    #[test]
    fn vector_index_test() {
        let v = Vector::from_iter([0.0, 1.0, 2.0]);
        assert_eq!(&v[0..2], &[0.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn vect0r_index_fail() {
        initit_log();

        // a vector for test
        let v = Vector::from_iter([0.0, 1.0, 2.0]);

        // a panic occurs
        println!("{}", v[100]);
    }

    #[test]
    #[should_panic]
    fn vector_dim_fail() {
        initit_log();

        let v1 = Vector::from_iter([1.0, 2.0]);
        let v2 = Vector::from_iter([2.0, 1.0, 0.0]);

        // a panic occurs
        let v = v1 + v2;
        println!("{v}");
    }

    fn initit_log() {
        //! Initializes the log4rs.
        log4rs::init_file("log4rs.yaml", Default::default()).unwrap_or_else(|err| {
            panic!("{err}");
        });
    }
}
