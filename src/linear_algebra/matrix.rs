use crate::linear_algebra::vector::Vector;
use crate::JmError;
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::{error::Error, fmt::Display, io::BufRead, ops::Mul};

#[cfg_attr(doc, katexit::katexit)]
/// A struct for matrix operation.
///
/// The `Matrix` is a struct for sparse matrix.
/// The <ins>C</ins>ompressed <ins>S</ins>parse <ins>R</ins>ow data structure is used.
/// In the following matrix, the CSR arrays are as follows:
///
/// $$
/// \begin{bmatrix}
///   1.0 & 0.0 & 1.0 \\newline
///   0.0 & 1.0 & 0.0 \\newline
///   1.0 & 0.0 & 1.0
/// \end{bmatrix}
/// $$
///
/// - ${\small IA}$: [0, 2, 3, 5]
/// - ${\small JA}$: [0, 2, 1, 0, 2]
/// - ${\small AA}$: [1.0, 1.0, 1.0, 1.0, 1.0]
///
/// The [`rayon`](https://docs.rs/rayon/latest/rayon/) was used for data parallelism.
///
/// # Initialization
/// There are several methods to initialize a `Matrix`.
/// - from an array of rows
/// - from an arrays of simple coordinates
/// - from CSR arrays
///
/// **Examples**
///
/// ```rust
/// # #![allow(non_snake_case)]
/// # use std::error::Error;
/// use jm_math::prelude::Matrix;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// // Initializes from an array of rows.
/// // 1.0    0.0
/// // 0.0    1.0
/// let mut M = Matrix::new();
/// M.append_row([1.0, 0.0])?;
/// M.append_row([0.0, 1.0])?;
/// println!("{M:.4}");
///
/// // Initializes from an array of simple coordinates.
/// // 1.0    0.0    0.0
/// // 0.0    1.0    0.0
/// // 0.0    0.0    1.0
/// let mut M = Matrix::from_simple_coordinate([
///     (0, 0, 1.0),
///     (1, 1, 1.0),
///     (2, 2, 1.0)
/// ])?;
/// println!("{M:.4}");
///
/// // Initializes from CSR arrays.
/// // 1.0    0.0    0.0    0.0
/// // 0.0    1.0    0.0    0.0
/// // 0.0    0.0    1.0    0.0
/// // 0.0    0.0    0.0    1.0
/// let IA = vec![0, 1, 2, 3, 4];
/// let JA = vec![0, 1, 2, 3];
/// let AA = vec![1.0, 1.0, 1.0, 1.0];
/// let M = Matrix::from_csr(IA, JA, AA)?;
/// println!("{M:.4}");
///
/// # Ok(())
/// # }
/// ```
///
/// # Matrix operations
///
/// The `Matrix` struct provides some operations
/// - matrix - vector operation (dot product)
/// - scalar - matrix multiplication
///
/// **Examples**
///
/// ```rust
/// # #![allow(non_snake_case)]
/// # use std::error::Error;
/// use jm_math::prelude::{Vector, Matrix};
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let v = Vector::from_iter([1.0, 1.0, 1.0]);
///
/// let mut M = Matrix::new();
/// M.append_row([ 1.0, 0.0, 0.0 ])?;
/// M.append_row([ 0.0, 2.0, 0.0 ])?;
/// M.append_row([ 0.0, 0.0, 3.0 ])?;
///
/// let N = M.clone();
///
/// // dot product
/// assert_eq!(&M * &v, Vector::from_iter([1.0, 2.0, 3.0]));
///
/// // the ownership of Matrix M moves
/// assert_eq!(M * &v, Vector::from_iter([1.0, 2.0, 3.0]));
///
/// // scalar multiplication
/// assert_eq!(2.0 * &N, Matrix::from_simple_coordinate([
///     (0, 0, 2.0),
///     (1, 1, 4.0),
///     (2, 2, 6.0),
/// ])?);
///
/// // the ownership of Matrix N moves
/// assert_eq!(2.0 * N, Matrix::from_simple_coordinate([
///     (0, 0, 2.0),
///     (1, 1, 4.0),
///     (2, 2, 6.0),
/// ])?);
///
/// # Ok(())
/// # }
///
/// ```

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matrix {
    m: usize,
    n: usize,
    IA: Vec<usize>,
    JA: Vec<usize>,
    AA: Vec<f64>,
}

impl Matrix {
    pub fn new() -> Matrix {
        //! Returns an empty `Matrix`.

        Matrix {
            m: 0,
            n: 0,
            IA: vec![0],
            JA: Vec::new(),
            AA: Vec::new(),
        }
    }

    pub fn AA(&self) -> &Vec<f64> {
        //! Returns an array of non-zero values for a `Matrix`.
        &self.AA
    }

    pub fn JA(&self) -> &Vec<usize> {
        //! Returns an array of column numbers for non-zero components.
        &self.JA
    }

    pub fn IA(&self) -> &Vec<usize> {
        //! Returns an array of pointers to the first non-zero component in each row
        &self.IA
    }

    pub fn append_row<T: AsRef<[f64]>>(&mut self, row: T) -> Result<(), Box<dyn Error>> {
        //! Appends a array to the last row.
        //!
        //! # Error
        //!
        //! When the size of a input array is not equal to the number of columns.
        //!
        //! # Examples
        //!
        //! ```
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! # use jm_math::prelude::Matrix;
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! let mut M = Matrix::from_simple_coordinate([
        //!     (0, 0, 1.0),
        //!     (1, 1, 1.0)
        //! ])?;
        //! let row = [3.0, 3.0];
        //!
        //! M.append_row(row)?;
        //! println!("{M:.2}");
        //!
        //! # Ok(())
        //! # }
        //! ```

        let row = row.as_ref();
        let n = self.n;

        // check dimension
        if n != row.len() && n != 0 {
            let err_msg = format!(
                "Failed to append a new row. \
                The number of columns is {}, but the length of the row is {}",
                n,
                row.len()
            );
            let err = JmError::DimError(err_msg);
            return Err(Box::new(err));
        }

        // set demension
        self.m += 1;
        if n == 0 {
            // to append a row to an empty matrix.
            self.n = row.len();
        }

        let IA = &mut self.IA;
        let JA = &mut self.JA;
        let AA = &mut self.AA;

        // to allocate additional memory for the new row.
        let non_zero_count = row.iter().filter(|value| 0f64.ne(*value)).count();
        JA.reserve(non_zero_count);
        AA.reserve(non_zero_count);

        row.iter().enumerate().for_each(|(ja, value)| {
            if !0f64.eq(value) {
                JA.push(ja);
                AA.push(*value);
            }
        });

        IA.push(AA.len());

        Ok(())
    }

    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        //! Returns the component at the indices (i, j).
        //!
        //! # Error
        //!
        //! When the indices (i, j) are out of bounds.
        //!
        //! # Examples
        //!
        //! ```
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! # use jm_math::prelude::Matrix;
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! let M = Matrix::from_simple_coordinate([
        //!     (0, 0, 1.0),
        //!     (1, 1, 1.0)
        //! ])?;
        //!
        //! assert_eq!(M.get(0, 0), Some(1.0));
        //! assert_eq!(M.get(1, 0), Some(0.0));
        //! assert_eq!(M.get(2, 1), None);
        //! # Ok(())
        //! # }
        //! ```

        let (m, n) = self.dim();

        // check dimensions
        if i >= m || j >= n {
            log::error!(
                "Index out of bounds. \
                The dimensions of matrix is ({m}, {n}), \
                but indices is ({i}, {j})"
            );
            return None;
        }

        let IA = &self.IA;
        let JA = &self.JA;
        let AA = &self.AA;

        match JA[IA[i]..IA[i + 1]].binary_search(&j) {
            Ok(ja) => Some(AA[ja]),
            Err(_) => {
                log::warn!(
                    "Failed to find the column number of {j} at the row of {i}.\
                    Returns 0.0",
                );
                Some(0.0)
            }
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        //! Returns the dimensions of a Matrix.
        //!
        //! # Examples
        //!
        //! ```
        //! # #![allow(non_snake_case)]
        //! // 3 x 3 identity matrix
        //! # use std::error::Error;
        //! # use jm_math::prelude::Matrix;
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! let M = Matrix::from_simple_coordinate([
        //!     (0, 0, 1.0),
        //!     (1, 1, 1.0),
        //!     (2, 2, 1.0)
        //! ])?;
        //!
        //! // m: the number of rows
        //! // n: the number of columns
        //! let (m, n) = M.dim();
        //!
        //! assert_eq!(m, 3);
        //! assert_eq!(n, 3);
        //! # Ok(())
        //! # }
        //! ```

        (self.m, self.n)
    }

    pub fn from_simple_coordinate<T: AsMut<[(usize, usize, f64)]>>(
        mut iter: T,
    ) -> Result<Matrix, Box<dyn Error>> {
        //! Returns the `Matrix` constructed from an array of simple coordinates.
        //!
        //! # Error
        //! When the input array is empty.
        //!
        //! # Example
        //!
        //! ```
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! # use jm_math::prelude::Matrix;
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! let M = Matrix::from_simple_coordinate([
        //!     (0, 0, 1.0),
        //!     (1, 1, 1.0)
        //! ])?;
        //!
        //! println!("{M:.2}");
        //! # Ok(())
        //! # }
        //! ```

        let elements = iter.as_mut();

        // sort each coordinate by its row and column indices
        elements.par_sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // unzip each column
        let (I, (JA, AA)): (Vec<_>, (Vec<_>, Vec<_>)) = elements
            .par_iter()
            .map(|(i, j, value)| (*i, (*j, *value)))
            .unzip();

        let mut pre_value = usize::MAX;
        let mut IA = Vec::with_capacity(I.len());

        I.iter().enumerate().for_each(|(i, value)| {
            if pre_value != *value {
                pre_value = *value;
                IA.push(i);
            }
        });
        IA.push(AA.len());

        let m = IA.len() - 1;
        let n = get_max_value(&JA)? + 1;
        let matrix = Matrix { m, n, IA, JA, AA };

        Ok(matrix)
    }

    pub fn from_rows<U, T>(rows: T) -> Result<Matrix, Box<dyn Error>>
    where
        U: IntoIterator<Item = f64>,
        T: IntoIterator<Item = U>,
    {
        //! Returns a Matrix constructed from an array of rows.
        //!
        //! # Error
        //! When every component of a row is zero.
        //!
        //! # Examples
        //!
        //! ```
        //! # #![allow(non_snake_case)]
        //! # use std::error::Error;
        //! # use jm_math::prelude::Matrix;
        //! # fn main() -> Result<(), Box<dyn Error>> {
        //! let M = Matrix::from_rows([
        //!     [1.0, 0.0, 0.0],
        //!     [0.0, 1.0, 0.0],
        //!     [0.0, 0.0, 1.0]
        //! ])?;
        //!
        //! println!("{M:.2}");
        //! # Ok(())
        //! # }
        //! ```

        let mut IA = Vec::new();
        let mut JA = Vec::new();
        let mut AA = Vec::new();
        let mut ia_counter = 0;

        IA.push(0);

        for row in rows {
            let mut non_zero_count = 0;
            for (j, value) in row.into_iter().enumerate() {
                if !0f64.eq(&value) {
                    JA.push(j);
                    AA.push(value);
                    non_zero_count += 1;
                }
            }
            ia_counter += non_zero_count;
            IA.push(ia_counter);
        }

        let m = IA.len() - 1;
        let n = get_max_value(&JA)? + 1;

        Ok(Matrix { m, n, IA, JA, AA })
    }

    pub fn from_csr<S, T, U>(IA: S, JA: T, AA: U) -> Result<Matrix, Box<dyn Error>>
    where
        S: IntoIterator<Item = usize>,
        T: IntoIterator<Item = usize>,
        U: IntoIterator<Item = f64>,
    {
        //! Returns a `Matrix` constructed from CSR arrays.

        let IA = IA.into_iter().collect::<Vec<_>>();
        let JA = JA.into_iter().collect::<Vec<_>>();
        let AA = AA.into_iter().collect::<Vec<_>>();

        let m = IA.len() - 1;
        let n = get_max_value(&JA)? + 1;

        Ok(Matrix { m, n, IA, JA, AA })
    }

    pub fn from_matrix_market(path: &str) -> Result<Matrix, Box<dyn Error>> {
        //! Reads a matrix market format and returns a `Matrix` constructed from it.

        let path = std::path::absolute(path)?;
        let f = std::fs::File::open(&path)?;
        let reader = std::io::BufReader::new(f);
        let (mut row, mut column, mut value);
        let mut line_num = 0;
        let mut coord = Vec::new();

        log::info!("Import '{}'.", path.display());

        for line in reader.lines().map_while(Result::ok) {
            if line.contains("%") {
                // skip header
                continue;
            }

            let arr = line.split_whitespace().collect::<Vec<_>>();

            if arr.len() != 3 {
                continue;
            }

            if line_num == 0 {
                let m = arr[0].parse::<usize>()?;
                let n = arr[1].parse::<usize>()?;
                let entries = arr[2].parse::<usize>()?;

                println!("rows: {m}, n: {n}, entries: {entries}");

                line_num += 1;
                continue;
            }

            row = arr[0].parse::<usize>()?;
            column = arr[1].parse::<usize>()?;
            value = arr[2].parse::<f64>()?;
            coord.push((row - 1, column - 1, value));

            line_num += 1;
        }

        Matrix::from_simple_coordinate(coord)
    }

    pub fn get_ptr_diagonal(&self) -> Result<Vec<usize>, Box<dyn Error>> {
        //! Returns a array of pointer to diagonal components.
        //!
        //! # Error
        //!
        //! If at least one of the diagonal component is zero then an `DimErr` is returned.

        let (m, _) = self.dim();
        let IA = &self.IA;
        let JA = &self.JA;

        let uptr = (0..m)
            .into_par_iter()
            .filter_map(|i| {
                let start = IA[i];
                let end = IA[i + 1];
                // `JA` 배열에서 start와 end 사이에 i가 있는지 확인
                JA[start..end]
                    .iter()
                    .position(|&ja| ja == i)
                    .map(|pos| start + pos)
            })
            .collect::<Vec<_>>();

        if uptr.len() != m {
            let err_msg = String::from("Failed to find the pointer to diagonal components.");
            let err = JmError::DimError(err_msg);
            return Err(Box::new(err));
        }

        Ok(uptr)
    }

    fn get_depth(&self) -> Option<Vec<usize>> {
        //! Returns a array of depth for each each component.

        let m = self.m;
        let mut depth = vec![0; m];
        let IA = &self.IA;
        let JA = &self.JA;

        for i in 0..m {
            depth[i] = JA[IA[i]..IA[i + 1]].iter().map(|ja| depth[*ja]).max()? + 1;
        }

        Some(depth)
    }

    pub fn level_scheduling(&self) -> Result<(Vec<usize>, Vec<usize>), Box<dyn Error>> {
        //! Return a tuple of arrays for level scheduling.
        //! (lev, permuatation)

        let depth = self
            .get_depth()
            .ok_or("Failed to get the array of depth.")?;

        // set a permutation array
        let mut q = (0..depth.len()).collect::<Vec<_>>();
        q.par_sort_by(|i, j| depth[*i].cmp(&depth[*j]));

        // set a array of level
        let chunk_sizes = q
            .par_chunk_by(|i, j| depth[*i] == depth[*j])
            .map(|chunk| chunk.len())
            .collect::<Vec<_>>();

        //? parallelization needed?
        let level = std::iter::once(0)
            .chain(chunk_sizes.iter().scan(0, |acc, &size| {
                *acc += size;
                Some(*acc)
            }))
            .collect();

        Ok((level, q))
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let IA = &self.IA;
        let AA = &self.AA;
        let JA = &self.JA;

        let width = f.width().unwrap_or(0);
        let precision = f.precision().unwrap_or(2);
        let (m, n) = self.dim();

        for i in 0..m {
            let mut row = vec![0.0; n];

            for j in IA[i]..IA[i + 1] {
                row[JA[j]] = AA[j];
            }

            row.iter().take(n).for_each(|value| {
                write!(
                    f,
                    "{:>width$.precision$} ",
                    value,
                    width = width,
                    precision = precision
                )
                .ok();
            });

            writeln!(f)?;
        }

        Ok(())
    }
}

macro_rules! impl_mat_vec_mul {
    ($($t:ty),*) => {
        $(
            #[allow(clippy::suspicious_arithmetic_impl)]
            impl<T: AsRef<[f64]>> Mul<T> for $t {
                type Output = Vector;

                fn mul(self, rhs: T) -> Self::Output {
                    let (m, n) = self.dim();
                    let IA = &self.IA;
                    let JA = &self.JA;
                    let AA = &self.AA;
                    let rhs = rhs.as_ref();

                    if n != rhs.len() {
                        // todo: error should be handled properly.
                        let err_msg = format!(
                            "Matrix vector multiplication failed.\n
                            The number of columns: {}\n,
                            The lenghth of a vector: {}",
                            m,
                            n
                        );
                        panic!("{err_msg}");
                    }

                    let AA = (0..m).into_par_iter().map(|i| {
                        let mut sum = 0.0;

                        for ia in IA[i]..IA[i+1] {
                            sum += AA[ia] * rhs[JA[ia]];
                        }
                        sum
                    }).collect::<Vec<_>>();

                    Vector::from_iter(AA)
                }
            }
        )*
    };
}

impl_mat_vec_mul!(Matrix, &Matrix);

impl Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        let IA = rhs.IA.to_owned();
        let JA = rhs.JA.to_owned();
        let AA = rhs
            .AA
            .par_iter()
            .map(|&value| self * value)
            .collect::<Vec<_>>();

        Matrix {
            m: rhs.m,
            n: rhs.n,
            IA,
            JA,
            AA,
        }
    }
}

impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        let AA = rhs
            .AA
            .par_iter()
            .map(|&value| self * value)
            .collect::<Vec<_>>();

        Matrix { AA, ..rhs }
    }
}

fn get_max_value(JA: &[usize]) -> Result<usize, &str> {
    //! Returns the maximum column number.
    JA.par_iter()
        .max()
        .copied()
        .ok_or("Failed to get the maximum column index.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_definition() -> Result<(), Box<dyn Error>> {
        let m = Matrix::from_simple_coordinate(vec![
            (0usize, 0usize, 1.0),
            (0usize, 1usize, 2.0),
            (1usize, 0usize, 4.0),
            (1usize, 1usize, 5.0),
            (1usize, 2usize, 6.0),
            (2usize, 0usize, 7.0),
            (2usize, 2usize, 9.0),
        ])?;
        println!("{m:.3}");

        let m = Matrix::from_csr([0, 1, 2, 3], [0, 1, 2], [1.0, 1.0, 1.0])?;
        println!("{m:.3}");

        Ok(())
    }

    #[test]
    fn matrix_arithmetics() -> Result<(), Box<dyn Error>> {
        let M = Matrix::from_simple_coordinate([
            (0, 0, 1.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ])?;

        let v = Vector::from_iter([1.0, 2.0, 3.0]);

        // 1.0 1.0 1.0     1.0 .   6.0
        // 0.0 1.0 0.0  *  2.0  = .2.0
        // 0.0 0.0 1.0     3.0     3.0
        assert_eq!(&M * v, Vector::from_iter([6.0, 2.0, 3.0]));

        Ok(())
    }

    #[test]
    fn append_row_test() -> Result<(), Box<dyn Error>> {
        let mut M = Matrix::from_simple_coordinate([
            (0, 0, 1.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ])?;
        // let mut M = Matrix::new();
        M.append_row([1.0, 0.0, 0.0])?;
        println!("{M:.4}");

        Ok(())
    }

    #[test]
    fn get_uptr_test() -> Result<(), Box<dyn Error>> {
        let IA = [0, 2, 4, 6, 7];
        let JA = [0, 1, 1, 2, 0, 2, 3];
        let AA = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        // let IA = [0, 2, 3, 5, 6];
        // let JA = [0, 1, 2, 0, 2, 3];
        // let AA = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0];

        let M = Matrix::from_csr(IA, JA, AA)?;
        let uptr = M.get_ptr_diagonal()?;

        assert_eq!(uptr, vec![0, 2, 5, 6]);

        Ok(())
    }
}
