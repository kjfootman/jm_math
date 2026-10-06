use crate::{
    error::Error,
    linear_algebra::{
        matrix::csr::CSRMatrix,
        preconditioner::{PreconType, Preconditioner},
        simd,
    },
};

pub struct Jacobi {
    // inverse values of the diagonal elements.
    ty: PreconType,
    diag_values: Vec<f64>,
}

impl Jacobi {
    pub fn new(matrix: &CSRMatrix) -> Result<Self, Error> {
        let diag_ptr = matrix
            .diag_ptr()
            .ok_or_else(|| Error::ValueError("Not initialized dia_ptr".to_string()))?;
        let values = matrix.values();

        let diag_values = diag_ptr
            .iter()
            .map(|&idx| values[idx as usize].recip())
            .collect();

        Ok(Jacobi {
            ty: PreconType::Jacobi("with Jacobi preconditioner"),
            diag_values,
        })
    }
}

// impl Default for Jacobi {
//     fn default() -> Self {
//         Self {
//             ty: PreconType::Jacobi("with Jacobi preconditioner"),
//             ..Default::default()
//         }
//     }
// }

impl Preconditioner for Jacobi {
    fn label(&self) -> &'static str {
        match self.ty {
            PreconType::Jacobi(lable) => lable,
            _ => "with Jacobi preconditioner",
        }
    }

    fn preconditioning(&self, v: &[f64], inplace: &mut [f64]) -> Result<(), Error> {
        // method1: without simd
        // inplace
        //     .iter_mut()
        //     .zip(v)
        //     .zip(self.diag_values.iter())
        //     .for_each(|((inplace, &v), &diag_value)| *inplace = v * diag_value);

        // method2: with auto simd
        let arch = simd::arch();
        arch.dispatch(|| {
            inplace
                .iter_mut()
                .zip(v)
                .zip(self.diag_values.iter())
                .for_each(|((inplace, &v), &diag_value)| *inplace = v * diag_value);
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Jacobi, Preconditioner};
    use crate::prelude::*;

    #[test]
    fn preconditioner_jacobi() -> Result<(), Error> {
        let (rows, cols) = (3, 3);
        let coordinates = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 0, 3.0),
            (1, 1, 4.0),
            (2, 2, 5.0),
        ];
        let M = CSRMatrix::from_coordinates(rows, cols, coordinates);
        let v = Vector::from(vec![1.0, 4.0, 5.0]);
        let mut z = Vector::new(v.len());
        let jacobi = Jacobi::new(&M)?;

        jacobi.preconditioning(&v, &mut z)?;

        assert_eq!(z, Vector::from(vec![1.0; 3]));

        Ok(())
    }
}
