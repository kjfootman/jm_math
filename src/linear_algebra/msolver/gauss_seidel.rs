use crate::msolver::{CSRMatrix, MSolver, Vector};

#[derive(Debug)]
pub struct GaussSeidel {
    residual: f64,
    tolerance: f64,
    iter_max: u32,
    iter: u32,
}

pub struct GaussSeidelBuilder {
    tolerance: Option<f64>,
    iter_max: Option<u32>,
}

impl Default for GaussSeidel {
    fn default() -> Self {
        GaussSeidel {
            residual: f64::MAX,
            tolerance: 1E-7,
            iter_max: 500,
            iter: 0,
        }
    }
}

impl GaussSeidelBuilder {
    pub fn new() -> Self {
        Self {
            tolerance: None,
            iter_max: None,
        }
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    pub fn iter_max(mut self, iter_max: u32) -> Self {
        self.iter_max = Some(iter_max);
        self
    }

    pub fn build(self) -> GaussSeidel {
        GaussSeidel {
            tolerance: self.tolerance.unwrap_or_default(),
            iter_max: self.iter_max.unwrap_or_default(),
            ..Default::default()
        }
    }
}

impl MSolver for GaussSeidel {
    fn solve(&mut self, matrix: &CSRMatrix, b: &Vector) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msolver::MSolver;

    #[test]
    fn gauss_seidel_test() {
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let values = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let M = CSRMatrix::new(4, 4, row_ptr, col_indices, values);
        let b = Vector::from(vec![6.0, 15.0, 15.0, 9.0]);

        let mut gs = GaussSeidelBuilder::new()
            .iter_max(50)
            .tolerance(1E-5)
            .build();

        gs.solve(&M, &b);
    }
}
