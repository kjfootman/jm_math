use crate::msolver::{CSRMatrix, MSolver, Vector};
use std::cell::RefCell;

#[derive(Debug)]
pub struct GaussSeidel {
    residual: RefCell<f64>,
    tolerance: f64,
    iter_max: usize,
    iter: RefCell<usize>,
}

pub struct GaussSeidelBuilder {
    tolerance: Option<f64>,
    iter_max: Option<usize>,
}

impl GaussSeidel {
    pub fn iter(&self) -> usize {
        self.iter.take()
    }

    pub fn residual(&self) -> f64 {
        self.residual.take()
    }
}

impl Default for GaussSeidel {
    fn default() -> Self {
        GaussSeidel {
            residual: RefCell::new(f64::MAX),
            tolerance: 1E-7,
            iter_max: 500,
            iter: RefCell::new(0),
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

    pub fn iter_max(mut self, iter_max: usize) -> Self {
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
    fn solve(&self, matrix: &CSRMatrix, b: &Vector) {
        let mut iter = self.iter.borrow_mut();

        for _ in 0..10 {
            *iter += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msolver::MSolver;

    #[test]
    fn gauss_seidel_test() {
        let (rows, cols) = (4, 4);
        let row_ptr = vec![0, 3, 6, 8, 9];
        let col_indices = vec![0, 2, 3, 0, 1, 3, 2, 3, 3];
        let values = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let M = CSRMatrix::new(rows, cols, row_ptr, col_indices, values);
        let b = Vector::from(vec![6.0, 15.0, 15.0, 9.0]);

        let gs = GaussSeidelBuilder::new()
            .iter_max(50)
            .tolerance(1E-5)
            .build();

        gs.solve(&M, &b);

        println!("iter: {}, residual: {:.2E}", gs.iter(), gs.residual());
    }
}
