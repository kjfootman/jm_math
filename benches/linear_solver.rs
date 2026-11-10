#![allow(non_snake_case)]

use divan::Bencher;
use jm_lib::{
    linear_algebra::msolver::{GmresBuilder, HgmresBuilder},
    prelude::*,
};

fn main() {
    divan::main();
}

fn get_source(M: &CSRMatrix) -> Vector {
    let ia = M.row_ptr();
    let aa = M.values();

    let arr = ia
        .windows(2)
        .map(|range| {
            let start = range[0] as usize;
            let end = range[1] as usize;

            aa[start..end].iter().sum::<f64>()
        })
        .collect::<Vec<_>>();

    Vector::from(arr)
}

#[divan::bench(sample_count = 5, sample_size = 1, args=["resources/mtx/bcsstk16.mtx", "resources/mtx/bcsstk18.mtx"])]
fn gmres(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            let M = CSRMatrix::from_mtx(path).unwrap();
            let f = get_source(&M);
            let x = Vector::from(vec![0.0; M.rows()]);

            (M, f, x)
        })
        .bench_values(|(M, f, mut x)| {
            let pc = pc::Jacobi::new(&M)
                .inspect_err(|e| println!("{e}"))
                .unwrap();
            let mut gmres = GmresBuilder::new()
                .with_max_iter(500)
                .with_restart(20)
                .with_tolerance(1E-12)
                .build();
            gmres.solve(&M, &pc, &f, &mut x).unwrap();
        });
}

#[divan::bench(sample_count = 5, sample_size = 1, args=["resources/mtx/bcsstk16.mtx", "resources/mtx/bcsstk18.mtx"])]
fn hgmres(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            let M = CSRMatrix::from_mtx(path).unwrap();
            let f = get_source(&M);
            let x = Vector::from(vec![0.0; M.rows()]);

            (M, f, x)
        })
        .bench_values(|(M, f, mut x)| {
            let pc = pc::Jacobi::new(&M).unwrap();
            let mut hgmres = HgmresBuilder::new()
                .with_max_iter(500)
                .with_restart(20)
                .with_tolerance(1E-12)
                .build();
            hgmres.solve(&M, &pc, &f, &mut x).unwrap();
        });
}
