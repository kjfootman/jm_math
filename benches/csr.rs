#![allow(non_snake_case)]

use divan::Bencher;
use jm_lib::prelude::*;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 20, sample_size = 5, args=["resources/mtx/e40r5000.mtx", "resources/mtx/bcsstk18.mtx"])]
fn spmv_bench(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            // let path = "resources/mtx/e40r5000.mtx";
            // let path = "resources/mtx/bcsstk18.mtx";
            let M = CSRMatrix::from_mtx(path).unwrap();
            let v = Vector::from(vec![1.0; M.cols()]);
            let result = Vector::new(M.cols());

            (M, v, result)
        })
        .bench_values(|(M, v, mut result)| {
            result.csr_spmv(&M, &v).unwrap();
        });
}

#[divan::bench(sample_count = 20, sample_size = 5, args=["resources/mtx/e40r5000.mtx", "resources/mtx/bcsstk18.mtx"])]
fn spmv2_bench(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            // let path = "resources/mtx/e40r5000.mtx";
            // let path = "resources/mtx/bcsstk18.mtx";
            let M = CSRMatrix::from_mtx(path).unwrap();
            let v = Vector::from(vec![1.0; M.cols()]);
            let result = Vector::new(M.cols());

            (M, v, result)
        })
        .bench_values(|(M, v, mut result)| {
            result.csr_spmv2(&M, &v).unwrap();
        });
}
