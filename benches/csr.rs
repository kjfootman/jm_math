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

#[divan::bench(args=[20_000, 40_000, 80_000])]
fn dot_product_bench(bencher: Bencher, N: usize) {
    bencher
        .with_inputs(|| {
            // const N: usize = 30_000;
            let v1 = Vector::from(vec![1.0; N]);
            let v2 = Vector::from(vec![1.0; N]);

            (N, v1, v2)
        })
        .bench_values(|(N, v1, v2)| {
            let result = v1.dot(&v2).unwrap();
            assert_eq!(N as f64, result);
        });
}
