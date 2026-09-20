#![allow(non_snake_case)]

use divan::Bencher;
use jm_lib::prelude::*;

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
fn vector_dot_product_bench(bencher: Bencher, N: usize) {
    bencher
        .with_inputs(|| {
            let v1 = Vector::from(vec![1.0; N]);
            let v2 = Vector::from(vec![1.0; N]);

            (N, v1, v2)
        })
        .bench_values(|(N, v1, v2)| {
            let result = v1.dot(&v2).unwrap();
            assert_eq!(N as f64, result);
        });
}

#[divan::bench(args=[20_000, 40_000, 80_000])]
fn vector_add_bench(bencher: Bencher, N: usize) {
    bencher
        .with_inputs(|| {
            // const N: usize = 30_000;
            let v1 = Vector::from(vec![1.0; N]);
            let v2 = Vector::from(vec![1.0; N]);
            let out = Vector::from(vec![0.0; N]);

            (v1, v2, out)
        })
        .bench_values(|(v1, v2, mut out)| {
            out.add(&v1, &v2).unwrap();
        });
}

#[divan::bench(sample_count = 20, sample_size = 5, args=["resources/mtx/bcsstk18.mtx"])]
fn vector_calc_residual_bench(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            let M = CSRMatrix::from_mtx(path).unwrap();
            let rows = M.rows();
            let b = get_source(&M);
            let x = Vector::from(vec![1.0; rows]);
            let r = Vector::from(vec![0.0; rows]);
            let tmp = Vector::from(vec![0.0; rows]);

            (r, b, M, x, tmp)
        })
        .bench_values(|(mut r, b, M, x, mut tmp)| {
            tmp.csr_spmv2(&M, &x).unwrap();
            r.sub(&b, &tmp).unwrap();
        });
}

// #[divan::bench(sample_count = 20, sample_size = 5, args=["resources/mtx/e40r5000.mtx", "resources/mtx/bcsstk18.mtx"])]
#[divan::bench(sample_count = 20, sample_size = 5, args=["resources/mtx/bcsstk18.mtx"])]
fn vector_calc_residual2_bench(bencher: Bencher, path: &str) {
    bencher
        .with_inputs(|| {
            let M = CSRMatrix::from_mtx(path).unwrap();
            let rows = M.rows();
            let b = get_source(&M);
            let x = Vector::from(vec![1.0; rows]);
            let r = Vector::from(vec![0.0; rows]);

            (r, b, M, x)
        })
        .bench_values(|(mut r, b, M, x)| {
            r.calc_residual(&b, &M, &x).unwrap();
        });
}
