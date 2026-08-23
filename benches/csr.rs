use divan::Bencher;
use jm_lib::prelude::*;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 5, sample_size = 1)]
fn test2(b: Bencher) {
    b.with_inputs(|| {
        // Prepare dataset for bench
        let path = "resources/mtx/e40r5000.mtx";
        CSRMatrix::from_mtx(path).unwrap()
    })
    .bench_refs(|matrix| {
        let cll = matrix.col_indices();
    });
}
