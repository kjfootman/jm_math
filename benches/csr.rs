// use divan::Bencher;
use jm_lib::prelude::*;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 5, sample_size = 1)]
fn read_file_iter() {
    let path = "resources/mtx/e40r5000.mtx";
    // let tmp = read_mtx_iter(path).unwrap();
}
