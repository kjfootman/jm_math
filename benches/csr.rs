// use divan::Bencher;
use jm_lib::prelude::*;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 5, sample_size = 1)]
fn read_file_collect() {}
