#!/bin/bash
path="/Users/h1007185/workspace/Rust/jm_math"

cd $path &&
cargo clean --doc &&
# RUSTDOCFLAGS="--html-in-header $path/src/docs-header.html" cargo doc -r --no-deps --open --examples
cargo doc -r --no-deps --open --examples