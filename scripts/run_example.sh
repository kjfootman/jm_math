#!/bin/bash

# EXAMPLE="gauss_seidel_demo"
EXAMPLE="conjugate_gradient_demo"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

# RUST_LOG=jm_lib=debug cargo run -r --example "$EXAMPLE"
# RUSTFLAGS="-C target-cpu=native" RUST_LOG=info cargo run -r --example "$EXAMPLE"
RUST_LOG=info cargo run -r --example "$EXAMPLE"
