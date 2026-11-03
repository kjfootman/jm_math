#!/bin/bash

# TEST_NAME="vector_from_mtx"
# TEST_NAME="csr_diagonal"
# TEST_NAME="gauss_seidel"
# TEST_NAME="vector_spmv"
# TEST_NAME="conjugate_gradient"
TEST_NAME="gmres"
# TEST_NAME="hgmres"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

RUST_LOG=jm_lib=debug cargo test "$TEST_NAME" -- --nocapture
