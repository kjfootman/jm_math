#!/bin/bash

TEST_NAME="vector_from_mtx_test"
# TEST_NAME="csr_diagonal_test"
# TEST_NAME="gauss_seidel_test"
# TEST_NAME="vector_spmv"
# TEST_NAME="conjugate_gradient_test"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

RUST_LOG=jm_lib=debug cargo test -r "$TEST_NAME" -- --nocapture
